import pandas as pd
import numpy as np
from typing import Dict, Any
import joblib
from pathlib import Path
import pandas_ta as ta
from sklearn.ensemble import GradientBoostingClassifier
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

class MLBoostSignal:
    """ML Boost 전략 신호 생성"""
    
    def __init__(self, 
                 model_path: str = None,
                 min_holding_periods: int = 3,
                 max_holding_periods: int = 24,
                 stop_loss_pct: float = 0.008,   # 손절폭 강화
                 take_profit_pct: float = 0.045, # 익절폭 강화
                 volume_threshold: float = 1.25): # 거래량 조건 강화
        self.model = None
        self.min_holding_periods = min_holding_periods
        self.max_holding_periods = max_holding_periods
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.volume_threshold = volume_threshold
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            # 기본 모델 생성 - 파라미터 재조정
            self.model = GradientBoostingClassifier(
                n_estimators=450,      # 트리 개수 증가
                learning_rate=0.015,   # 학습률 감소
                max_depth=8,           # 트리 깊이 유지
                min_samples_split=5,   # 분할 기준 유지
                min_samples_leaf=2,    # 잎 노드 기준 유지
                subsample=0.8,
                random_state=42
            )
            
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특성 생성"""
        try:
            # 기본 데이터 복사
            df = df.copy()
            
            # 결측치 처리
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 가격 변화율
            df['returns'] = df['close'].pct_change()
            df['returns_5'] = df['close'].pct_change(5)
            df['returns_10'] = df['close'].pct_change(10)
            df['returns_std'] = df['returns'].rolling(20).std()
            
            # 기술적 지표 계산
            df['ema_20'] = ta.ema(df['close'], length=20)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_200'] = ta.ema(df['close'], length=200)
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['rsi_ma'] = ta.sma(df['rsi'], length=5)
            
            # 볼린저 밴드
            bb = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_ratio'] = df['atr'] / df['close']
            df['atr_ma'] = ta.sma(df['atr'], length=5)
            
            # 변동성
            df['volatility'] = df['returns'].rolling(20).std()
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
            df['volatility_ma'] = ta.sma(df['volatility'], length=5)
            
            # 거래량 지표
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_trend'] = df['volume'].pct_change(5)
            df['volume_std'] = df['volume'].rolling(20).std()
            
            # 추세 지표
            df['trend'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
            df['trend_strength'] = abs(df['trend'])
            df['trend_ma'] = df['trend'].rolling(10).mean()
            df['trend_std'] = df['trend'].rolling(20).std()
            
            # 모멘텀 지표
            df['momentum'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_ma'] = df['momentum'].rolling(5).mean()
            df['momentum_std'] = df['momentum'].rolling(20).std()
            
            # 가격 범위
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['price_range_ma'] = df['price_range'].rolling(10).mean()
            df['price_range_std'] = df['price_range'].rolling(20).std()
            
            # 추가 특성
            df['price_momentum'] = df['close'].pct_change(5)
            df['volume_price_trend'] = df['volume_ratio'] * df['price_momentum']
            df['rsi_trend'] = df['rsi'].diff(5)
            df['price_volatility'] = df['returns'].rolling(10).std()
            
            # 결측치 처리
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"특성 생성 중 오류 발생: {e}")
            return df
            
    def _train_model(self, df: pd.DataFrame) -> None:
        """모델 학습"""
        try:
            # 특성 선택 - 개선된 특성 세트
            features = [
                'rsi', 'rsi_ma', 'bb_width', 'bb_position',
                'atr_ratio', 'atr_ma', 'volatility_ratio',
                'volume_ratio', 'volume_trend', 'trend_strength',
                'momentum', 'price_range', 'volume_price_trend',
                'rsi_trend', 'price_volatility'
            ]
            
            # 특성 엔지니어링
            X = df[features].copy()
            
            # 결측치 처리
            X = X.ffill().bfill()
            
            # 이상치 처리
            for col in X.columns:
                q1 = X[col].quantile(0.01)
                q3 = X[col].quantile(0.99)
                X[col] = X[col].clip(q1, q3)
            
            # 특성 스케일링
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 레이블 생성 (1기간 미래 수익률)
            y = (df['close'].shift(-1) / df['close'] - 1) > 0.002
            y = y.fillna(False)
            
            # 학습/검증 데이터 분할
            train_size = int(len(X_scaled) * 0.8)
            X_train = X_scaled[:train_size]
            y_train = y[:train_size]
            X_val = X_scaled[train_size:]
            y_val = y[train_size:]
            
            # 모델 파라미터
            params = {
                'n_estimators': 500,
                'max_depth': 7,
                'min_samples_split': 8,
                'min_samples_leaf': 4,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'random_state': 42
            }
            
            # 모델 학습
            self.model = GradientBoostingClassifier(**params)
            self.model.fit(X_train, y_train)
            
            # 검증 성능 평가
            val_pred = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred)
            logger.info(f"Validation AUC: {val_auc:.4f}")
            
            # 특성 중요도 저장
            self.feature_importance = dict(zip(features, self.model.feature_importances_))
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            raise

    def _generate_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """매매 신호 생성"""
        try:
            # 기본 데이터 복사
            df = df.copy()
            
            # 결측치 처리 - 새로운 방식 사용
            df = df.ffill().bfill()
            
            # 기술적 지표 계산
            df = self.calculate_indicators(df)
            
            # 특성 선택
            features = [
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower',
                'atr', 'adx', 'cci', 'mfi', 'obv',
                'vwap', 'vwap_std', 'volume_ma_ratio',
                'price_volatility', 'volume_volatility'
            ]
            
            # 특성 엔지니어링
            X = df[features].copy()
            
            # 결측치 처리 - 새로운 방식 사용
            X = X.ffill().bfill()
            
            # 이상치 처리
            for col in X.columns:
                if col != 'volume':
                    q1 = X[col].quantile(0.01)
                    q3 = X[col].quantile(0.99)
                    X[col] = X[col].clip(q1, q3)
            
            # 특성 스케일링
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 학습 데이터 준비
            y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
            
            # 학습/검증 데이터 분할
            train_size = int(len(X_scaled) * 0.8)
            X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # 모델 학습
            self.model.fit(X_train, y_train)
            
            # 검증 데이터 예측
            val_pred = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred)
            logger.info(f"Validation AUC: {val_auc:.4f}")
            
            # 전체 데이터 예측
            predictions = self.model.predict_proba(X_scaled)[:, 1]
            
            # 리스크 관리 파라미터 재조정
            max_position_size = 0.12   # 포지션 크기 축소
            stop_loss_atr = 0.7        # ATR 기반 손절 기준 강화
            take_profit_atr = 4.0      # ATR 기반 익절 기준 강화
            max_holding_periods = 8     # 최대 보유 시간 축소
            
            # 신호 초기화
            df['entry_signal'] = 0
            df['exit_signal'] = 0
            df['position'] = 0
            
            # 포지션 관리
            current_position = 0
            entry_price = 0
            entry_time = None
            
            for i in range(len(df)):
                if i < 20:  # 초기 데이터 스킵
                    continue
                    
                current_price = df['close'].iloc[i]
                current_atr = df['atr'].iloc[i]
                current_time = df.index[i]
                
                # 진입 신호
                if current_position == 0:
                    # 추가 진입 조건 재조정
                    volume_condition = df['volume'].iloc[i] > df['volume'].rolling(20).mean().iloc[i] * self.volume_threshold
                    trend_condition = df['adx'].iloc[i] > 18  # ADX 조건 강화
                    volatility_condition = df['atr'].iloc[i] < df['atr'].rolling(20).mean().iloc[i] * 1.1  # 변동성 조건 강화
                    
                    if predictions[i] > 0.74 and volume_condition and trend_condition and volatility_condition:  # 롱 진입 임계값 강화
                        current_position = 1
                        entry_price = current_price
                        entry_time = current_time
                        df.loc[current_time, 'entry_signal'] = 1
                    elif predictions[i] < 0.26 and volume_condition and trend_condition and volatility_condition:  # 숏 진입 임계값 강화
                        current_position = -1
                        entry_price = current_price
                        entry_time = current_time
                        df.loc[current_time, 'entry_signal'] = -1
                        
                # 청산 신호
                elif current_position != 0:
                    # 최대 보유 시간 체크
                    holding_periods = (current_time - entry_time).total_seconds() / 3600
                    if holding_periods > max_holding_periods:
                        df.loc[current_time, 'exit_signal'] = 1
                        current_position = 0
                        entry_price = 0
                        entry_time = None
                        continue
                    
                    # 손절/익절 체크
                    if current_position == 1:  # 롱 포지션
                        stop_loss = entry_price - (current_atr * stop_loss_atr)
                        take_profit = entry_price + (current_atr * take_profit_atr)
                        
                        if current_price <= stop_loss or current_price >= take_profit:
                            df.loc[current_time, 'exit_signal'] = 1
                            current_position = 0
                            entry_price = 0
                            entry_time = None
                            
                    else:  # 숏 포지션
                        stop_loss = entry_price + (current_atr * stop_loss_atr)
                        take_profit = entry_price - (current_atr * take_profit_atr)
                        
                        if current_price >= stop_loss or current_price <= take_profit:
                            df.loc[current_time, 'exit_signal'] = 1
                            current_position = 0
                            entry_price = 0
                            entry_time = None
                            
                df.loc[current_time, 'position'] = current_position
                
            return df
            
        except Exception as e:
            logger.error(f"신호 생성 중 오류 발생: {str(e)}")
            raise
        
    def load_model(self, path: str):
        """모델 로드"""
        try:
            self.model = joblib.load(path)
            logger.info(f"모델 로드 완료: {path}")
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
        
    def save_model(self, path: str):
        """모델 저장"""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            joblib.dump(self.model, path)
            logger.info(f"모델 저장 완료: {path}")
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {e}")
            raise

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        try:
            # RSI
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'])
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            
            # 볼린저 밴드
            bb = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            
            # ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # ADX
            df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
            
            # CCI
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
            
            # MFI - 직접 구현
            try:
                # 데이터 타입 변환
                high = df['high'].astype('float64')
                low = df['low'].astype('float64')
                close = df['close'].astype('float64')
                volume = df['volume'].astype('float64')
                
                # 전형가격 계산
                typical_price = (high + low + close) / 3
                
                # 자금 흐름 계산
                money_flow = typical_price * volume
                
                # 양수/음수 자금 흐름 계산
                price_diff = typical_price.diff()
                positive_flow = pd.Series(0.0, index=df.index)
                negative_flow = pd.Series(0.0, index=df.index)
                
                positive_flow[price_diff > 0] = money_flow[price_diff > 0]
                negative_flow[price_diff < 0] = money_flow[price_diff < 0]
                
                # 14일 이동평균
                positive_mf = positive_flow.rolling(window=14).sum()
                negative_mf = negative_flow.rolling(window=14).sum()
                
                # MFI 계산
                mfi = 100 - (100 / (1 + positive_mf / negative_mf))
                df['mfi'] = mfi.fillna(50.0)  # 결측값은 50으로 대체
                
            except Exception as e:
                logger.warning(f"MFI 계산 중 오류 발생: {str(e)}")
                df['mfi'] = 50.0  # 기본값 설정
            
            # OBV
            df['obv'] = ta.obv(df['close'], df['volume'])
            
            # VWAP
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            df['vwap_std'] = df['vwap'].rolling(20).std()
            
            # 거래량 지표
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # 변동성 지표
            df['price_volatility'] = df['close'].pct_change().rolling(20).std()
            df['volume_volatility'] = df['volume'].pct_change().rolling(20).std()
            
            return df
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 중 오류 발생: {str(e)}")
            raise 