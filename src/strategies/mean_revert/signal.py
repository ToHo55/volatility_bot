import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Tuple, List
from loguru import logger
from enum import Enum
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class MarketState(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class BaseStrategy:
    """기본 전략 클래스"""
    def __init__(self):
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.holding_periods = 0

    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, float]:
        """진입 여부와 포지션 사이즈 반환"""
        return False, 0.0

    def should_exit(self, df: pd.DataFrame, current_idx: int) -> bool:
        """청산 여부 반환"""
        return False

class TrendFollowingStrategy(BaseStrategy):
    """추세 추종 전략"""
    def __init__(self):
        super().__init__()
        self.adx_threshold = 25
        self.ema_fast = 10
        self.ema_slow = 20
        self.stop_loss_pct = 0.01
        self.take_profit_pct = 0.03

    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, float]:
        if self.position != 0:
            return False, 0.0

        # ADX로 추세 강도 확인
        adx = df['adx'].iloc[current_idx]
        if adx < self.adx_threshold:
            return False, 0.0

        # EMA 크로스오버 확인
        ema_fast = ta.ema(df['close'], length=self.ema_fast).iloc[current_idx]
        ema_slow = ta.ema(df['close'], length=self.ema_slow).iloc[current_idx]
        ema_fast_prev = ta.ema(df['close'], length=self.ema_fast).iloc[current_idx-1]
        ema_slow_prev = ta.ema(df['close'], length=self.ema_slow).iloc[current_idx-1]

        # 롱 진입
        if ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow:
            self.position = 1
            self.entry_price = df['close'].iloc[current_idx]
            return True, 0.1  # 계좌의 10%

        # 숏 진입
        elif ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow:
            self.position = -1
            self.entry_price = df['close'].iloc[current_idx]
            return True, 0.1  # 계좌의 10%

        return False, 0.0

    def should_exit(self, df: pd.DataFrame, current_idx: int) -> bool:
        if self.position == 0:
            return False

        current_price = df['close'].iloc[current_idx]
        
        # 손절/익절 체크
        if self.position == 1:  # 롱 포지션
            if (current_price <= self.entry_price * (1 - self.stop_loss_pct) or
                current_price >= self.entry_price * (1 + self.take_profit_pct)):
                return True
        else:  # 숏 포지션
            if (current_price >= self.entry_price * (1 + self.stop_loss_pct) or
                current_price <= self.entry_price * (1 - self.take_profit_pct)):
                return True

        return False

class MeanRevertingStrategy(BaseStrategy):
    """평균 회귀 전략"""
    def __init__(self):
        super().__init__()
        self.z_score_threshold = 1.0
        self.rsi_oversold = 40
        self.rsi_overbought = 60
        self.stop_loss_pct = 0.006
        self.take_profit_pct = 0.018

    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, float]:
        if self.position != 0:
            return False, 0.0

        z_score = df['z_score'].iloc[current_idx]
        rsi = df['rsi'].iloc[current_idx]
        volume_ratio = df['volume_ratio'].iloc[current_idx]

        # 롱 진입
        if (z_score < -self.z_score_threshold and 
            rsi < self.rsi_oversold and 
            volume_ratio > 0.7):
            self.position = 1
            self.entry_price = df['close'].iloc[current_idx]
            return True, 0.05  # 계좌의 5%

        # 숏 진입
        elif (z_score > self.z_score_threshold and 
              rsi > self.rsi_overbought and 
              volume_ratio > 0.7):
            self.position = -1
            self.entry_price = df['close'].iloc[current_idx]
            return True, 0.05  # 계좌의 5%

        return False, 0.0

    def should_exit(self, df: pd.DataFrame, current_idx: int) -> bool:
        if self.position == 0:
            return False

        current_price = df['close'].iloc[current_idx]
        z_score = df['z_score'].iloc[current_idx]
        
        # 손절/익절 체크
        if self.position == 1:  # 롱 포지션
            if (current_price <= self.entry_price * (1 - self.stop_loss_pct) or
                current_price >= self.entry_price * (1 + self.take_profit_pct) or
                z_score > -0.2):
                return True
        else:  # 숏 포지션
            if (current_price >= self.entry_price * (1 + self.stop_loss_pct) or
                current_price <= self.entry_price * (1 - self.take_profit_pct) or
                z_score < 0.2):
                return True

        return False

class VolatilityBreakoutStrategy(BaseStrategy):
    """변동성 돌파 전략"""
    def __init__(self):
        super().__init__()
        self.atr_period = 14
        self.atr_multiplier = 1.5
        self.stop_loss_pct = 0.008
        self.take_profit_pct = 0.024

    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, float]:
        if self.position != 0:
            return False, 0.0

        atr = df['atr'].iloc[current_idx]
        high = df['high'].iloc[current_idx]
        low = df['low'].iloc[current_idx]
        close = df['close'].iloc[current_idx]
        volume_ratio = df['volume_ratio'].iloc[current_idx]

        # 상향 돌파
        if (high > df['bb_upper'].iloc[current_idx] and 
            volume_ratio > 1.2):
            self.position = 1
            self.entry_price = close
            return True, 0.08  # 계좌의 8%

        # 하향 돌파
        elif (low < df['bb_lower'].iloc[current_idx] and 
              volume_ratio > 1.2):
            self.position = -1
            self.entry_price = close
            return True, 0.08  # 계좌의 8%

        return False, 0.0

    def should_exit(self, df: pd.DataFrame, current_idx: int) -> bool:
        if self.position == 0:
            return False

        current_price = df['close'].iloc[current_idx]
        
        # 손절/익절 체크
        if self.position == 1:  # 롱 포지션
            if (current_price <= self.entry_price * (1 - self.stop_loss_pct) or
                current_price >= self.entry_price * (1 + self.take_profit_pct)):
                return True
        else:  # 숏 포지션
            if (current_price >= self.entry_price * (1 + self.stop_loss_pct) or
                current_price <= self.entry_price * (1 - self.take_profit_pct)):
                return True

        return False

class MarketMetrics:
    """시장 지표 관리 클래스"""
    def __init__(self):
        self.fear_greed_index = None
        self.funding_rate = None
        self.open_interest = None
        self.exchange_flow = None
        self.last_update = None
        self.update_interval = timedelta(minutes=5)

    def update(self):
        """시장 지표 업데이트"""
        try:
            # TODO: 실제 API 연동 구현
            self.fear_greed_index = self._fetch_fear_greed_index()
            self.funding_rate = self._fetch_funding_rate()
            self.open_interest = self._fetch_open_interest()
            self.exchange_flow = self._fetch_exchange_flow()
            self.last_update = datetime.now()
        except Exception as e:
            logger.error(f"시장 지표 업데이트 중 오류 발생: {str(e)}")

    def _fetch_fear_greed_index(self) -> float:
        """공포/탐욕 지수 조회"""
        # TODO: 실제 API 연동
        return 50.0

    def _fetch_funding_rate(self) -> float:
        """선물 기초 조회"""
        # TODO: 실제 API 연동
        return 0.0

    def _fetch_open_interest(self) -> float:
        """미체결약정 조회"""
        # TODO: 실제 API 연동
        return 0.0

    def _fetch_exchange_flow(self) -> float:
        """거래소 유입/유출 조회"""
        # TODO: 실제 API 연동
        return 0.0

class StrategyPerformance:
    """전략별 성과 추적 클래스"""
    def __init__(self):
        self.trades = []
        self.metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0,
            'recent_profit': 0.0,  # 최근 10개 거래의 평균 수익
            'market_state_accuracy': 0.0  # 해당 전략이 적합한 시장 상황에서의 성과
        }
        self.last_update = None
        self.update_interval = timedelta(minutes=5)

    def update(self, trade_data: Dict):
        """거래 데이터 업데이트"""
        self.trades.append(trade_data)
        self._calculate_metrics()
        self.last_update = datetime.now()

    def _calculate_metrics(self):
        """성과 지표 계산"""
        if not self.trades:
            return

        # 최근 50개 거래만 사용
        recent_trades = self.trades[-50:]
        returns = [t['profit'] for t in recent_trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]

        # 승률
        self.metrics['win_rate'] = len(wins) / len(returns) if returns else 0.0

        # 수익 팩터
        if losses:
            self.metrics['profit_factor'] = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
        else:
            self.metrics['profit_factor'] = float('inf')

        # 샤프 비율
        if len(returns) > 1:
            self.metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)

        # 평균 수익
        self.metrics['avg_profit'] = np.mean(returns)

        # 최대 드로다운
        cumulative_returns = np.cumsum(returns)
        max_dd = 0
        peak = cumulative_returns[0]
        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        self.metrics['max_drawdown'] = max_dd

        # 최근 수익 (최근 10개 거래)
        recent_returns = returns[-10:] if len(returns) >= 10 else returns
        self.metrics['recent_profit'] = np.mean(recent_returns)

        # 시장 상황 정확도
        correct_market_states = sum(1 for t in recent_trades 
                                  if t['market_state'] == t['strategy_market_state'])
        self.metrics['market_state_accuracy'] = correct_market_states / len(recent_trades)

    def get_score(self) -> float:
        """전략 점수 계산"""
        if not self.trades:
            return 0.0

        # 각 지표별 가중치
        weights = {
            'win_rate': 0.2,
            'profit_factor': 0.2,
            'sharpe_ratio': 0.2,
            'avg_profit': 0.1,
            'max_drawdown': -0.1,  # 음수 가중치 (드로다운이 클수록 낮은 점수)
            'recent_profit': 0.15,
            'market_state_accuracy': 0.05
        }

        # 점수 계산
        score = 0.0
        for metric, weight in weights.items():
            value = self.metrics[metric]
            if metric == 'max_drawdown':
                score += value * weight  # 이미 음수 가중치
            else:
                score += value * weight

        return score

class StrategySelector:
    """전략 선택 클래스"""
    def __init__(self):
        self.strategies = {
            'trend': StrategyPerformance(),
            'mean_revert': StrategyPerformance(),
            'volatility': StrategyPerformance()
        }
        self.market_state_weights = {
            MarketState.TRENDING: {'trend': 0.5, 'mean_revert': 0.2, 'volatility': 0.3},
            MarketState.RANGING: {'trend': 0.2, 'mean_revert': 0.5, 'volatility': 0.3},
            MarketState.VOLATILE: {'trend': 0.3, 'mean_revert': 0.2, 'volatility': 0.5}
        }
        self.last_update = None
        self.update_interval = timedelta(minutes=5)
        self.last_switch_time = None
        self.switch_cooldown = timedelta(minutes=30)  # 30분 쿨다운

    def update(self, trade_data: Dict):
        """거래 데이터 업데이트"""
        strategy_name = trade_data['strategy_name']
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update(trade_data)
        self.last_update = datetime.now()

    def select_strategy(self, market_state: MarketState) -> str:
        """시장 상황에 따른 전략 선택"""
        if not any(s.trades for s in self.strategies.values()):
            # 초기에는 기본 가중치 사용
            return self._get_default_strategy(market_state)

        # 각 전략의 점수 계산
        strategy_scores = {
            name: perf.get_score() 
            for name, perf in self.strategies.items()
        }

        # 시장 상황별 가중치 적용
        market_weights = self.market_state_weights[market_state]
        weighted_scores = {
            name: score * market_weights[name]
            for name, score in strategy_scores.items()
        }

        # 최고 점수의 전략 선택
        return max(weighted_scores.items(), key=lambda x: x[1])[0]

    def _get_default_strategy(self, market_state: MarketState) -> str:
        """기본 전략 선택"""
        if market_state == MarketState.TRENDING:
            return 'trend'
        elif market_state == MarketState.RANGING:
            return 'mean_revert'
        else:
            return 'volatility'

    def should_switch_strategy(self, current_strategy: str, market_state: MarketState) -> bool:
        """전략 전환 필요 여부 확인"""
        # 쿨다운 체크
        if self.last_switch_time is not None:
            if datetime.now() - self.last_switch_time < self.switch_cooldown:
                return False

        if not any(s.trades for s in self.strategies.values()):
            return False

        # 현재 전략의 성과
        current_performance = self.strategies[current_strategy]
        current_score = current_performance.get_score()

        # 다른 전략들의 성과
        other_scores = {
            name: perf.get_score() * self.market_state_weights[market_state][name]
            for name, perf in self.strategies.items()
            if name != current_strategy
        }

        # 최고 점수의 전략
        best_strategy = max(other_scores.items(), key=lambda x: x[1])

        # 추가 리스크 체크
        if current_performance.metrics['max_drawdown'] > 0.1:  # 최대 드로다운이 10% 초과
            self.last_switch_time = datetime.now()
            return True
        
        if current_performance.metrics['recent_profit'] < -0.02:  # 최근 수익이 -2% 미만
            self.last_switch_time = datetime.now()
            return True

        # 현재 전략의 점수가 최고 점수의 80% 미만이면 전환
        should_switch = current_score < best_strategy[1] * 0.8
        if should_switch:
            self.last_switch_time = datetime.now()
        
        return should_switch

    def get_best_strategy(self, market_state: MarketState) -> str:
        """현재 시장 상황에서 최고 성과의 전략 반환"""
        if not any(s.trades for s in self.strategies.values()):
            return self._get_default_strategy(market_state)

        strategy_scores = {
            name: perf.get_score() * self.market_state_weights[market_state][name]
            for name, perf in self.strategies.items()
        }

        return max(strategy_scores.items(), key=lambda x: x[1])[0]

class MeanRevertSignal:
    """Mean Revert 전략 신호 생성"""
    
    def __init__(self,
                 lookback_period: int = 20,
                 entry_threshold: float = 1.0,  # 진입 임계값 더욱 완화
                 exit_threshold: float = 0.2,   # 청산 임계값 더욱 완화
                 stop_loss_pct: float = 0.006,  # 손절폭 축소
                 take_profit_pct: float = 0.018, # 익절폭 조정
                 volume_threshold: float = 0.8,  # 거래량 조건 더욱 완화
                 max_holding_periods: int = 4,   # 최대 보유 시간 단축
                 min_holding_periods: int = 1,   # 최소 보유 시간 단축
                 param_optimization_period: int = 30,  # 파라미터 최적화 기간 (일)
                 max_position_size: float = 0.05,  # 최대 포지션 크기 축소 (10% → 5%)
                 max_daily_loss: float = 0.01,    # 최대 일일 손실 축소 (2% → 1%)
                 max_drawdown: float = 0.03,     # 최대 드로다운 축소 (5% → 3%)
                 model_update_period: int = 7):  # 모델 업데이트 주기 (일)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.volume_threshold = volume_threshold
        self.max_holding_periods = max_holding_periods
        self.min_holding_periods = min_holding_periods
        self.param_optimization_period = param_optimization_period
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.model_update_period = model_update_period
        self.last_model_update = None
        
        # 시장 상황별 기본 파라미터
        self.market_params = {
            MarketState.TRENDING: {
                'entry_threshold': 1.5,
                'exit_threshold': 0.4,
                'stop_loss_pct': 0.01,
                'take_profit_pct': 0.03,
                'volume_threshold': 1.2
            },
            MarketState.RANGING: {
                'entry_threshold': 1.3,
                'exit_threshold': 0.3,
                'stop_loss_pct': 0.008,
                'take_profit_pct': 0.025,
                'volume_threshold': 1.05
            },
            MarketState.VOLATILE: {
                'entry_threshold': 1.1,
                'exit_threshold': 0.2,
                'stop_loss_pct': 0.006,
                'take_profit_pct': 0.02,
                'volume_threshold': 1.0
            }
        }
        
        # 최적화된 파라미터 저장
        self.optimized_params = None
        self.last_optimization_time = None
        
        # 리스크 레벨별 파라미터
        self.risk_params = {
            RiskLevel.LOW: {
                'position_size': 0.05,    # 계좌의 5%
                'stop_loss_pct': 0.006,   # 0.6%
                'take_profit_pct': 0.018  # 1.8%
            },
            RiskLevel.MEDIUM: {
                'position_size': 0.1,     # 계좌의 10%
                'stop_loss_pct': 0.008,   # 0.8%
                'take_profit_pct': 0.025  # 2.5%
            },
            RiskLevel.HIGH: {
                'position_size': 0.15,    # 계좌의 15%
                'stop_loss_pct': 0.01,    # 1.0%
                'take_profit_pct': 0.035  # 3.5%
            }
        }
        
        # 거래 통계 초기화
        self.trade_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'daily_pnl': 0.0,
            'last_trade_time': None
        }
        
        # 머신러닝 모델 초기화
        self.market_state_model = None
        self.entry_model = None
        self.exit_model = None
        self.risk_model = None
        
        # 스케일러 초기화
        self.feature_scaler = StandardScaler()
        
        # 모델 저장 경로
        self.model_dir = "models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # 시장 상황별 전략 초기화
        self.trend_strategy = TrendFollowingStrategy()
        self.mean_revert_strategy = MeanRevertingStrategy()
        self.volatility_strategy = VolatilityBreakoutStrategy()

        # 시장 지표 관리자 초기화
        self.market_metrics = MarketMetrics()
        
        # 전략 선택기 초기화
        self.strategy_selector = StrategySelector()

    def _classify_market_state(self, df: pd.DataFrame, current_idx: int) -> MarketState:
        """시장 상황 분류"""
        try:
            # ADX로 추세 강도 확인
            adx = df['adx'].iloc[current_idx]
            
            # 변동성 확인
            volatility = df['volatility'].iloc[current_idx]
            avg_volatility = df['volatility'].rolling(50).mean().iloc[current_idx]
            
            # 추세 확인
            sma = df['sma'].iloc[current_idx]
            price = df['close'].iloc[current_idx]
            price_sma_ratio = abs(price - sma) / sma
            
            # 시장 상황 분류 (조건 완화)
            if adx > 25 and price_sma_ratio > 0.015:  # 추세 조건 완화
                return MarketState.TRENDING
            elif volatility > avg_volatility * 1.3:  # 변동성 조건 완화
                return MarketState.VOLATILE
            else:  # 횡보장
                return MarketState.RANGING
                
        except Exception as e:
            logger.error(f"시장 상황 분류 중 오류 발생: {str(e)}")
            return MarketState.RANGING  # 기본값으로 횡보장 반환

    def _optimize_parameters(self, df: pd.DataFrame, current_idx: int) -> Dict[MarketState, Dict[str, float]]:
        """과거 데이터 기반 파라미터 최적화"""
        try:
            # 최적화 기간 데이터 추출
            start_idx = max(0, current_idx - self.param_optimization_period)
            optimization_data = df.iloc[start_idx:current_idx].copy()
            
            if len(optimization_data) < 20:  # 최소 데이터 필요
                return self.market_params
            
            optimized_params = {}
            
            for market_state in MarketState:
                # 해당 시장 상황의 데이터만 필터링
                state_data = optimization_data[optimization_data['market_state'] == market_state.value]
                
                if len(state_data) < 10:  # 최소 데이터 필요
                    optimized_params[market_state] = self.market_params[market_state]
                    continue
                
                # 수익성 기반 파라미터 최적화
                best_params = self._find_best_parameters(state_data)
                optimized_params[market_state] = best_params
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"파라미터 최적화 중 오류 발생: {str(e)}")
            return self.market_params

    def _find_best_parameters(self, data: pd.DataFrame) -> Dict[str, float]:
        """최적의 파라미터 찾기"""
        try:
            best_sharpe = -np.inf
            best_params = None
            
            # 파라미터 그리드 정의
            param_grid = {
                'entry_threshold': np.linspace(1.0, 2.0, 5),
                'exit_threshold': np.linspace(0.1, 0.5, 5),
                'stop_loss_pct': np.linspace(0.005, 0.015, 5),
                'take_profit_pct': np.linspace(0.015, 0.035, 5),
                'volume_threshold': np.linspace(0.9, 1.3, 5)
            }
            
            # 그리드 서치
            for entry in param_grid['entry_threshold']:
                for exit in param_grid['exit_threshold']:
                    for sl in param_grid['stop_loss_pct']:
                        for tp in param_grid['take_profit_pct']:
                            for vol in param_grid['volume_threshold']:
                                params = {
                                    'entry_threshold': entry,
                                    'exit_threshold': exit,
                                    'stop_loss_pct': sl,
                                    'take_profit_pct': tp,
                                    'volume_threshold': vol
                                }
                                
                                # 파라미터로 백테스트 수행
                                returns = self._backtest_with_params(data, params)
                                
                                if len(returns) > 0:
                                    sharpe = self._calculate_sharpe_ratio(returns)
                                    
                                    if sharpe > best_sharpe:
                                        best_sharpe = sharpe
                                        best_params = params
            
            return best_params if best_params is not None else self.market_params[MarketState.RANGING]
            
        except Exception as e:
            logger.error(f"최적 파라미터 찾기 중 오류 발생: {str(e)}")
            return self.market_params[MarketState.RANGING]

    def _backtest_with_params(self, data: pd.DataFrame, params: Dict[str, float]) -> List[float]:
        """주어진 파라미터로 백테스트 수행"""
        try:
            returns = []
            position = 0
            entry_price = 0
            
            for i in range(len(data)):
                if position == 0:  # 진입
                    if (data['z_score'].iloc[i] < -params['entry_threshold'] and 
                        data['rsi'].iloc[i] < 30 and
                        data['volume_ratio'].iloc[i] > params['volume_threshold']):
                        position = 1
                        entry_price = data['close'].iloc[i]
                    elif (data['z_score'].iloc[i] > params['entry_threshold'] and 
                          data['rsi'].iloc[i] > 70 and
                          data['volume_ratio'].iloc[i] > params['volume_threshold']):
                        position = -1
                        entry_price = data['close'].iloc[i]
                
                elif position != 0:  # 청산
                    current_price = data['close'].iloc[i]
                    
                    if position == 1:  # 롱 포지션
                        if (current_price <= entry_price * (1 - params['stop_loss_pct']) or
                            current_price >= entry_price * (1 + params['take_profit_pct']) or
                            data['z_score'].iloc[i] > -params['exit_threshold']):
                            returns.append((current_price - entry_price) / entry_price)
                            position = 0
                    
                    else:  # 숏 포지션
                        if (current_price >= entry_price * (1 + params['stop_loss_pct']) or
                            current_price <= entry_price * (1 - params['take_profit_pct']) or
                            data['z_score'].iloc[i] < params['exit_threshold']):
                            returns.append((entry_price - current_price) / entry_price)
                            position = 0
            
            return returns
            
        except Exception as e:
            logger.error(f"백테스트 중 오류 발생: {str(e)}")
            return []

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """샤프 비율 계산"""
        try:
            if not returns:
                return -np.inf
            
            returns = np.array(returns)
            return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # 연간화
            
        except Exception as e:
            logger.error(f"샤프 비율 계산 중 오류 발생: {str(e)}")
            return -np.inf

    def _should_optimize(self, current_time: datetime) -> bool:
        """파라미터 최적화 필요 여부 확인"""
        if self.last_optimization_time is None:
            return True
        
        time_diff = current_time - self.last_optimization_time
        return time_diff.days >= 1  # 매일 최적화

    def _get_market_params(self, market_state: MarketState, df: pd.DataFrame, current_idx: int) -> Dict[str, float]:
        """시장 상황별 파라미터 반환 (최적화 포함)"""
        current_time = df.index[current_idx]
        
        # 최적화 필요 여부 확인
        if self._should_optimize(current_time):
            self.optimized_params = self._optimize_parameters(df, current_idx)
            self.last_optimization_time = current_time
            logger.info(f"파라미터 최적화 완료: {current_time}")
        
        # 최적화된 파라미터가 있으면 사용, 없으면 기본 파라미터 사용
        if self.optimized_params is not None:
            return self.optimized_params[market_state]
        return self.market_params[market_state]

    def _calculate_position_size(self, df: pd.DataFrame, current_idx: int, market_state: MarketState) -> float:
        """포지션 사이즈 계산"""
        try:
            # 기본 리스크 레벨 결정
            risk_level = self._determine_risk_level(df, current_idx)
            base_position_size = self.risk_params[risk_level]['position_size']
            
            # 변동성 조정
            volatility = df['volatility'].iloc[current_idx]
            avg_volatility = df['volatility'].rolling(50).mean().iloc[current_idx]
            volatility_ratio = volatility / avg_volatility
            
            # 승률 기반 조정
            win_rate = self.trade_stats['winning_trades'] / max(1, self.trade_stats['total_trades'])
            win_rate_factor = min(1.5, max(0.5, win_rate / 0.5))  # 50% 기준
            
            # 최근 수익성 기반 조정
            recent_profit_factor = self._calculate_recent_profit_factor()
            
            # 최종 포지션 사이즈 계산
            position_size = base_position_size * (1 / volatility_ratio) * win_rate_factor * recent_profit_factor
            
            # 한도 적용
            position_size = min(position_size, self.max_position_size)
            
            # 일일 손실 한도 체크
            if self.trade_stats['daily_pnl'] < -self.max_daily_loss:
                position_size = 0
                logger.warning("일일 손실 한도 도달: 거래 중단")
            
            # 드로다운 한도 체크
            if self.trade_stats['current_drawdown'] > self.max_drawdown:
                position_size = 0
                logger.warning("최대 드로다운 한도 도달: 거래 중단")
            
            return position_size
            
        except Exception as e:
            logger.error(f"포지션 사이즈 계산 중 오류 발생: {str(e)}")
            return self.risk_params[RiskLevel.LOW]['position_size']

    def _determine_risk_level(self, df: pd.DataFrame, current_idx: int) -> RiskLevel:
        """리스크 레벨 결정"""
        try:
            # 변동성 기반
            volatility = df['volatility'].iloc[current_idx]
            avg_volatility = df['volatility'].rolling(50).mean().iloc[current_idx]
            
            # 승률 기반
            win_rate = self.trade_stats['winning_trades'] / max(1, self.trade_stats['total_trades'])
            
            # 최근 수익성 기반
            recent_profit = self._calculate_recent_profit_factor()
            
            # 리스크 점수 계산
            risk_score = 0
            
            # 변동성 점수
            if volatility > avg_volatility * 1.5:
                risk_score += 2
            elif volatility > avg_volatility:
                risk_score += 1
            
            # 승률 점수
            if win_rate > 0.6:
                risk_score += 2
            elif win_rate > 0.5:
                risk_score += 1
            
            # 수익성 점수
            if recent_profit > 1.2:
                risk_score += 2
            elif recent_profit > 1.0:
                risk_score += 1
            
            # 리스크 레벨 결정
            if risk_score >= 5:
                return RiskLevel.HIGH
            elif risk_score >= 3:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"리스크 레벨 결정 중 오류 발생: {str(e)}")
            return RiskLevel.LOW

    def _calculate_recent_profit_factor(self) -> float:
        """최근 수익성 계산"""
        try:
            if self.trade_stats['total_trades'] < 10:
                return 1.0
            
            recent_trades = min(10, self.trade_stats['total_trades'])
            recent_profit = self.trade_stats['total_profit'] / recent_trades
            
            if recent_profit > 0:
                return min(1.5, 1 + recent_profit)
            else:
                return max(0.5, 1 + recent_profit)
                
        except Exception as e:
            logger.error(f"최근 수익성 계산 중 오류 발생: {str(e)}")
            return 1.0

    def _update_trade_stats(self, profit: float, current_time: datetime):
        """거래 통계 업데이트"""
        try:
            # 일일 PnL 초기화
            if self.trade_stats['last_trade_time'] is None or \
               (current_time - self.trade_stats['last_trade_time']).days >= 1:
                self.trade_stats['daily_pnl'] = 0
            
            # 거래 통계 업데이트
            self.trade_stats['total_trades'] += 1
            self.trade_stats['daily_pnl'] += profit
            
            if profit > 0:
                self.trade_stats['winning_trades'] += 1
                self.trade_stats['total_profit'] += profit
            else:
                self.trade_stats['losing_trades'] += 1
                self.trade_stats['total_loss'] += abs(profit)
            
            # 드로다운 업데이트
            self.trade_stats['current_drawdown'] = min(0, self.trade_stats['daily_pnl'])
            self.trade_stats['max_drawdown'] = min(self.trade_stats['max_drawdown'], 
                                                 self.trade_stats['current_drawdown'])
            
            self.trade_stats['last_trade_time'] = current_time
            
        except Exception as e:
            logger.error(f"거래 통계 업데이트 중 오류 발생: {str(e)}")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특성 엔지니어링"""
        try:
            features = pd.DataFrame()
            
            # 기존 특성
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log1p(features['returns'])
            features['volatility'] = features['returns'].rolling(20).std()
            
            # 기술적 지표
            features['rsi'] = df['rsi']
            features['adx'] = df['adx']
            features['macd'] = df['macd']
            features['macd_signal'] = df['macd_signal']
            features['macd_hist'] = df['macd_hist']
            features['stoch_k'] = df['stoch_k']
            features['stoch_d'] = df['stoch_d']
            
            # 볼린저 밴드 관련
            features['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 거래량 관련
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['volume_ma_ratio'] = df['volume'] / df['volume_sma']
            
            # 추세 관련
            features['trend_strength'] = abs(df['close'] - df['sma']) / df['std']
            features['momentum'] = df['momentum']
            features['momentum_ma'] = df['momentum_ma']
            
            # 변동성 관련
            features['atr'] = df['atr']
            features['atr_ratio'] = df['atr'] / df['atr'].rolling(20).mean()
            
            # 시장 심리 지표
            features['fear_greed'] = self.market_metrics.fear_greed_index
            features['funding_rate'] = self.market_metrics.funding_rate
            features['open_interest'] = self.market_metrics.open_interest
            features['exchange_flow'] = self.market_metrics.exchange_flow
            
            # 시계열 특성
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            
            # 계절성 특성
            features['seasonal'] = np.sin(2 * np.pi * features['hour'] / 24)
            
            # 주기성 특성
            features['price_cycle'] = np.sin(2 * np.pi * np.arange(len(df)) / 20)
            
            # 결측치 처리
            features = features.ffill().bfill().fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"특성 엔지니어링 중 오류 발생: {str(e)}")
            return pd.DataFrame()

    def _prepare_labels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """레이블 준비"""
        try:
            # 시장 상황 레이블
            market_state_labels = pd.Series(MarketState.RANGING.value, index=df.index)
            for i in range(len(df)):
                market_state = self._classify_market_state(df, i)
                market_state_labels.iloc[i] = market_state.value
            
            # 진입 레이블 (수익성 기반)
            entry_labels = pd.Series(0, index=df.index)
            for i in range(len(df)-6):
                # 롱 진입 가능성
                if (df['z_score'].iloc[i] < -self.entry_threshold and 
                    df['rsi'].iloc[i] < 30):
                    future_returns = df['close'].iloc[i+1:i+6].pct_change().mean()
                    entry_labels.iloc[i] = 1 if future_returns > 0.001 else 0
                
                # 숏 진입 가능성
                elif (df['z_score'].iloc[i] > self.entry_threshold and 
                      df['rsi'].iloc[i] > 70):
                    future_returns = df['close'].iloc[i+1:i+6].pct_change().mean()
                    entry_labels.iloc[i] = 1 if future_returns < -0.001 else 0
            
            # 청산 레이블 (수익성 기반)
            exit_labels = pd.Series(0, index=df.index)
            for i in range(len(df)-3):
                # 롱 포지션 청산 가능성
                if df['position'].iloc[i] == 1:
                    future_returns = df['close'].iloc[i+1:i+3].pct_change().mean()
                    exit_labels.iloc[i] = 1 if future_returns < 0.0005 else 0
                
                # 숏 포지션 청산 가능성
                elif df['position'].iloc[i] == -1:
                    future_returns = df['close'].iloc[i+1:i+3].pct_change().mean()
                    exit_labels.iloc[i] = 1 if future_returns > -0.0005 else 0
            
            # 클래스 불균형 확인 및 조정
            for labels in [entry_labels, exit_labels]:
                positive_ratio = labels.mean()
                if positive_ratio < 0.1:  # 양성 클래스가 10% 미만인 경우
                    # 양성 클래스 샘플 증가
                    positive_indices = labels[labels == 1].index
                    if len(positive_indices) > 0:
                        additional_samples = min(len(positive_indices) * 2, len(labels) - len(positive_indices))
                        new_positive_indices = np.random.choice(positive_indices, size=additional_samples, replace=True)
                        labels.loc[new_positive_indices] = 1
            
            return market_state_labels, entry_labels, exit_labels
            
        except Exception as e:
            logger.error(f"레이블 준비 중 오류 발생: {str(e)}")
            return pd.Series(), pd.Series(), pd.Series()

    def _train_models(self, df: pd.DataFrame):
        """머신러닝 모델 학습"""
        try:
            # 특성 준비
            features = self._prepare_features(df)
            if features.empty:
                return
            
            # 특성 스케일링
            scaled_features = self.feature_scaler.fit_transform(features)
            
            # 레이블 준비
            market_state_labels, entry_labels, exit_labels = self._prepare_labels(df)
            
            # 데이터 분할
            X_train, X_test, y_market_train, y_market_test = train_test_split(
                scaled_features, market_state_labels, test_size=0.2, random_state=42
            )
            train_indices = np.arange(len(X_train))
            
            # 시장 상황 예측 모델
            self.market_state_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.market_state_model.fit(X_train, y_market_train)
            
            # 진입 시점 예측 모델
            self.entry_model = None
            entry_y = entry_labels.iloc[train_indices]
            if len(np.unique(entry_y)) < 2:
                logger.warning("진입 레이블에 클래스가 2개 미만입니다. 진입 모델 학습을 건너뜁니다.")
            else:
                self.entry_model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
                entry_pos = (entry_y == 1).sum()
                entry_neg = (entry_y == 0).sum()
                if entry_pos > 0 and entry_neg > 0:
                    entry_weight = np.where(entry_y == 1, entry_neg / entry_pos, 1)
                    self.entry_model.fit(X_train, entry_y, sample_weight=entry_weight)
                else:
                    self.entry_model.fit(X_train, entry_y)
            
            # 청산 시점 예측 모델
            self.exit_model = None
            exit_y = exit_labels.iloc[train_indices]
            if len(np.unique(exit_y)) < 2:
                logger.warning("청산 레이블에 클래스가 2개 미만입니다. 청산 모델 학습을 건너뜁니다.")
            else:
                self.exit_model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
                exit_pos = (exit_y == 1).sum()
                exit_neg = (exit_y == 0).sum()
                if exit_pos > 0 and exit_neg > 0:
                    exit_weight = np.where(exit_y == 1, exit_neg / exit_pos, 1)
                    self.exit_model.fit(X_train, exit_y, sample_weight=exit_weight)
                else:
                    self.exit_model.fit(X_train, exit_y)
            
            # 리스크 예측 모델
            self.risk_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.risk_model.fit(X_train, features['volatility'].iloc[train_indices])
            
            # 모델 저장
            self._save_models()
            
            logger.info("머신러닝 모델 학습 완료")
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            # 모델 초기화
            self.market_state_model = None
            self.entry_model = None
            self.exit_model = None
            self.risk_model = None

    def _save_models(self):
        """모델 저장"""
        try:
            joblib.dump(self.market_state_model, f"{self.model_dir}/market_state_model.joblib")
            joblib.dump(self.entry_model, f"{self.model_dir}/entry_model.joblib")
            joblib.dump(self.exit_model, f"{self.model_dir}/exit_model.joblib")
            joblib.dump(self.risk_model, f"{self.model_dir}/risk_model.joblib")
            joblib.dump(self.feature_scaler, f"{self.model_dir}/feature_scaler.joblib")
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {str(e)}")

    def _load_models(self):
        """모델 로드"""
        try:
            self.market_state_model = joblib.load(f"{self.model_dir}/market_state_model.joblib")
            self.entry_model = joblib.load(f"{self.model_dir}/entry_model.joblib")
            self.exit_model = joblib.load(f"{self.model_dir}/exit_model.joblib")
            self.risk_model = joblib.load(f"{self.model_dir}/risk_model.joblib")
            self.feature_scaler = joblib.load(f"{self.model_dir}/feature_scaler.joblib")
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")

    def _predict_market_state(self, features: pd.DataFrame) -> MarketState:
        """시장 상황 예측"""
        try:
            if self.market_state_model is None:
                return MarketState.RANGING
            
            scaled_features = self.feature_scaler.transform(features)
            prediction = self.market_state_model.predict(scaled_features)[-1]
            
            return MarketState.TRENDING if prediction == MarketState.TRENDING.value else \
                   MarketState.RANGING if prediction == MarketState.RANGING.value else \
                   MarketState.VOLATILE
            
        except Exception as e:
            logger.error(f"시장 상황 예측 중 오류 발생: {str(e)}")
            return MarketState.RANGING

    def _predict_entry(self, features: pd.DataFrame) -> bool:
        """진입 시점 예측"""
        try:
            if self.entry_model is None:
                return False
            
            scaled_features = self.feature_scaler.transform(features)
            prediction = self.entry_model.predict_proba(scaled_features)[-1]
            
            return prediction[1] > 0.6  # 60% 이상의 확률로 진입
            
        except Exception as e:
            logger.error(f"진입 시점 예측 중 오류 발생: {str(e)}")
            return False

    def _predict_exit(self, features: pd.DataFrame) -> bool:
        """청산 시점 예측"""
        try:
            if self.exit_model is None:
                return False
            
            scaled_features = self.feature_scaler.transform(features)
            prediction = self.exit_model.predict_proba(scaled_features)[-1]
            
            return prediction[1] > 0.6  # 60% 이상의 확률로 청산
            
        except Exception as e:
            logger.error(f"청산 시점 예측 중 오류 발생: {str(e)}")
            return False

    def _predict_risk(self, features: pd.DataFrame) -> float:
        """리스크 예측"""
        try:
            if self.risk_model is None:
                return 1.0
            
            scaled_features = self.feature_scaler.transform(features)
            prediction = self.risk_model.predict(scaled_features)[-1]
            
            return min(2.0, max(0.5, prediction))  # 0.5 ~ 2.0 사이로 제한
            
        except Exception as e:
            logger.error(f"리스크 예측 중 오류 발생: {str(e)}")
            return 1.0

    def _should_update_models(self, current_time: datetime) -> bool:
        """모델 업데이트 필요 여부 확인"""
        if self.last_model_update is None:
            return True
        
        time_diff = current_time - self.last_model_update
        return time_diff.days >= self.model_update_period

    def _generate_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """매매 신호 생성"""
        try:
            # 기본 데이터 복사
            df = df.copy()
            
            # 결측치 처리 (경고 수정)
            df = df.ffill().bfill()
            
            # 이동평균 계산
            df['sma'] = ta.sma(df['close'], length=self.lookback_period)
            df['std'] = df['close'].rolling(window=self.lookback_period).std()
            
            # Z-score 계산
            df['z_score'] = (df['close'] - df['sma']) / df['std']
            
            # 볼린저 밴드
            bb = ta.bbands(df['close'], length=self.lookback_period, std=2)
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # 거래량 지표
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # 추세 지표
            df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
            
            # 변동성 지표
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
            
            # 추가 모멘텀 지표
            df['momentum'] = df['close'].pct_change(5)
            df['momentum_ma'] = df['momentum'].rolling(10).mean()
            
            # 추가 지표: MACD
            macd = ta.macd(df['close'])
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            
            # 추가 지표: Stochastic
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
            
            # 신호 초기화
            df['entry_signal'] = 0
            df['exit_signal'] = 0
            df['position'] = 0
            df['market_state'] = MarketState.RANGING.value
            
            # 포지션 관리
            current_position = 0
            entry_price = 0
            entry_time = None
            holding_periods = 0
            
            # 거래 통계
            total_trades = 0
            winning_trades = 0
            
            # 모델 업데이트 확인
            current_time = df.index[-1]
            if self._should_update_models(current_time):
                self._train_models(df)
                self.last_model_update = current_time
            
            # 특성 준비
            features = self._prepare_features(df)
            
            # 시장 지표 업데이트
            self.market_metrics.update()
            
            for i in range(len(df)):
                if i < self.lookback_period:
                    continue
                    
                current_price = df['close'].iloc[i]
                current_time = df.index[i]
                
                # 시장 상황 분류
                market_state = self._classify_market_state(df, i)
                
                # 전략 선택
                selected_strategy = self.strategy_selector.select_strategy(market_state)
                
                # 전략 객체 선택
                if selected_strategy == 'trend':
                    strategy = self.trend_strategy
                elif selected_strategy == 'mean_revert':
                    strategy = self.mean_revert_strategy
                else:
                    strategy = self.volatility_strategy
                
                # 전략 전환 필요 여부 확인
                if strategy.position != 0 and \
                   self.strategy_selector.should_switch_strategy(selected_strategy, market_state):
                    # 현재 포지션 청산
                    df.loc[current_time, 'exit_signal'] = 1
                    profit = (current_price - strategy.entry_price) / strategy.entry_price * strategy.position
                    
                    # 전환 로그 추가
                    best_strategy = self.strategy_selector.get_best_strategy(market_state)
                    current_score = self.strategy_selector.strategies[selected_strategy].get_score()
                    best_score = self.strategy_selector.strategies[best_strategy].get_score()
                    
                    logger.info(f"전략 전환: {selected_strategy} -> {best_strategy}")
                    logger.info(f"전환 사유: 현재 점수 {current_score:.2f}, 최고 점수 {best_score:.2f}")
                    logger.info(f"청산 정보: {current_time}, 가격: {current_price:.2f}, 수익률: {profit:.2%}")
                    
                    # 전략 초기화
                    strategy.position = 0
                    strategy.entry_price = 0
                    strategy.entry_time = None
                    
                    # 새로운 전략으로 전환
                    if best_strategy == 'trend':
                        strategy = self.trend_strategy
                    elif best_strategy == 'mean_revert':
                        strategy = self.mean_revert_strategy
                    else:
                        strategy = self.volatility_strategy

                # 진입 신호 확인
                if strategy.position == 0:
                    should_enter, position_size = strategy.should_enter(df, i)
                    if should_enter:
                        df.loc[current_time, 'entry_signal'] = strategy.position
                        df.loc[current_time, 'position_size'] = position_size
                        logger.info(f"{selected_strategy} 전략 진입: {current_time}, 가격: {current_price:.2f}, 포지션: {strategy.position}")
                
                # 청산 신호 확인
                elif strategy.should_exit(df, i):
                    df.loc[current_time, 'exit_signal'] = 1
                    profit = (current_price - strategy.entry_price) / strategy.entry_price * strategy.position
                    logger.info(f"{selected_strategy} 전략 청산: {current_time}, 가격: {current_price:.2f}, 수익률: {profit:.2%}")
                    
                    # 거래 데이터 업데이트
                    trade_data = {
                        'entry_time': strategy.entry_time,
                        'exit_time': current_time,
                        'entry_price': strategy.entry_price,
                        'exit_price': current_price,
                        'position': strategy.position,
                        'profit': profit,
                        'market_state': market_state.value,
                        'strategy_name': selected_strategy,
                        'strategy_market_state': market_state.value
                    }
                    self.strategy_selector.update(trade_data)
                    
                    strategy.position = 0
                    strategy.entry_price = 0
                    strategy.entry_time = None
                
                df.loc[current_time, 'position'] = strategy.position
                df.loc[current_time, 'market_state'] = market_state.value
                df.loc[current_time, 'strategy'] = selected_strategy
            
            return df
            
        except Exception as e:
            logger.error(f"신호 생성 중 오류 발생: {str(e)}")
            raise 