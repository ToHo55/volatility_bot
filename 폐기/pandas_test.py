import pandas as pd
import os

# report_summary.txt를 스크립트와 같은 폴더에 저장
script_dir = os.path.dirname(os.path.abspath(__file__))
report_path = os.path.join(script_dir, 'report_summary.txt')

df = pd.read_csv('market_debug_clean.log', header=None,
                 names=['datetime', 'ticker', 'trend_strength', 'vol_short', 'vol_long', 'volume_ratio'])

# nan 값은 0으로 대체(임시)
df['vol_long'] = pd.to_numeric(df['vol_long'], errors='coerce').fillna(0)

# 주요 통계
print(df[['trend_strength', 'vol_short', 'vol_long', 'volume_ratio']].describe())

# trend_strength 임계값 추천 (상위/하위 5%)
upper_trend = df['trend_strength'].quantile(0.95)
lower_trend = df['trend_strength'].quantile(0.05)
print(f"trend_strength 임계값 추천: 상위 5% {upper_trend:.4f}, 하위 5% {lower_trend:.4f}")

# vol_short 임계값 추천 (상위 5%)
upper_vol = df['vol_short'].quantile(0.95)
print(f"vol_short 임계값 추천: 상위 5% {upper_vol:.4f}")

# volume_ratio 임계값 추천 (상위 5%)
upper_volratio = df['volume_ratio'].quantile(0.95)
print(f"volume_ratio 임계값 추천: 상위 5% {upper_volratio:.2f}")

def classify_market(row):
    if row['vol_short'] > 0.0063:
        return '고변동성'
    elif row['trend_strength'] > 0.0030:
        return '상승추세'
    elif row['trend_strength'] < -0.0031:
        return '하락추세'
    else:
        return '횡보장'

df['market_type'] = df.apply(classify_market, axis=1)
print(df['market_type'].value_counts())

log = pd.read_csv('market_debug_clean.log', header=None,
    names=['datetime', 'ticker', 'trade_type', 'price', 'profit_pct', 'reason', 'market_condition'])

# EXIT만 추출
log['date'] = pd.to_datetime(log['datetime'], errors='coerce').dt.date
exits = log[log['trade_type'] == 'EXIT']
exits = exits[log['date'].notnull()]

# 시장상태별 평균/표준편차/거래수
print(exits.groupby('market_condition')['profit_pct'].agg(['mean', 'std', 'count']))

# 시장상태별 승률
print(exits.groupby('market_condition').apply(lambda x: (x['profit_pct'] > 0).mean()))

# 코인별 성과
print(exits.groupby('ticker')['profit_pct'].agg(['mean', 'std', 'count']))

# 진입/청산 사유별 성과
print(exits.groupby('reason')['profit_pct'].agg(['mean', 'std', 'count']))

# 날짜별 성과(최근 10일)
print(exits.groupby('date')['profit_pct'].agg(['mean', 'std', 'count']).tail(10))

# ====== 자동 리포트 파일 저장 ======
with open(report_path, 'w', encoding='utf-8') as rf:
    rf.write('[시장상태별 EXIT 시점 성과]\n')
    market_report = exits.groupby('market_condition')['profit_pct'].agg(['mean', 'std', 'count'])
    rf.write(str(market_report)+'\n\n')
    print('\n[시장상태별 EXIT 시점 성과]')
    print(market_report)
    rf.write('[시장상태별 EXIT 시점 승률]\n')
    win_report = exits.groupby('market_condition').apply(lambda x: (x['profit_pct'] > 0).mean())
    rf.write(str(win_report)+'\n\n')
    print('\n[시장상태별 EXIT 시점 승률]')
    print(win_report)
    rf.write('[코인별 EXIT 시점 성과]\n')
    coin_report = exits.groupby('ticker')['profit_pct'].agg(['mean', 'std', 'count'])
    rf.write(str(coin_report)+'\n\n')
    print('\n[코인별 EXIT 시점 성과]')
    print(coin_report)
    rf.write('[진입/청산 사유별 성과]\n')
    reason_report = exits.groupby('reason')['profit_pct'].agg(['mean', 'std', 'count'])
    rf.write(str(reason_report)+'\n\n')
    print('\n[진입/청산 사유별 성과]')
    print(reason_report)
    rf.write('[날짜별 EXIT 시점 성과]\n')
    date_report = exits.groupby('date')['profit_pct'].agg(['mean', 'std', 'count'])
    rf.write(str(date_report.tail(10))+'\n\n')
    print('\n[최근 10일간 EXIT 시점 성과]')
    print(date_report.tail(10))
print(f'리포트가 {report_path} 파일에 저장되었습니다.')

print('리포트 파일 절대경로:', os.path.abspath(report_path))