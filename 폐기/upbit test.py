import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

symbol = "DOGEUSDT"
interval = "1m"
base_url = "https://data.binance.vision/data/spot/daily/klines"

# 오늘로부터 1년치 날짜 리스트 생성 → 원하는 기간으로 변경
start = datetime(2024, 5, 27, tzinfo=timezone.utc)
end = datetime(2025, 4, 27, tzinfo=timezone.utc)
dates = pd.date_range(start, end, freq='D').to_pydatetime()

dfs = []
for dt in tqdm(dates, desc="다운로드 진행"):
    y, m, d = dt.year, dt.month, dt.day
    m_str = f"{m:02d}"
    d_str = f"{d:02d}"
    url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{y}-{m_str}-{d_str}.zip"
    r = requests.get(url)
    if r.status_code != 200:
        tqdm.write(f"다운로드 실패: {url} (status={r.status_code})")
        continue
    tqdm.write(f"다운로드 성공: {url}")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_files = [name for name in z.namelist() if name.endswith('.csv')]
    if not csv_files:
        continue
    csv_name = csv_files[0]
    df = pd.read_csv(z.open(csv_name), header=None)
    dfs.append(df)

if not dfs:
    print("다운로드된 데이터가 없습니다.")
else:
    df_all = pd.concat(dfs)
    # open_time 단위 자동 판별 및 변환
    if df_all[0].max() > 1e14:
        df_all[0] = (df_all[0] // 1000)
    df_all.columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df_all["datetime"] = pd.to_datetime(df_all["open_time"], unit="ms")
    # 필요한 컬럼만 추출 (datetime, open, high, low, close, volume)
    df_all = df_all[["datetime", "open", "high", "low", "close", "volume"]]
    # 정렬 및 인덱스 초기화
    df_all = df_all.sort_values("datetime").reset_index(drop=True)
    # 저장
    save_path = "C:/Cording/volatility_bot/data.csv"
    df_all.to_csv(save_path, index=False)
    print(f"저장 완료: {save_path}, shape={df_all.shape}")
    print("최종 데이터 범위:", df_all['datetime'].min(), "~", df_all['datetime'].max())
    print(df_all[df_all['datetime'] >= '2025-01-01'].head())
    print(df_all[df_all['datetime'] >= '2025-04-01'].tail())