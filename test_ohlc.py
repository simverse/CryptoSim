import pandas as pd
from src.config_manager import ConfigManager
from src.data_fetcher import fetch_okx_data
from src.exchange import OKXExchange

# 설정 및 거래소 초기화
config_manager = ConfigManager()
exchange = OKXExchange(config_manager=config_manager)

# 2025-01-01 00:00:00 시간대 데이터 확인
data = fetch_okx_data(exchange, 'BTC/USD:BTC', '1h', 24, '2025-07-04 00:00:00')
print('2025-01-01 00:00:00 시간대 데이터:')
print(data.head())
print()
print('첫 번째 행 상세 정보:')
print(f'시간: {data.index[0]}')
print(f'시가: {data.iloc[0]["open"]}')
print(f'고가: {data.iloc[0]["high"]}')
print(f'저가: {data.iloc[0]["low"]}')
print(f'종가: {data.iloc[0]["close"]}')