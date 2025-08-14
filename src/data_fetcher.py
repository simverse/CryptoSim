import time
from datetime import timezone
import pandas as pd
from src.exchange import OKXExchange

def fetch_okx_data(exchange: OKXExchange, symbol: str, timeframe: str, limit: int, start_date=None) -> pd.DataFrame:
    """OKX에서 OHLCV 데이터 가져오기"""
    
    print(f"OKX 데이터 수집: {symbol} {timeframe} {limit}개")
    if start_date:
        print(f"   시작일: {start_date}")
    
    try:
        all_data = []
        max_per_request = 1000
        
        if start_date:
            current_since = int(pd.to_datetime(start_date).timestamp() * 1000)
            seen_timestamps = set()

            while len(all_data) < limit:
                remaining = min(max_per_request, limit - len(all_data))
                print(f"  요청: {remaining}개 (총 {len(all_data)}/{limit})")
                
                params = {'limit': remaining, 'since': current_since}
                ohlcv = exchange.exchange.fetch_ohlcv(symbol, timeframe, **params)
                
                if not ohlcv:
                    print("  더 이상 데이터가 없습니다")
                    break
                
                new_data = []
                for candle in ohlcv:
                    timestamp = candle[0]
                    if timestamp not in seen_timestamps:
                        new_data.append(candle)
                        seen_timestamps.add(timestamp)
                
                if not new_data:
                    # 새로운 데이터가 없으면 루프를 중단하여 무한 루프 방지
                    print("  새로운 데이터가 없어 조기 종료합니다.")
                    break

                all_data.extend(new_data)
                
                if ohlcv:
                    # timeframe을 밀리초 단위로 변환하여 다음 since를 정확하게 계산
                    timeframe_ms = exchange.exchange.parse_timeframe(timeframe) * 1000
                    latest_timestamp = max(candle[0] for candle in ohlcv)
                    current_since = latest_timestamp + timeframe_ms
                
                # time.sleep(0.2)
                
        else:
            # 'since' 파라미터 없이 최신 데이터부터 역순으로 가져오기
            needed = limit
            while needed > 0:
                fetch_limit = min(needed, max_per_request)
                print(f"  요청: {fetch_limit}개 (남은 양: {needed})")
                
                ohlcv = exchange.exchange.fetch_ohlcv(symbol, timeframe, limit=fetch_limit)
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # 중복 제거
                all_data = sorted(list({tuple(i) for i in all_data}), key=lambda x: x[0])
                
                needed = limit - len(all_data)
                
                # 다음 요청을 위해 가장 오래된 데이터의 이전 시점 설정
                oldest_timestamp = min(candle[0] for candle in ohlcv)
                # CCXT는 'since'가 아닌 'until'을 사용하지 않으므로, 
                # 이 방식은 한 번에 모든 데이터를 가져오는 것을 가정함.
                # 대량 데이터 수집 시 로직 수정 필요.
                # 여기서는 limit 만큼만 가져오도록 단순화.
                break # 현재 로직에서는 한 번만 요청
        
        # 시간순 정렬
        all_data.sort(key=lambda x: x[0])
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
        print(f"DataFrame datetime column timezone: {df['datetime'].dt.tz}")
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        df = df.astype(float)
        
        print(f"데이터 수집 완료: {len(df)}봉")
        if not df.empty:
            print(f"   기간: {df.index[0]} ~ {df.index[-1]}")
        
        return df
        
    except Exception as e:
        print(f"데이터 수집 실패: {e}")
        raise
