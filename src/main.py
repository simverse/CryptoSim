import argparse
import logging
import sys
from datetime import datetime

import pandas as pd

from src.config_manager import ConfigManager
from src.data_fetcher import fetch_historical_data
from src.recorder import BacktestRecorder
from src.strategy import SMACrossoverStrategy, ParabolicSARStrategy # ParabolicSARStrategy 임포트
from src.trade_analyzer import TradeAnalyzer

# 삭제: create_strategy_config 함수

def setup_logging():
    """로깅 설정"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler("backtest_log.txt", mode='w', encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

def run_backtest_pipeline(all_data: pd.DataFrame, backtest_data: pd.DataFrame, 
                 initial_balance: float, report_start_date: str, backtest_config: dict, config_manager: ConfigManager) -> None: # symbol 인자 제거
    """백테스트 실행"""
    
    print(f"\n백테스트 실행")
    print(f"   전체 데이터: {len(all_data)}봉")
    print(f"   백테스트 데이터: {len(backtest_data)}봉")
    print(f"   초기자본: {initial_balance:.6f} BTC")
    
    # 최소 데이터 요구량은 전략에 따라 다를 수 있으므로, 각 전략에서 처리하도록 위임
    # if len(all_data) < 720:
    #     print(f"데이터 부족: {len(all_data)}봉 (최소 720봉 필요)")
    #     return

    # --- 동적 전략 선택 로직 시작 ---
    strategy_name = config_manager.get('strategy.name') # 기본값 제거

    if strategy_name == 'sma_crossover':
        strategy = SMACrossoverStrategy(config_manager)
    elif strategy_name == 'psar':
        strategy = ParabolicSARStrategy(config_manager)
    else:
        raise ValueError(f"지원하지 않는 전략입니다: {strategy_name}")

    # --- 동적 전략 선택 로직 종료 ---

    trade_analyzer = TradeAnalyzer(config=config_manager, initial_balance=initial_balance)
    
    result = trade_analyzer.run_backtest(
        strategy=strategy,
        all_data=all_data,
        backtest_data=backtest_data,
        commission=backtest_config['fees']['commission'],
        funding_rate=backtest_config['fees']['funding_rate']
    )
    
    print(f"백테스트 완료!")
    print(f"   총 거래수: {result.total_trades}")
    print(f"   승률: {result.win_rate:.1%}")
    print(f"   총 수익률: {result.total_return:.2%}")
    print(f"   연간 수익률: {result.annual_return:.2%}")
    print(f"   최대 낙폭: {result.max_drawdown:.2%}")
    print(f"   샤프 비율: {result.sharpe_ratio:.2f}")
    print(f"   최종 잔고: {result.final_balance:.2f}")
    
    print(f"\n리포트 생성 중...")
    
    recorder = BacktestRecorder(
        initial_balance=initial_balance,
        output_dir=f"./{backtest_config['report']['output_dir']}",
        report_start_date=report_start_date,
        config_manager=config_manager
    )
    
    signals_data = strategy.generate_signals(all_data.copy())
    detailed_records = getattr(result, 'detailed_records', [])

    backtest_start_datetime = backtest_data.index[0] if len(backtest_data) > 0 else pd.to_datetime(report_start_date)
    backtest_end_datetime = backtest_data.index[-1] if len(backtest_data) > 0 else pd.to_datetime(report_start_date)
    
    filtered_signals_data = signals_data[
        (signals_data.index >= backtest_start_datetime) & 
        (signals_data.index <= backtest_end_datetime)
    ].copy()

    ohlc_signals = recorder._create_ohlc_signals_sheet(backtest_data, filtered_signals_data, detailed_records, strategy) # strategy 객체 전달
    trade_history = recorder._create_trade_history_sheet(result.trades) # 거래내역 시트 생성
    
    reports = {
        'ohlc_signals': ohlc_signals,
        'trade_history': trade_history # 리포트에 추가
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recorder.save_report(reports, f"{strategy.name}_{timestamp}")

def calculate_data_requirements(backtest_config: dict, timeframe: str) -> dict:
    """백테스트에 필요한 데이터량 계산"""
    
    trading_period = backtest_config['trading_period']
    # SMA 기간을 하드코딩하지 않고, 여유분을 두는 방식으로 변경
    indicator_period = 720 # 보수적으로 가장 긴 지표 기간으로 설정
    
    if trading_period.get('start_date') and trading_period.get('end_date'):
        start_date = pd.to_datetime(trading_period['start_date']).tz_localize('Asia/Seoul')
        end_date = pd.to_datetime(trading_period['end_date']).tz_localize('Asia/Seoul')
        
        if timeframe == '1h':
            hours_diff = int((end_date - start_date).total_seconds() / 3600)
        else:
            raise ValueError(f"지원하지 않는 timeframe: {timeframe}")
        
        trading_hours = hours_diff
        report_start_date = trading_period['start_date']
        
        data_start_date_kst = start_date - pd.Timedelta(hours=indicator_period)
        data_start_date = data_start_date_kst.tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S')
        total_data_needed = trading_hours + indicator_period
        
    else:
        # 날짜 지정이 없는 경우, trading_period.hours는 더 이상 사용하지 않음
        raise ValueError("백테스트 기간(start_date, end_date)이 설정되지 않았습니다.")
    
    return {
        'trading_hours': trading_hours,
        'indicator_period': indicator_period,
        'total_data_needed': total_data_needed,
        'report_start_date': report_start_date,
        'data_start_date': data_start_date
    }

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='모듈화된 백테스트 스크립트')
    
    # --symbol 인자 제거
    parser.add_argument('--timeframe', default='1h', help='시간봉')
    parser.add_argument('--initial-balance', type=float, default=None, help='초기 자본 BTC')
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("모듈화된 백테스트 스크립트")
    print("=" * 40)
    
    try:
        config_manager = ConfigManager()
        backtest_config = config_manager.get_backtest_config()
        trading_config = config_manager.get_trading_config()
        
        data_req = calculate_data_requirements(backtest_config, args.timeframe)
        
        symbol = trading_config['symbol'] # config에서 직접 심볼 로드
        initial_balance = args.initial_balance or backtest_config['initial_balance']
        report_start_date = data_req['report_start_date']
        
        print(f"백테스트 설정:")
        print(f"  - 심볼: {symbol}")
        print(f"  - 거래기간: {data_req['trading_hours']}시간")
        print(f"  - 초기자본: {initial_balance:.6f} BTC")
        print(f"  - 리포트 시작: {report_start_date}")
        
        print(f"\n필요 데이터:")
        print(f"  - 거래 기간: {data_req['trading_hours']}봉")
        print(f"  - 지표 계산용: {data_req['indicator_period']}봉")
        print(f"  - 총 필요량: {data_req['total_data_needed']}봉")
        
        all_data = fetch_historical_data(config_manager, symbol, args.timeframe, data_req['total_data_needed'], data_req['data_start_date'])
        
        if all_data.empty:
            print("데이터 수집에 실패하여 백테스트를 종료합니다.")
            return

        report_start_datetime = pd.to_datetime(report_start_date)
        report_end_datetime = pd.to_datetime(backtest_config['trading_period']['end_date'])

        if all_data.index.tz is not None:
            report_start_datetime = report_start_datetime.tz_localize(all_data.index.tz)
            report_end_datetime = report_end_datetime.tz_localize(all_data.index.tz)
        
        backtest_data = all_data[(all_data.index >= report_start_datetime) & (all_data.index <= report_end_datetime)].copy()
        
        logging.info(f"Total data fetched: {len(all_data)} candles")
        logging.info(f"Backtest data from {report_start_date}: {len(backtest_data)} candles")

        # strategy_config 및 symbol 인자 제거
        run_backtest_pipeline(all_data, backtest_data, initial_balance, report_start_date, backtest_config, config_manager)

        
        print("\n완료!")
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()