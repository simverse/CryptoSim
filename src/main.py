import argparse
import logging
import sys
from datetime import datetime

import pandas as pd

# 모듈 경로 추가
import sys


from src.config_manager import ConfigManager
from src.data_fetcher import fetch_okx_data
from src.exchange import OKXExchange
from src.recorder import BacktestRecorder
from src.strategy import SMACrossoverStrategy
from src.trade_analyzer import TradeAnalyzer

def create_strategy_config(symbol: str, config_manager: ConfigManager) -> dict:
    """전략 설정 (config.yaml에서 읽어옴)"""
    trading_config = config_manager.get_trading_config()
    strategy_config = config_manager.config.get('strategy', {}).get('simple_ma', {})
    
    # 필수 설정값 추출 (디폴트 값 없음 - 설정 누락시 에러 발생)
    try:
        leverage = trading_config['leverage']
        position_size = trading_config['position_size']
        take_profit_pct = trading_config['take_profit_pct']
        stop_loss_pct = trading_config['stop_loss_pct']
        short_period = strategy_config['short_period']
        long_period = strategy_config['long_period']
    except KeyError as e:
        raise ValueError(f"필수 설정값이 누락되었습니다: {e}")
    
    return {
        'symbol': symbol,
        'short_sma_period': short_period,
        'long_sma_period': long_period,
        'leverage': leverage,
        'margin_per_trade': position_size / leverage,  # 포지션 크기를 레버리지로 ��누어 마진 계산
        'position_size': position_size,
        'take_profit_pct': take_profit_pct,
        'stop_loss_pct': stop_loss_pct,
        'stop_loss_enabled': True,
        'max_concurrent_positions': 5,
        'daily_loss_limit_pct': 0.05
    }

def setup_logging():
    """로깅 설정"""
    # 로그 디렉토리 생성 (config_manager에서 처리)
    # Path("backtest_log.txt").parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거 (중복 로깅 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 파일 핸들러 (상세 로그)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler("backtest_log.txt", mode='w', encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 (간결한 타임스탬프만 출력)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

def run_backtest_pipeline(all_data: pd.DataFrame, backtest_data: pd.DataFrame, strategy_config: dict, 
                 initial_balance: float, report_start_date: str, backtest_config: dict, config_manager: ConfigManager) -> None:
    """백테스트 실행"""
    
    print(f"\n백테스트 실행")
    print(f"   전체 데이터: {len(all_data)}봉")
    print(f"   백테스트 데이터: {len(backtest_data)}봉")
    print(f"   초기자본: {initial_balance:.6f} BTC")
    
    if len(all_data) < 720:
        print(f"데이터 부족: {len(all_data)}봉 (최소 720봉 필요)")
        return
    
    # 전략 및 백테스터 초기화
    strategy = SMACrossoverStrategy(strategy_config, config_manager)
    trade_analyzer = TradeAnalyzer(config=config_manager, initial_balance=initial_balance)
    
    # 백테스트 실행 (지표 계산은 all_data, 실제 테스트는 backtest_data 사용)
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
    print(f"   연간 수익률: {result.annual_return:.2f}")
    print(f"   최대 낙폭: {result.max_drawdown:.2%}")
    print(f"   샤프 비율: {result.sharpe_ratio:.2f}")
    
    print(f"   최종 잔고: {result.final_balance:.2f}")
    
    # 리포트 생성 (상세 정보 포함)
    print(f"\n리포트 생성 중...")
    
    recorder = BacktestRecorder(
        initial_balance=backtest_config['initial_balance_btc'],
        output_dir=f"./{backtest_config['report']['output_dir']}",
        report_start_date=report_start_date
    )
    
    signals_data = strategy.generate_signals(all_data.copy())
    detailed_records = getattr(result, 'detailed_records', [])

    # 백테스트 데이터 기준으로 신호 데이터 필터링 (실제 백테스트에 사용된 데이터만 포함)
    backtest_start_datetime = backtest_data.index[0] if len(backtest_data) > 0 else pd.to_datetime(report_start_date)
    backtest_end_datetime = backtest_data.index[-1] if len(backtest_data) > 0 else pd.to_datetime(report_start_date)
    
    # 백테스트 기간에 해당하는 신호 데이터만 필터링
    filtered_signals_data = signals_data[
        (signals_data.index >= backtest_start_datetime) & 
        (signals_data.index <= backtest_end_datetime)
    ].copy()

    ohlc_signals = recorder._create_ohlc_signals_sheet(backtest_data, filtered_signals_data, detailed_records)
    
    reports = {'ohlc_signals': ohlc_signals}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = recorder._save_to_excel(reports, f"SMACrossoverStrategy_{timestamp}")
    
    print(f"리포트 저장: {excel_path}")

def calculate_data_requirements(backtest_config: dict, timeframe: str) -> dict:
    """백테스트에 필요한 데이터량 계산"""
    
    trading_period = backtest_config['trading_period']
    sma_period = 720
    
    if trading_period.get('start_date') and trading_period.get('end_date'):
        # config.yaml의 날짜는 KST 기준이므로 명시적으로 시간대 설정
        start_date = pd.to_datetime(trading_period['start_date']).tz_localize('Asia/Seoul')
        end_date = pd.to_datetime(trading_period['end_date']).tz_localize('Asia/Seoul')
        
        if timeframe == '1h':
            hours_diff = int((end_date - start_date).total_seconds() / 3600)
        else:
            raise ValueError(f"지원하지 않는 timeframe: {timeframe}")
        
        trading_hours = hours_diff
        report_start_date = trading_period['start_date']
        
        # KST 기준으로 SMA 계산용 시작일을 구한 후 UTC로 변환하여 API 호출에 사용
        data_start_date_kst = start_date - pd.Timedelta(hours=sma_period)
        data_start_date = data_start_date_kst.tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S')
        total_data_needed = trading_hours + sma_period
        
    else:
        trading_hours = trading_period['hours']
        total_data_needed = trading_hours + sma_period
        report_start_date = "2025-01-01 00:00:00"
        data_start_date = None
    
    return {
        'trading_hours': trading_hours,
        'sma_period': sma_period,
        'total_data_needed': total_data_needed,
        'report_start_date': report_start_date,
        'data_start_date': data_start_date
    }

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='모듈화된 백테스트 스크립트')
    
    parser.add_argument('--symbol', default=None, help='거래 심볼')
    parser.add_argument('--timeframe', default='1h', help='시간봉')
    parser.add_argument('--hours', type=int, default=None, help='거래할 시간봉 개수')
    parser.add_argument('--initial-balance', type=float, default=None, help='초기 자본 BTC')
    parser.add_argument('--report-start', default=None, help='리포트 시작 날짜')
    parser.add_argument('--sandbox', action='store_true', default=True, help='샌드박스 모드')
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("모듈화된 백테스트 스크립트")
    print("=" * 40)
    
    try:
        # 설정 파일 로드
        config_manager = ConfigManager()
        backtest_config = config_manager.get_backtest_config()
        trading_config = config_manager.get_trading_config()
        
        # 데이터 요구사항 계산
        data_req = calculate_data_requirements(backtest_config, args.timeframe)
        
        # CLI 인수로 오버라이드
        symbol = args.symbol or trading_config['symbol']
        initial_balance = args.initial_balance or backtest_config['initial_balance_btc']
        report_start_date = args.report_start or data_req['report_start_date']
        
        print(f"백테스트 설정:")
        print(f"  - 심볼: {symbol}")
        print(f"  - 거래기간: {data_req['trading_hours']}시간")
        print(f"  - 초기자본: {initial_balance:.6f} BTC")
        print(f"  - 리포트 시작: {report_start_date}")
        
        # OKX 연결
        print("\nOKX 연결 중...")
        exchange = OKXExchange(config_manager=config_manager)
        
        if not exchange.test_connection():
            print("OKX 연결 실패")
            print("API 키 환경변수를 확인하세요:")
            print("  - OKX_API_KEY")
            print("  - OKX_SECRET_KEY") 
            print("  - OKX_PASSPHRASE")
            print("  - OKX_SANDBOX")
            return
        
        print("OKX 연결 성공")
        
        print(f"\n필요 데이터:")
        print(f"  - 거래 기간: {data_req['trading_hours']}봉")
        print(f"  - SMA 계산용: {data_req['sma_period']}봉")
        print(f"  - 총 필요량: {data_req['total_data_needed']}봉")
        
        # 데이터 수집
        all_data = fetch_okx_data(exchange, symbol, args.timeframe, data_req['total_data_needed'], data_req['data_start_date'])
        
        # 리포트 시작 및 종료 날짜 기준으로 데이터 필터링
        report_start_datetime = pd.to_datetime(report_start_date)
        report_end_datetime = pd.to_datetime(backtest_config['trading_period']['end_date'])

        if all_data.index.tz is not None:
            report_start_datetime = report_start_datetime.tz_localize(all_data.index.tz)
            report_end_datetime = report_end_datetime.tz_localize(all_data.index.tz)
        
        # 실제 백테스트에 사용할 데이터는 start_date와 end_date 사이의 데이터
        backtest_data = all_data[(all_data.index >= report_start_datetime) & (all_data.index <= report_end_datetime)].copy()
        
        logging.info(f"Total data fetched: {len(all_data)} candles")
        logging.info(f"Backtest data from {report_start_date}: {len(backtest_data)} candles")

        # 백테스트 실행 (전체 데이터를 전달하여 SMA 계산은 유지)
        strategy_config = create_strategy_config(symbol, config_manager)
        run_backtest_pipeline(all_data, backtest_data, strategy_config, initial_balance, report_start_date, backtest_config, config_manager)
        
        print("\n완료!")
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()