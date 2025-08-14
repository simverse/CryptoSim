import time
import logging
from datetime import datetime
from src.config_manager import ConfigManager
from src.data_fetcher import DataFetcher
from src.exchange import Exchange
from src.strategy import Strategy
from src.trading_manager import TradingManager
from src.state_manager import StateManager
from src.trade_analyzer import TradeAnalyzer # TradeAnalyzer 임포트
from src.recorder import BacktestRecorder # BacktestRecorder 임포트

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    실시간 거래 메인 실행 함수
    """
    logging.info("실시간 거래 시스템을 시작합니다.")

    # 1. 설정 로드
    config_manager = ConfigManager('config/config.yaml')
    config = config_manager.get_config()

    # 2. 컴포넌트 초기화
    exchange = Exchange(config_manager)
    data_fetcher = DataFetcher(exchange.exchange_instance) # 실제 ccxt 객체 전달
    strategy = Strategy(config)
    trading_manager = TradingManager(config, exchange)
    state_manager = StateManager(config)
    
    initial_balance = config.get('backtest', {}).get('initial_balance', 1.0)
    trade_analyzer = TradeAnalyzer(config, initial_balance) # config 전달

    # BacktestRecorder 초기화
    recorder = BacktestRecorder(initial_balance=initial_balance)

    # 3. 상태 복구 (필요시)
    try:
        positions, open_orders, transactions, long_position_sets, short_position_sets = state_manager.load_state()
        if positions or open_orders:
            trading_manager.restore_state(positions, open_orders, transactions, long_position_sets, short_position_sets)
            logging.info("이전 상태를 성공적으로 복구했습니다.")
    except FileNotFoundError:
        logging.info("저장된 상태 파일이 없습니다. 새로운 상태로 시작합니다.")

    # 4. 메인 루프
    logging.info("메인 루프를 시작합니다.")
    try:
        while True:
            # 설정된 간격(예: 60초)마다 루프 실행
            time.sleep(config.get('system', {}).get('update_interval', 60))

            # 1. 최신 데이터 가져오기
            ohlc_data = data_fetcher.fetch_data(
                start_date_str=None,
                end_date_str=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                timeframe=config.get('strategy', {}).get('timeframe', '1h'),
                limit=config.get('strategy', {}).get('long_period', 720) + 50
            )

            if ohlc_data.empty:
                logging.warning("데이터를 가져오지 못했습니다. 다음 주기를 기다립니다.")
                continue

            # 2. 전략 실행 및 신호 생성
            signals_data = strategy.generate_signals(ohlc_data.copy())
            latest_signal = signals_data.iloc[-1]

            # 3. 신호를 바탕으로 거래 실행
            trading_manager.process_signal(latest_signal)

            # 4. 미체결 주문 관리
            trading_manager.check_open_orders()

            # 5. 상세 기록 생성 및 TradeAnalyzer에 추가
            trade_analyzer._record_detailed_data(latest_signal.name, latest_signal, strategy) # timestamp, row, strategy

    except KeyboardInterrupt:
        logging.info("사용자에 의해 프로그램이 중단되었습니다.")
    finally:
        # 6. 종료 처리
        logging.info("시스템을 종료합니다.")
        # 현재 상태 저장
        positions, open_orders, transactions, long_position_sets, short_position_sets = trading_manager.get_current_state()
        state_manager.save_state(positions, open_orders, transactions, long_position_sets, short_position_sets)
        logging.info("현재 상태를 성공적으로 저장했습니다.")

        # 엑셀 파일 저장
        if trade_analyzer.detailed_records:
            import pandas as pd
            # TradeAnalyzer의 detailed_records를 사용하여 리포트 생성
            # BacktestRecorder의 _create_ohlc_signals_sheet는 original_data와 signals_data를 필요로 함.
            # 여기서는 detailed_records를 기반으로 ohlc_signals_sheet를 생성하는 것이 더 적합.
            # detailed_records의 timestamp를 인덱스로 하는 DataFrame을 만들어 전달하는 것이 가장 정확함.

            # detailed_records를 DataFrame으로 변환
            detailed_df_for_recorder = pd.DataFrame(trade_analyzer.detailed_records)
            detailed_df_for_recorder['timestamp'] = pd.to_datetime(detailed_df_for_recorder['timestamp'])
            detailed_df_for_recorder.set_index('timestamp', inplace=True)

            # BacktestRecorder의 _create_ohlc_signals_sheet는 signals_data를 기준으로 병합하므로,
            # detailed_records의 timestamp와 일치하는 signals_data를 만들어야 함.
            # 가장 간단한 방법은 detailed_records를 기반으로 signals_data를 재구성하는 것.
            # 여기서는 ohlc_data와 signals_data를 직접 전달하지 않고, detailed_records를 기반으로 시트 생성.
            # BacktestRecorder의 _create_ohlc_signals_sheet가 detailed_records만으로도 동작하도록 수정이 필요할 수 있음.
            # 현재는 detailed_records를 직접 시트 데이터로 사용.

            # BacktestRecorder의 _create_ohlc_signals_sheet가 기대하는 형식에 맞추기 위해
            # ohlc_data와 signals_data를 detailed_records에서 추출하여 재구성
            # 이 부분은 BacktestRecorder의 내부 로직을 정확히 이해해야 함.
            # 현재 BacktestRecorder는 signals_data를 기준으로 ohlc_signals_sheet를 생성하고,
            # detailed_records는 추가 정보로 사용됨.

            # 임시로 detailed_records를 그대로 시세분석 시트로 사용
            # 실제 BacktestRecorder의 _create_ohlc_signals_sheet는 ohlc_data와 signals_data를 필요로 함.
            # 따라서, 매 루프마다 ohlc_data와 signals_data를 저장해야 함.
            # 이 부분은 현재 구조에서 바로 적용하기 어려움.

            # 일단 detailed_records를 직접 엑셀로 저장하는 방식으로 진행.
            # BacktestRecorder의 _save_to_excel을 사용하기 위해 reports 딕셔너리 생성
            reports = {'ohlc_signals': detailed_df_for_recorder.reset_index()}
            
            timestamp_str = datetime.now().strftime('%Y%m%d-%H%M%S')
            filename = f"cryptoSim_live_{timestamp_str}"
            recorder._save_to_excel(reports, filename)
            logging.info(f"실시간 거래 리포트가 {filename}.xlsx로 저장되었습니다.")

if __name__ == "__main__":
    main()