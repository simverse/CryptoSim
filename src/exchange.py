
import logging
from src.config_manager import ConfigManager
from src.okx_exchange import OKXExchange
from src.virtual_exchange import VirtualExchange

class Exchange:
    """
    거래 모드(실시간/백테스트)에 따라 적절한 거래소 인스턴스를 선택하고,
    일관된 인터페이스를 제공하는 Wrapper 클래스.
    """
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.logger = logging.getLogger(__name__)
        self.exchange_instance = self._get_exchange_instance()

    def _get_exchange_instance(self):
        """설정 파일의 run_mode에 따라 적절한 거래소 인스C턴스를 반환합니다."""
        run_mode = self.config.get('run_mode', {}).get('mode', 'backtest')
        
        if run_mode == 'live':
            self.logger.info("실시간 거래 모드로 Exchange를 초기화합니다.")
            return OKXExchange(self.config_manager)
        elif run_mode == 'backtest':
            self.logger.info("백테스트 모드로 Exchange를 초기화합니다.")
            initial_balance = self.config.get('backtest', {}).get('initial_balance', 1.0)
            return VirtualExchange(self.config, initial_balance)
        else:
            raise ValueError(f"지원하지 않는 실행 모드입니다: {run_mode}")

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params: dict = {}):
        """지정가 주문을 생성합니다."""
        return self.exchange_instance.create_limit_order(symbol, side, amount, price, params)

    def cancel_order(self, order_id: str, symbol: str):
        """주문을 취소합니다."""
        return self.exchange_instance.cancel_order(order_id, symbol)

    def fetch_open_orders(self, symbol: str):
        """미체결 주문 목록을 조회합니다."""
        return self.exchange_instance.fetch_open_orders(symbol)

    def fetch_balance(self):
        """계좌 잔고를 조회합니다."""
        return self.exchange_instance.fetch_balance()

    # 필요한 다른 모든 거래소 메서드를 여기에 래핑(wrapping)합니다.
    # 예: fetch_order, fetch_position 등
