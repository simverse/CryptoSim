import ccxt
import logging
from src.config_manager import ConfigManager

class OKXExchange:
    """OKX 거래소 API 클래스"""
    
    def __init__(self, config_manager: ConfigManager):
        """OKXExchange 초기화"""
        self.config_manager = config_manager
        self.config = config_manager.config
        self.exchange = None
        self.logger = logging.getLogger(__name__)
        
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """거래소 초기화"""
        try:
            exchange_config = self.config_manager.get_exchange_config()
            
            self.exchange = ccxt.okx({
                'apiKey': exchange_config['api_key'],
                'secret': exchange_config['secret_key'],
                'password': exchange_config['passphrase'],
                'options': {
                    'defaultType': 'swap',
                },
                'enableRateLimit': True,
            })
            
            # config.yaml에서 trade_mode 값 가져오기
            trade_mode = exchange_config.get('trade_mode', 'demo') 
            self.logger.info(f"OKX 거래소 trade_mode: {trade_mode}")

            # trade_mode 값에 따른 추가 설정 (필요시 여기에 로직 추가)
            # self.exchange.options['defaultType'] = trade_mode; # 이 줄은 제거

            is_demo = exchange_config['sandbox']
            self.exchange.set_sandbox_mode(is_demo)
            
            self.logger.info(f"OKX 거래소 초기화 완료 (데모 모드: {is_demo})")
            
        except Exception as e:
            self.logger.error(f"OKX 거래소 초기화 실패: {e}", exc_info=True)
            raise

    def create_market_order(self, symbol: str, side: str, amount: float):
        """시장가 주문 생성"""
        try:
            return self.exchange.create_market_order(symbol, side, amount)
        except Exception as e:
            self.logger.error(f"시장가 주문 생성 실패: {e}", exc_info=True)
            raise

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params: dict = {}):
        """지정가 주문 생성"""
        try:
            return self.exchange.create_limit_order(symbol, side, amount, price, params)
        except Exception as e:
            self.logger.error(f"지정가 주문 생성 실패: {e}", exc_info=True)
            raise

    def fetch_order(self, order_id: str, symbol: str):
        """주문 조회"""
        try:
            return self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            self.logger.error(f"주문 조회 실패: {e}", exc_info=True)
            raise

    def cancel_order(self, order_id: str, symbol: str):
        """주문 취소"""
        try:
            return self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {e}", exc_info=True)
            raise

    def fetch_open_orders(self, symbol: str):
        """미체결 주문 목록 조회"""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"미체결 주문 조회 실패: {e}", exc_info=True)
            raise

    def fetch_position(self, symbol: str):
        """포지션 정보 조회"""
        try:
            # ccxt의 통합 fetch_positions 사용
            positions = self.exchange.fetch_positions([symbol])
            # 단일 심볼에 대한 포지션을 반환 (여러 포지션이 있을 경우 첫 번째 반환)
            return positions[0] if positions else None
        except Exception as e:
            self.logger.error(f"포지션 조회 실패: {e}", exc_info=True)
            raise

    def fetch_balance(self):
        """계좌 잔고 조회"""
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            self.logger.error(f"잔고 조회 실패: {e}", exc_info=True)
            raise
    
    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            self.fetch_balance()
            self.logger.info("OKX 연결 테스트 성공")
            return True
        except Exception as e:
            self.logger.error(f"OKX 연결 테스트 실패: {e}")
            return False
