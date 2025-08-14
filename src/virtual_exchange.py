
import logging
from datetime import datetime
from typing import Dict, List
from src.models import Transaction, PositionSide

class VirtualExchange:
    """
    백테스팅을 위한 가상 거래소 클래스.
    실제 거래소와 유사한 인터페이스를 제공하지만, 내부적으로는 상태만 관리합니다.
    """
    def __init__(self, config, initial_balance):
        self.config = config
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.transactions = {}
        self.open_orders = {} # 미체결 주문을 저장할 딕셔너리
        self.next_transaction_id = 1
        self.logger = logging.getLogger(__name__)

        self.leverage = self.config.get('trading.leverage')

    def _get_next_transaction_id(self):
        """다음 거래 ID를 생성합니다."""
        tx_id = self.next_transaction_id
        self.next_transaction_id += 1
        return str(tx_id)

    def create_limit_order(self, symbol: str, position_side: PositionSide, order_type: str, amount: float, price: float, timestamp: datetime, params: dict = {}):
        """
        지정가 주문을 시뮬레이션합니다.
        모든 주문은 미체결 상태로 추가되고, process_candle_for_orders에서 처리됩니다.
        """
        self.logger.info(f"[VIRTUAL] 지정가 주문 생성 요청: {position_side.value} {order_type} {amount} {symbol} @ {price}")

        order_id = f"{position_side.value}_{order_type}_{timestamp.strftime('%Y%m%d%H%M%S')}_{self._get_next_transaction_id()}"

        commission_rate = self.config.get('backtest.fees.commission')
        commission_usd = amount * price * commission_rate # USD 기준 수수료

        # USD 수수료를 BTC로 변환
        # 주문 가격(price)을 사용하여 변환
        commission_btc = commission_usd / price if price != 0 else 0.0
        margin = amount / self.leverage 

        order_info = {
            'id': order_id,
            'symbol': symbol,
            'position_side': position_side, # PositionSide.LONG or PositionSide.SHORT
            'order_type': order_type, # 'open' or 'close'
            'type': 'limit',
            'amount': amount,
            'price': price,
            'timestamp': int(timestamp.timestamp() * 1000),
            'datetime': timestamp.isoformat(),
            'margin': margin,
            'leverage': self.leverage ,
            'fee': {'cost': commission_btc, 'currency': 'BTC'},
            'status': 'open',
            'filled': 0,
            'average': None,
            'clientOrderId': params.get('clOrdId', '')
        }

        self.open_orders[order_id] = order_info
        self.logger.info(f"[VIRTUAL] 미체결 주문 추가: {order_id}")
        return order_info

    def update_balance_after_trade(self, filled_order: Dict, transaction: Transaction):
        """
        체결된 거래 정보를 바탕으로 가상 잔고를 업데이트합니다.
        """
        fee = filled_order['fee']['cost']
        
        if filled_order['order_type'] == 'open':
            # 진입 시: 수수료만 차감
            self.balance -= fee
            self.logger.info(f"[VIRTUAL] 진입 거래({filled_order['id']}) 후 잔고 업데이트: {self.balance:.8f} (수수료: -{fee:.8f})")
        else: # 'close'
            # 청산 시: 실현 손익을 잔고에 반영 (수수료는 이미 transaction.realized_pnl에 포함됨)
            # filled_order에 realized_pnl이 있으면 그것을 사용 (전체 포지션 청산의 경우)
            pnl = filled_order.get('realized_pnl', transaction.realized_pnl)
            self.balance += pnl
            self.logger.info(f"[VIRTUAL] 청산 거래({filled_order['id']}) 후 잔고 업데이트: {self.balance:.8f} (실현손익: {pnl:.8f})")

    def fetch_order(self, order_id: str, symbol: str) -> Dict:
        """가상 주문 조회"""
        if order_id in self.transactions:
            tx = self.transactions[order_id]
            return {
                'id': tx.id,
                'symbol': tx.symbol,
                'position_side': tx.side.value,
                'filled': tx.amount,
                'average': tx.entry_price,
                'timestamp': int(tx.entry_time.timestamp() * 1000),
                'datetime': tx.entry_time.isoformat(),
                'margin': tx.margin,
                'leverage': tx.leverage,
                'fee': {'cost': tx.fees, 'currency': 'BTC'},
                'status': 'closed',
                'clientOrderId': f'tp_{tx.id}' if tx.exit_reason == 'take_profit' else f'exit_{tx.id}' if tx.exit_reason else ''
            }
        elif order_id in self.open_orders:
            return self.open_orders[order_id]
        return {'id': order_id, 'status': 'canceled'} # 없는 주문은 취소된 것으로 처리

    def cancel_order(self, order_id: str, symbol: str = None) -> Dict:
        """미체결 주문을 취소합니다."""
        if order_id in self.open_orders:
            self.logger.info(f"[VIRTUAL] 주문 취소 요청: {order_id}")
            canceled_order = self.open_orders.pop(order_id)
            canceled_order['status'] = 'canceled'
            self.logger.info(f"[VIRTUAL] 주문 취소 완료: {order_id}")
            return canceled_order
        else:
            self.logger.warning(f"[VIRTUAL] 취소할 주문을 찾을 수 없습니다: {order_id}")
            # 실제 거래소와 유사하게, 찾을 수 없는 주문에 대한 정보를 반환할 수 있습니다.
            return {
                'id': order_id,
                'status': 'not_found',
                'message': 'Order not found in open orders.'
            }


    def process_candle_for_orders(self, candle_data: Dict) -> List[Dict]:
        """현재 캔들 데이터를 기반으로 미체결 주문을 처리합니다."""
        filled_orders = []
        orders_to_remove = []

        current_high = candle_data['high']
        current_low = candle_data['low']

        for order_id, order in list(self.open_orders.items()): # 순회 중 삭제를 위해 list로 복사
            order_price = order['price']
            order_type = order['order_type']
            position_side = order['position_side']

            # 매수 주문 체결 조건 (지정가 <= 캔들 저가)
            # 'open' 롱 포지션 또는 'close' 숏 포지션
            if (order_type == 'open' and position_side == PositionSide.LONG) or \
               (order_type == 'close' and position_side == PositionSide.SHORT):
                if order_price >= current_low:
                    self.logger.info(f"[VIRTUAL] Attempting to fill BUY order {order_id}. Order Price: {order_price}, Current Low: {current_low}")
                    order['status'] = 'closed'
                    order['filled'] = order['amount']
                    order['average'] = order_price # 지정가로 체결
                    order['timestamp'] = int(candle_data.name.timestamp() * 1000) # 체결 시간 업데이트
                    order['datetime'] = candle_data.name.isoformat() # 체결 시간 업데이트
                    filled_orders.append(order)
                    orders_to_remove.append(order_id)
                    self.logger.info(f"[VIRTUAL] 매수 주문 체결: {order_id} @ {order_price}, 체결량: {order['filled']}")

            # 매도 주문 체결 조건 (지정가 >= 캔들 고가)
            # 'open' 숏 포지션 또는 'close' 롱 포지션
            elif (order_type == 'open' and position_side == PositionSide.SHORT) or \
                 (order_type == 'close' and position_side == PositionSide.LONG):
                if order_price <= current_high:
                    self.logger.info(f"[VIRTUAL] Attempting to fill SELL order {order_id}. Order Price: {order_price}, Current High: {current_high}")
                    order['status'] = 'closed'
                    order['filled'] = order['amount']
                    order['average'] = order_price # 지정가로 체결
                    order['timestamp'] = int(candle_data.name.timestamp() * 1000) # 체결 시간 업데이트
                    order['datetime'] = candle_data.name.isoformat() # 체결 시간 업데이트
                    filled_orders.append(order)
                    orders_to_remove.append(order_id)
                    self.logger.info(f"[VIRTUAL] 매도 주문 체결: {order_id} @ {order_price}, 체결량: {order['filled']}")

        for order_id in orders_to_remove:
            if order_id in self.open_orders:
                del self.open_orders[order_id]

        return filled_orders
