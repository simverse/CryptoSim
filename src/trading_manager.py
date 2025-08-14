import logging
from datetime import datetime, timedelta
from typing import Dict, List
import uuid

from src.models import Position, PositionSide, Transaction, PositionSet # PositionSet 임포트
from src.exchange import Exchange

class TradingManager:
    """
    실시간 거래를 총괄하는 클래스.
    신호 처리, 주문 생성, 포지션 관리, 상태 복구 등을 담당합니다.
    """
    def __init__(self, config, exchange: Exchange):
        self.config = config
        self.exchange = exchange
        self.positions: Dict[PositionSide, Position] = {
            PositionSide.LONG: Position(side=PositionSide.LONG),
            PositionSide.SHORT: Position(side=PositionSide.SHORT)
        }
        self.open_orders: List[Dict] = []
        self.transaction_manager = {} # tx_id -> Transaction
        self.logger = logging.getLogger(__name__)

        # PositionSet 관련 설정 로드 (# config.yaml에서 설정)
        self.position_set_config = self.config.get('position_set', {})
        self.position_set_size_threshold = self.position_set_config.get('position_set_size') 
        self.max_long_sets = self.position_set_config.get('max_long_set')
        self.max_short_sets = self.position_set_config.get('max_short_set')

        # 디버깅을 위한 로그 추가
        self.logger.info(f"PositionSet 설정 - 크기 임계값: {self.position_set_size_threshold}, 최대 롱 세트: {self.max_long_sets}, 최대 숏 세트: {self.max_short_sets}")

        self.long_position_sets: List[PositionSet] = []
        self.short_position_sets: List[PositionSet] = []
        self.last_full_exit_order_info: Optional[Dict] = None # 전체 청산 주문 정보 저장

    def process_signal(self, signal_data: Dict):
        """
        최신 신호 데이터를 받아 거래를 실행합니다.
        """
        signal = signal_data.get('signal', 0)
        price = signal_data['close']
        symbol = self.config.get_trading_config()['symbol']
        amount = self.config.get_trading_config()['position_size']
        current_datetime = signal_data.name # pandas Timestamp (timezone-aware)

        if signal == 1:  # 롱 진입 신호
            if len(self.long_position_sets) >= self.max_long_sets and len(self.positions[PositionSide.LONG].transaction_ids) >= (self.position_set_size_threshold-1):
                self.logger.warning(f"최대 롱 포지션 한도 도달. 신규 롱 진입 주문을 생성하지 않습니다.")
                return
            self.logger.info(f"롱 진입 신호 수신. 가격: {price}")
            order = self.exchange.create_limit_order(symbol, PositionSide.LONG, 'open', amount, price, current_datetime)
            self.open_orders.append(order)
            self.logger.info(f"롱 진입 주문 생성: {order}")

        elif signal == -1:  # 숏 진입 신호
            if len(self.short_position_sets) >= self.max_short_sets and len(self.positions[PositionSide.SHORT].transaction_ids) >= (self.position_set_size_threshold-1):
                self.logger.warning(f"최대 숏 포지션 한도 도달. 신규 숏 진입 주문을 생성하지 않습니다.")
                return
            self.logger.info(f"숏 진입 신호 수신. 가격: {price}")
            order = self.exchange.create_limit_order(symbol, PositionSide.SHORT, 'open', amount, price, current_datetime)
            self.open_orders.append(order)
            self.logger.info(f"숏 진입 주문 생성: {order}")

        elif signal == 2:  # 롱 청산 신호
            self.logger.info(f"전략에 의한 롱 포지션 청산 신호 수신.")
            self._close_positions_by_signal(PositionSide.LONG, price, current_datetime)

        elif signal == -2:  # 숏 청산 신호
            self.logger.info(f"전략에 의한 숏 포지션 청산 신호 수신.")
            self._close_positions_by_signal(PositionSide.SHORT, price, current_datetime)

    def _close_positions_by_signal(self, side: PositionSide, price: float, timestamp: datetime):
        """전략 신호에 의해 해당 사이드의 모든 포지션을 지정가로 청산합니다."""
        position_to_close = self.positions[side]
        if position_to_close.is_empty():
            self.logger.info(f"{side.name} 포지션이 없어 청산을 건너뜁니다.")
            return

        symbol = self.config.get_trading_config()['symbol']
        total_amount_to_close = position_to_close.total_amount

        self.logger.info(f"{side.name} 포지션 전체({total_amount_to_close})를 지정가({price})로 청산 주문합니다.")
        params = {'clOrdId': f'exit_{side.value}'} # 포지션 전체 청산임을 나타내는 clOrdId
        order = self.exchange.create_limit_order(symbol, side, 'close', total_amount_to_close, price, timestamp, params)
        self.open_orders.append(order)
        self.logger.info(f"포지션 전체 청산 주문 생성: {order}")

    def check_open_orders(self, candle_data: Dict):
        """미체결 주문들을 확인하고 상태를 업데이트합니다."""
        # VirtualExchange의 process_candle_for_orders를 호출하여 체결된 주문을 가져옴
        filled_orders = self.exchange.process_candle_for_orders(candle_data)
        
        for order_status in filled_orders:
            if order_status['status'] == 'closed':
                self.logger.info(f"주문 체결 확인: {order_status['id']}")
                self._handle_filled_order(order_status)
            elif order_status['status'] == 'canceled':
                self.logger.info(f"주문이 취소되었습니다: {order_status['id']}")

    def _handle_filled_order(self, filled_order: Dict):
        """체결된 주문을 처리합니다."""
        client_order_id = filled_order.get('clientOrderId', '')
        
        if client_order_id.startswith('tp_'):
            # 포지션/세트 단위 익절 주문 처리
            self._handle_take_profit_order(filled_order)
        elif client_order_id.startswith('exit_long') or client_order_id.startswith('exit_short'):
            # 포지션 전체 청산 주문 처리 (전략 신호에 의한 청산)
            self._handle_full_position_exit_order(filled_order)
        elif client_order_id.startswith('exit_'):
            # 개별 거래 청산 주문 처리 (전략 또는 수동)
            original_tx_id = client_order_id.split('_', 1)[1]
            self._handle_exit_order(original_tx_id, filled_order)
        else:
            # 진입 주문 처리
            self._handle_entry_order(filled_order)

    def _handle_entry_order(self, filled_order: Dict):
        """체결된 '진입' 주문을 처리합니다."""
        position_side = filled_order['position_side'] # 주문에서 포지션 방향 가져오기
        
        # config_manager에서 trade_type 값 가져오기
        trade_type = self.config.get('exchange.trade_type') 

        transaction = Transaction.from_ccxt_order(filled_order, position_side, trade_type) # trade_type 전달
        self.transaction_manager[transaction.id] = transaction
        self.logger.info(f"Transaction {transaction.id} added to transaction_manager.")
        self.logger.info(f"Current transaction_manager keys: {list(self.transaction_manager.keys())}")

        # 잔고 업데이트
        self.exchange.update_balance_after_trade(filled_order, transaction)

        # 포지션에 거래를 추가하고, 즉시 익절 주문 업데이트
        position = self.positions[transaction.side]
        position.add_transaction(transaction)
        self.logger.info(f"새로운 거래가 {transaction.side.name} 포지션에 추가되었습니다: {transaction.id}")
        self._update_take_profit_order(position, datetime.fromisoformat(filled_order['datetime']))

        # PositionSet 관리 로직
        current_sets = self.long_position_sets if transaction.side == PositionSide.LONG else self.short_position_sets
        max_sets = self.max_long_sets if transaction.side == PositionSide.LONG else self.max_short_sets

        # 디버깅 로그 추가
        self.logger.info(f"PositionSet 체크 - {transaction.side.name} 포지션의 거래 수: {len(position.transaction_ids)}, 임계값: {self.position_set_size_threshold}")

        # 해당 방향의 총 거래 수가 PositionSet 생성 임계값과 같아졌을 때 새로운 Set 생성
        if len(position.transaction_ids) == self.position_set_size_threshold:
            self.logger.info(f"PositionSet 생성 조건 충족! {transaction.side.name} 포지션에 {len(position.transaction_ids)}개의 거래가 있습니다.")
            if len(current_sets) < max_sets:
                # PositionSet 생성 시, Position의 집계된 값을 그대로 사용
                new_set = PositionSet(
                    id=str(uuid.uuid4()),
                    side=position.side,
                    transaction_ids=list(position.transaction_ids), # 현재 모든 거래 ID를 포함
                    total_amount=position.total_amount,
                    avg_entry_price=position.weighted_avg_price,
                    total_margin=position.total_margin,
                    created_time=transaction.entry_time, # 생성 시점은 마지막 거래 시간
                    trade_type=position.trade_type
                )
                current_sets.append(new_set)
                self.logger.info(f"새로운 {transaction.side.name} PositionSet 생성: {new_set.id} ({len(new_set.transaction_ids)}개의 거래 포함)")
                # 새 PositionSet에 대한 익절 주문 생성
                self._update_take_profit_order(new_set, datetime.fromisoformat(filled_order['datetime']))

                # PositionSet 생성 후, 기존 포지션은 초기화하고 익절 주문도 취소
                self._update_take_profit_order(position, datetime.fromisoformat(filled_order['datetime'])) # 주문 취소 목적
                position.transaction_ids.clear()
                position.total_amount = 0.0
                position.total_margin = 0.0
                position.weighted_avg_price = 0.0
                position.take_profit_order_id = None # 익절 주문 ID 초기화
                self.logger.info(f"{transaction.side.name} 포지션이 초기화되고 거래들이 PositionSet으로 이동했습니다.")

            else:
                self.logger.warning(f"{transaction.side.name} PositionSet 최대 개수({max_sets}) 초과. 새로운 PositionSet을 생성하지 않습니다.")

    def _handle_take_profit_order(self, filled_order: Dict):
        """체결된 '익절' 주문을 처리합니다. 포지션 또는 세트 전체를 청산합니다."""
        client_order_id = filled_order['clientOrderId']
        target_id = client_order_id.split('_', 1)[1]
        exit_price = filled_order['average']
        exit_time = datetime.fromisoformat(filled_order['datetime'])
        exit_fee = filled_order['fee']['cost']
        position_side = filled_order['position_side']

        # Position 에서 찾기
        position = self.positions[position_side]
        if position.id == target_id: # Position의 ID가 UUID가 아니므로, side로 구분해야 함
            self.logger.info(f"{position.side.name} 포지션({position.id})의 익절 주문({filled_order['id']})이 체결되었습니다.")
            # 모든 거래를 청산 처리
            total_realized_pnl = 0.0
            for tx_id in list(position.transaction_ids):
                transaction = self.transaction_manager.get(tx_id)
                if transaction:
                    # 개별 거래의 PnL 계산 시에는 분배된 수수료를 사용
                    pnl = transaction.close_transaction(exit_price, exit_time, 'take_profit', exit_fee / len(position.transaction_ids))
                    total_realized_pnl += pnl
            
            # 포지션 초기화
            position.transaction_ids.clear()
            position.total_amount = 0.0
            position.total_margin = 0.0
            position.weighted_avg_price = 0.0
            position.take_profit_order_id = None

            # 전체 청산 주문의 PnL과 수수료를 한 번에 잔고에 반영
            # VirtualExchange의 update_balance_after_trade는 transaction 객체를 기대하므로,
            # realized_pnl과 fees를 포함하는 임시 객체를 생성하여 전달
            temp_transaction = Transaction(
                id=filled_order['id'],
                symbol=filled_order['symbol'],
                side=position_side,
                amount=filled_order['amount'],
                entry_price=filled_order['average'], # 임시로 사용
                entry_time=exit_time, # 임시로 사용
                margin=0.0, # 임시로 사용
                realized_pnl=total_realized_pnl,
                fees=exit_fee,
                is_closed=True
            )
            self.exchange.update_balance_after_trade(filled_order, temp_transaction)
            return

        # PositionSet 에서 찾기
        target_sets = self.long_position_sets if position_side == PositionSide.LONG else self.short_position_sets
        for p_set in target_sets:
            if p_set.id == target_id:
                self.logger.info(f"PositionSet({p_set.id})의 익절 주문({filled_order['id']})이 체결되었습니다.")
                p_set.close_set(exit_price, exit_time, 'take_profit')
                p_set.take_profit_order_id = None
                # 모든 거래를 청산 처리
                for tx_id in list(p_set.transaction_ids):
                    transaction = self.transaction_manager.get(tx_id)
                    if transaction:
                        transaction.close_transaction(exit_price, exit_time, 'take_profit_set', exit_fee / len(p_set.transaction_ids))
                        self.exchange.update_balance_after_trade(filled_order, transaction)
                # PositionSet 리스트에서 제거
                target_sets.remove(p_set)
                return

        self.logger.warning(f"익절 주문에 해당하는 포지션/세트를 찾지 못했습니다: {client_order_id}")

    def _handle_full_position_exit_order(self, filled_order: Dict):
        """전략 신호에 의해 체결된 포지션 전체 청산 주문을 처리합니다."""
        client_order_id = filled_order['clientOrderId']
        # client_order_id는 'exit_long' 또는 'exit_short' 형태
        position_side_str = client_order_id.split('_', 1)[1]
        position_side = PositionSide.LONG if position_side_str == 'long' else PositionSide.SHORT

        exit_price = filled_order['average']
        exit_time = datetime.fromisoformat(filled_order['datetime'])
        exit_fee = filled_order['fee']['cost']

        position = self.positions[position_side]

        if position.is_empty():
            self.logger.warning(f"{position_side.name} 포지션이 이미 비어있어 전체 청산 처리할 거래가 없습니다.")
            return

        self.logger.info(f"{position_side.name} 포지션 전체 청산 주문({filled_order['id']})이 체결되었습니다.")

        # 모든 개별 거래를 청산 처리
        total_realized_pnl = 0.0
        num_transactions = len(position.transaction_ids)
        
        # filled_order의 amount를 기준으로 각 transaction에 분배할 비율 계산
        # filled_order['amount']는 전체 청산 주문의 크기
        # position.total_amount는 현재 포지션의 총 크기
        # 이 둘은 같아야 하지만, 혹시 모를 오차를 위해 비율로 계산
        if position.total_amount > 0:
            ratio = filled_order['amount'] / position.total_amount
        else:
            ratio = 0.0

        for tx_id in list(position.transaction_ids):
            transaction = self.transaction_manager.get(tx_id)
            if transaction:
                # 각 transaction에 할당될 수수료
                fee_for_this_tx = exit_fee * (transaction.amount / position.total_amount) if position.total_amount > 0 else 0.0
                
                # transaction.close_transaction에 closed_amount 전달
                pnl = transaction.close_transaction(exit_price, exit_time, 'strategy_exit', fee_for_this_tx, transaction.amount)
                total_realized_pnl += pnl

        # 포지션 초기화
        position.transaction_ids.clear()
        position.total_amount = 0.0
        position.total_margin = 0.0
        position.weighted_avg_price = 0.0
        position.take_profit_order_id = None # 익절 주문 ID 초기화

        # 잔고 업데이트는 전체 청산 주문에 대해 한 번만 수행
        self.last_full_exit_order_info = {
            'id': filled_order['id'],
            'amount': filled_order['amount'],
            'side': position_side.value,
            'realized_pnl': total_realized_pnl
        }
        # 전체 청산의 PnL을 반영하기 위해 임시 Transaction 객체 생성
        # VirtualExchange의 update_balance_after_trade는 transaction 객체를 기대하므로,
        # realized_pnl과 fees를 포함하는 임시 객체를 생성하여 전달
        temp_transaction = Transaction(
            id=filled_order['id'],
            symbol=filled_order['symbol'],
            side=position_side,
            amount=filled_order['amount'],
            entry_price=filled_order['average'], # 임시로 사용
            entry_time=exit_time, # 임시로 사용
            margin=0.0, # 임시로 사용
            realized_pnl=total_realized_pnl,
            fees=exit_fee,
            is_closed=True
        )
        self.exchange.update_balance_after_trade(filled_order, temp_transaction)

        self.logger.info(f"{position_side.name} 포지션이 완전히 청산되었습니다. 총 실현 손익: {total_realized_pnl:.8f}")

    def _handle_exit_order(self, original_tx_id: str, filled_order: Dict):
        """체결된 '개별 청산' 주문을 처리합니다. (전략 신호 등)"""
        transaction_to_close = self.transaction_manager.get(original_tx_id)
        if not transaction_to_close:
            self.logger.warning(f"청산할 거래를 찾을 수 없습니다: {original_tx_id}")
            return

        position = self.positions[transaction_to_close.side]
        if not position.has_transaction(original_tx_id):
            self.logger.warning(f"거래({original_tx_id})가 이미 포지션에서 제거되었거나 다른 세트에 속해있습니다.")
            return

        # 1. Transaction 상태 업데이트 (PnL 계산)
        exit_time = datetime.fromisoformat(filled_order['datetime'])
        transaction_to_close.close_transaction(
            filled_order['average'], 
            exit_time,
            'strategy_exit', 
            filled_order['fee']['cost']
        )
        self.exchange.update_balance_after_trade(filled_order, transaction_to_close)

        # 2. 포지션에서 거래 제거
        position.remove_transaction(transaction_to_close)
        self.logger.info(f"거래({original_tx_id})가 {position.side.name} 포지션에서 제거되었습니다.")

        # 3. 포지션이 남아있다면, 익절 주문을 새로운 평균가와 수량으로 업데이트
        self._update_take_profit_order(position, exit_time)



    def _update_take_profit_order(self, position_or_set, timestamp: datetime):
        """
        포지션 또는 포지션 세트의 익절 주문을 업데이트합니다.
        기존 주문을 취소하고, 업데이트된 평균가와 수량으로 신규 주문을 생성합니다.
        """
        # 타입에 따라 속성 접근을 일반화
        if isinstance(position_or_set, Position):
            avg_price = position_or_set.weighted_avg_price
            total_amount = position_or_set.total_amount
            side = position_or_set.side
            order_id_attr = 'take_profit_order_id'
        elif isinstance(position_or_set, PositionSet):
            avg_price = position_or_set.avg_entry_price
            total_amount = position_or_set.total_amount
            side = position_or_set.side
            order_id_attr = 'take_profit_order_id'
        else:
            self.logger.error(f"Unsupported type for take profit order: {type(position_or_set)}")
            return

        # 0. 기존 익절 주문이 있으면 취소
        existing_order_id = getattr(position_or_set, order_id_attr)
        if existing_order_id:
            self.logger.info(f"기존 익절 주문 취소 시도: {existing_order_id}")
            self.exchange.cancel_order(existing_order_id)
            setattr(position_or_set, order_id_attr, None)

        # 포지션이 비어있으면 주문을 생성하지 않음
        if total_amount <= 0:
            self.logger.info(f"{side.name} 포지션이 비어있어 익절 주문을 생성하지 않습니다.")
            return

        # 1. 신규 익절 가격 계산
        if side == PositionSide.LONG:
            exit_price = avg_price * (1 + self.config.get('trading.take_profit_pct'))
        else:  # SHORT
            exit_price = avg_price * (1 - self.config.get('trading.take_profit_pct'))

        # 2. 신규 익절 주문 생성
        try:
            symbol = self.config.get_trading_config()['symbol']
            # Position/PositionSet ID를 clientOrderId에 포함시켜 추적
            cl_ord_id = f"tp_{position_or_set.id}"
            params = {'clOrdId': cl_ord_id}
            
            self.logger.info(f"신규 익절 주문 생성: {side.name} {total_amount} @ {exit_price}")

            tp_order = self.exchange.create_limit_order(
                symbol,
                side,
                'close',
                total_amount,
                exit_price,
                timestamp,
                params
            )
            
            # 3. 생성된 주문 ID를 포지션/세트에 저장
            new_order_id = tp_order['id']
            setattr(position_or_set, order_id_attr, new_order_id)
            self.open_orders.append(tp_order) # open_orders 리스트에도 추가
            self.logger.info(f"신규 익절 주문({new_order_id})이 {type(position_or_set).__name__}({position_or_set.id})에 등록되었습니다.")

        except Exception as e:
            self.logger.error(f"익절 주문 생성 실패: {e}")


    def close_all_positions(self, price: float, timestamp: datetime):
        """
        백테스트 종료 시 모든 포지션을 현재가로 청산합니다.
        """
        self.logger.info(f"백테스트 종료. 모든 포지션을 현재가({price})로 청산합니다.")
        symbol = self.config.get_trading_config()['symbol']

        # 모든 롱 포지션 청산
        for tx_id in list(self.positions[PositionSide.LONG].transaction_ids):
            transaction = self.transaction_manager.get(tx_id)
            if transaction and not transaction.is_closed:
                self.logger.info(f"롱 포지션 거래({tx_id})를 지정가({price})로 청산 주문합니다.")
                order = self.exchange.create_limit_order(symbol, PositionSide.LONG, 'close', transaction.amount, price, timestamp, {'clOrdId': f'exit_{transaction.id}'})
                self.open_orders.append(order)

        # 모든 숏 포지션 청산
        for tx_id in list(self.positions[PositionSide.SHORT].transaction_ids):
            transaction = self.transaction_manager.get(tx_id)
            if transaction and not transaction.is_closed:
                self.logger.info(f"숏 포지션 거래({tx_id})를 지정가({price})로 청산 주문합니다.")
                order = self.exchange.create_limit_order(symbol, PositionSide.SHORT, 'close', transaction.amount, price, timestamp, {'clOrdId': f'exit_{transaction.id}'})
                self.open_orders.append(order)
        
        # 생성된 청산 주문 즉시 처리
        self.check_open_orders({'high': price, 'low': price, 'close': price, 'timestamp': timestamp})

    def get_current_state(self):
        """현재 포지션과 미체결 주문 상태를 반환합니다."""
        return self.positions, self.open_orders, self.transaction_manager, self.long_position_sets, self.short_position_sets

    def restore_state(self, positions, open_orders, transactions, long_position_sets, short_position_sets):
        """이전 상태를 복구합니다."""
        self.positions = positions
        self.open_orders = open_orders
        self.transaction_manager = transactions
        self.long_position_sets = long_position_sets
        self.short_position_sets = short_position_sets
        self.logger.info("TradingManager 상태가 성공적으로 복구되었습니다.")
