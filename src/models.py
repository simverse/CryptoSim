from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict
import logging
import pytz

class PositionSide(Enum):
    """포지션 방향"""
    LONG = "long"
    SHORT = "short"

@dataclass
class Transaction:
    """개별 거래 (진입/청산 쌍)"""
    id: str
    symbol: str
    side: PositionSide
    amount: float  # BTC 수량
    entry_price: float
    entry_time: datetime
    margin: float  # 투입 마진 (BTC)
    leverage: int = 1
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    fees: float = 0.0  # 총 수수료 (BTC)
    realized_pnl: Optional[float] = None  # 실현 손익 (BTC)
    is_closed: bool = False
    cl_ord_id: Optional[str] = None
    trade_type: Optional[str] = None # trade_type 추가

    def calculate_unrealized_pnl(self, current_price: float, amount: Optional[float] = None) -> float:
        """미실현 손익 계산 (trade_mode에 따라 분기)"""
        if self.is_closed or self.entry_price == 0 or current_price == 0:
            return 0.0

        calc_amount = amount if amount is not None else self.amount

        # trade_type에 따라 PnL 계산식 분기
        if self.trade_type == '1' or self.trade_type == '2': # USDT-Margined 또는 USDC-Margined
            if self.side == PositionSide.LONG:
                # 롱(BTC) : Position크기(amount:BTC) * 진입가격 *(청산가격 -진입가격) / 청산가격
                return calc_amount * self.entry_price * (current_price - self.entry_price) / current_price
            else:  # SHORT
                # 숏(BTC) : Position크기(amount:BTC) * 진입가격 *(진입가격 - 청산가격) / 청산가격
                return calc_amount * self.entry_price * (self.entry_price - current_price) / current_price
        else: # Crypto-Margined (인버스 계약) 또는 기본값
            if self.side == PositionSide.LONG:
                # 롱(BTC) : Position크기(amount:BTC) * 진입가격 * (1/진입가격 - 1/청산가격)
                return calc_amount * self.entry_price * (1 / self.entry_price - 1 / current_price)
            else:  # SHORT
                # 숏(BTC) : Position크기(amount:BTC) * 진입가격 * (1/청산가격 - 1/진입가격)
                return calc_amount * self.entry_price * (1 / current_price - 1 / self.entry_price)

    def close_transaction(self, exit_price: float, exit_time: datetime, reason: str, exit_fee: float = 0.0, closed_amount: Optional[float] = None) -> float:
        """거래 청산"""
        if self.is_closed:
            return self.realized_pnl or 0.0

        logging.info(f"Closing transaction {self.id}: exit_time={exit_time}, exit_time.tzinfo={exit_time.tzinfo}")

        # exit_time이 naive datetime인 경우, Asia/Seoul 시간대로 변환
        if exit_time.tzinfo is None:
            korea_tz = pytz.timezone('Asia/Seoul')
            exit_time = korea_tz.localize(exit_time)

        # 실현 손익 먼저 계산
        # closed_amount가 제공되면 해당 수량만큼의 PnL을 계산
        pnl_amount = closed_amount if closed_amount is not None else self.amount
        unrealized_pnl = self.calculate_unrealized_pnl(exit_price, pnl_amount)
        self.fees += exit_fee
        self.realized_pnl = unrealized_pnl - self.fees

        # 상태 변경은 마지막에
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.is_closed = True
        
        return self.realized_pnl

    @classmethod
    def from_ccxt_order(cls, order: dict, position_side: PositionSide, trade_type: Optional[str] = None) -> 'Transaction':
        """ccxt의 주문 정보에서 Transaction 객체를 생성합니다."""
        utc_dt = datetime.fromtimestamp(order['timestamp'] / 1000, tz=pytz.utc)
        korea_tz = pytz.timezone('Asia/Seoul')
        entry_time = korea_tz.normalize(utc_dt.astimezone(korea_tz))
        logging.info(f"Creating transaction {order['id']}: entry_time={entry_time}, entry_time.tzinfo={entry_time.tzinfo}")
        return cls(
            id=order['id'],
            symbol=order['symbol'],
            side=position_side, # 명시적으로 전달받은 side 사용
            amount=order['filled'],
            entry_price=order['average'],
            entry_time=entry_time,
            margin=order.get('margin', 0.0), # margin 키를 사용
            leverage=order.get('leverage', 1),
            fees=order.get('fee', {}).get('cost', 0.0),
            is_closed=False,
            trade_type=trade_type # trade_type 전달
        )

@dataclass
class Position:
    """포지션 집계 (같은 방향의 거래들 합계)"""
    side: PositionSide
    id: str = field(init=False) # side.value 값으로 자동 할당
    total_amount: float = 0.0  # 총 수량 (BTC)
    total_margin: float = 0.0  # 총 마진 (BTC)
    weighted_avg_price: float = 0.0  # 가중평균 진입가
    transaction_ids: List[str] = field(default_factory=list)
    take_profit_order_id: Optional[str] = None  # 익절 주문 ID
    trade_type: Optional[str] = None # trade_type 추가

    def __post_init__(self):
        self.id = self.side.value
    
    def is_empty(self) -> bool:
        """포지션이 비어있는지 확인"""
        return self.total_amount == 0.0 or len(self.transaction_ids) == 0

    def has_transaction(self, tx_id: str) -> bool:
        """거래가 포지션에 포함되어 있는지 확인"""
        return tx_id in self.transaction_ids
    
    def add_transaction(self, transaction: Transaction):
        """거래 추가"""
        if transaction.side != self.side:
            raise ValueError(f"포지션 방향 불일치: {self.side} vs {transaction.side}")

        # Position의 trade_type 설정 (첫 Transaction의 trade_type를 따름)
        if self.trade_type is None:
            self.trade_type = transaction.trade_type
        elif self.trade_type != transaction.trade_type:
            logging.warning(f"Position의 trade_type({self.trade_type})와 Transaction의 trade_type({transaction.trade_type})가 다릅니다.")

        new_total_amount = self.total_amount + transaction.amount
        if new_total_amount == 0:
            self.weighted_avg_price = 0
        elif self.total_amount > 0 and self.weighted_avg_price > 0 and transaction.entry_price > 0:
            if self.trade_type == '1' or self.trade_type == '2': # USDT-Margined 또는 USDC-Margined (선형 계약)
                # 평균가격 =  (현재 규모 * 진입 가격 + 추가된 규모 * 추가된 규모의 진입 가격) /(현재 규모 + 추가된 규모)
                self.weighted_avg_price =  (self.total_amount * self.weighted_avg_price + transaction.amount * transaction.entry_price) / new_total_amount
            else: # Crypto-Margined (인버스 계약) 또는 기본값
                # 평균가격 = (현재 규모 + 추가된 규모) / (현재 규모 / 진입 가격 + 추가된 규모 / 추가된 규모의 진입 가격)
                self.weighted_avg_price = new_total_amount / (self.total_amount / self.weighted_avg_price + transaction.amount / transaction.entry_price)
        else:
            self.weighted_avg_price = transaction.entry_price

        self.total_amount += transaction.amount
        self.total_margin += transaction.margin
        self.transaction_ids.append(transaction.id)
    
    def remove_transaction(self, transaction: Transaction):
        """거래 제거"""
        if transaction.id not in self.transaction_ids:
            return
        
        self.total_amount -= transaction.amount
        self.total_margin -= transaction.margin
        self.transaction_ids.remove(transaction.id)
        
        # 포지션이 비워지면 초기화
        if self.total_amount <= 0:
            self.total_amount = 0.0
            self.total_margin = 0.0
            self.weighted_avg_price = 0.0
            self.trade_type = None # 포지션이 비워지면 trade_type 초기화
            self.transaction_ids.clear()
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """미실현 손익 계산 (trade_type에 따라 분기)"""
        if self.is_empty():
            return 0.0

        avg_price = self.weighted_avg_price
        if avg_price == 0 or current_price == 0:
            return 0.0

        # trade_type에 따라 PnL 계산식 분기
        if self.trade_type == '1' or self.trade_type == '2': # USDT-Margined 또는 USDC-Margined (선형 계약)
            if self.side == PositionSide.LONG:
                return self.total_amount * avg_price *(current_price - avg_price) /current_price
            else:  # SHORT
                return self.total_amount * avg_price *(avg_price - current_price) /current_price
        else: # Crypto-Margined (인버스 계약) 또는 기본값
            if self.side == PositionSide.LONG:
                # 롱: Position크기(amount) * 진입가격 * (1/진입가격 - 1/현재가격)
                return self.total_amount * avg_price * (1 / avg_price - 1 / current_price)
            else:  # SHORT
                # 숏: Position크기(amount) * 진입가격 * (1/현재가격 - 1/진입가격)
                return self.total_amount * avg_price * (1 / current_price - 1 / avg_price)

@dataclass
class PositionSet:
    """분리되어 독립 관리되는 포지션 세트"""
    id: str
    side: PositionSide  # LONG or SHORT
    transaction_ids: List[str]
    total_amount: float  # 총 BTC 수량
    avg_entry_price: float  # 가중평균 진입가
    total_margin: float  # 총 마진 (BTC)
    created_time: datetime
    is_closed: bool = False
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    realized_pnl: Optional[float] = None
    take_profit_order_id: Optional[str] = None  # 익절 주문 ID
    trade_type: Optional[str] = None # trade_type 추가
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """미실현 손익 계산 (trade_type에 따라 분기)"""
        if self.is_closed:
            return 0.0

        avg_price = self.avg_entry_price
        if avg_price == 0 or current_price == 0:
            return 0.0

        # trade_type에 따라 PnL 계산식 분기
        if self.trade_type == '1' or self.trade_type == '2': # USDT-Margined 또는 USDC-Margined
            if self.side == PositionSide.LONG:
                # 롱(BTC) : Position크기(amount:BTC) * 진입가격 *(청산가격 -진입가격) / 청산가격
                return self.total_amount * avg_price * (current_price - avg_price) / current_price
            else:  # SHORT
                # 숏(BTC) : Position크기(amount:BTC) * 진입가격 *(진입가격 - 청산가격) / 청산가격
                return self.total_amount * avg_price * (avg_price - current_price) / current_price
        else: # Crypto-Margined (인버스 계약) 또는 기본값
            if self.side == PositionSide.LONG:
                # 롱(BTC) : Position크기(amount:BTC) * 진입가격 * (1/진입가격 - 1/청산가격)
                return self.total_amount * avg_price * (1 / avg_price - 1 / current_price)
            else:  # SHORT
                # 숏: Position크기(amount) * 진입가격 * (1/현재가격 - 1/진입가격)
                return self.total_amount * avg_price * (1 / current_price - 1 / avg_price)

    def close_set(self, exit_price: float, exit_time: datetime, reason: str = 'set_exit') -> float:
        """포지션 세트 청산"""
        if self.is_closed:
            return 0.0
            
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.is_closed = True
        
        # 실현 손익 계산 (수수료는 별도 처리)
        unrealized_pnl = self.calculate_unrealized_pnl(exit_price)
        self.realized_pnl = unrealized_pnl
        
        return self.realized_pnl

    def to_report_string(self) -> str:
        """리포트 출력을 위한 문자열 형식 반환"""
        return f"{len(self.transaction_ids)}({self.avg_entry_price:.6f})"

class SignalType(Enum):
    """신호 타입"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class TradingSignal:
    """거래 신호"""
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)