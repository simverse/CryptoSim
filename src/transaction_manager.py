from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from src.models import PositionSide, Transaction


class TransactionManager:
    """개별 거래를 생성하고 관리"""
    
    def __init__(self):
        self.transactions: Dict[str, Transaction] = {}
        self.next_tx_id = 1

    def create_transaction(self, symbol: str, side: PositionSide, amount: float, 
                           entry_price: float, entry_time: datetime, margin: float, leverage: int) -> Transaction:
        """새로운 거래 생성"""
        tx_id = f"T{self.next_tx_id:06d}"
        self.next_tx_id += 1
        
        new_tx = Transaction(
            id=tx_id,
            symbol=symbol,
            side=side,
            amount=amount,
            entry_price=entry_price,
            entry_time=entry_time,
            margin=margin,
            leverage=leverage
        )
        self.transactions[tx_id] = new_tx
        return new_tx

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """거래 ID로 거래를 조회"""
        return self.transactions.get(transaction_id)

    def get_open_transactions(self) -> List[Transaction]:
        """열려있는 모든 거래를 반환"""
        return [tx for tx in self.transactions.values() if not tx.is_closed]

    def close_transaction(self, transaction_id: str, exit_price: float, exit_time: datetime, reason: str, exit_fee: float) -> Optional[float]:
        """거래를 청산하고 실현 손익을 반환"""
        transaction = self.get_transaction(transaction_id)
        if transaction:
            return transaction.close_transaction(exit_price, exit_time, reason, exit_fee)
        return None
