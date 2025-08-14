#!/usr/bin/env python3
"""
올인원 백테스트 스크립트
모든 의존성을 하나의 파일에 통합하여 간단한 실행 환경 제공

사용법:
    python run_backtest_standalone.py --symbol BTC/USDT:USDT --timeframe 1h --limit 2000
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yaml
import logging
import time
import ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# python-dotenv가 설치되어 있으면 .env 파일 지원
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일에서 환경변수 로드
except ImportError:
    pass


# FutureWarning 해결을 위한 전역 설정
pd.set_option('future.no_silent_downcasting', True)


# =============================================================================
# 설정 관리 클래스
# =============================================================================

class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """ConfigManager 초기화"""
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """설정 파일 로드"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    self.config = yaml.safe_load(file)
            except Exception as e:
                print(f"설정 파일 로드 오류: {e}")
                self.create_default_config()
        else:
            print(f"설정 파일이 없습니다. 기본 설정을 생성합니다: {self.config_path}")
            self.create_default_config()
    
    def create_default_config(self) -> None:
        """기본 설정 생성"""
        self.config = {
            'exchange': {
                'name': 'okx',
                'api_key': '',
                'secret_key': '',
                'passphrase': '',
                'sandbox': True
            },
            'trading': {
                'symbol': 'BTC/USDT:USDT',
                'position_size': 0.01,
                'leverage': 1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            },
            'backtest': {
                'initial_balance': 1.0,
                'trading_period': {'hours': 2000},
                'fees': {'commission': 0.0001, 'funding_rate': 0.0001},
                'report': {'output_dir': 'logs'}
            }
        }
        self.save_config()
    
    def save_config(self) -> None:
        """설정 파일 저장"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"설정 파일 저장 오류: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정값 조회"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_exchange_config(self) -> Dict[str, Any]:
        """거래소 설정 조회 (환경변수 우선)"""
        return {
            'api_key': self._get_secure_value('OKX_API_KEY', 'exchange.api_key'),
            'secret_key': self._get_secure_value('OKX_SECRET_KEY', 'exchange.secret_key'),
            'passphrase': self._get_secure_value('OKX_PASSPHRASE', 'exchange.passphrase'),
            'sandbox': self._get_env_bool('OKX_SANDBOX', self.get('exchange.sandbox', True))
        }
    
    def _get_secure_value(self, env_key: str, config_key: str) -> str:
        """보안이 중요한 값을 환경변수 우선으로 조회"""
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
        
        config_value = self.get(config_key, '')
        if config_value:
            logging.warning(f"API 키가 설정 파일에 저장되어 있습니다. 보안을 위해 환경변수 사용을 권장합니다: {env_key}")
        
        return config_value
    
    def _get_env_bool(self, env_key: str, default: bool) -> bool:
        """환경변수에서 bool 값 조회"""
        env_value = os.getenv(env_key)
        if env_value is None:
            return default
        
        return env_value.lower() in ('true', '1', 'yes', 'on')
    
    def get_trading_config(self) -> Dict[str, Any]:
        """거래 설정 조회"""
        return {
            'symbol': self.get('trading.symbol', 'BTC/USDT:USDT'),
            'position_size': self.get('trading.position_size', 0.01),
            'leverage': self.get('trading.leverage', 1),
            'stop_loss_pct': self.get('trading.stop_loss_pct', 0.02),
            'take_profit_pct': self.get('trading.take_profit_pct', 0.04)
        }
    
    def get_backtest_config(self) -> Dict[str, Any]:
        """백테스트 설정 조회"""
        initial_balance_btc = self.get('backtest.initial_balance')
        if initial_balance_btc is None:
            raise ValueError("config.yaml에 'backtest.initial_balance' 설정이 없습니다.")
        
        return {
            'initial_balance_btc': initial_balance_btc,
            'trading_period': self.get('backtest.trading_period', {'hours': 2000}),
            'fees': self.get('backtest.fees', {'commission': 0.0001, 'funding_rate': 0.0001}),
            'report': self.get('backtest.report', {'output_dir': 'logs'})
        }

# =============================================================================
# OKX 거래소 클래스
# =============================================================================

class OKXExchange:
    """OKX 거래소 API 클래스"""
    
    def __init__(self, config_manager: ConfigManager, sandbox: bool = True):
        """OKXExchange 초기화"""
        self.config_manager = config_manager
        self.sandbox = sandbox
        self.exchange = None
        self.logger = logging.getLogger(__name__)
        
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """거래소 초기화"""
        try:
            exchange_config = self.config_manager.get_exchange_config()
            
            # OKX 거래소 초기화
            self.exchange = ccxt.okx({
                'apiKey': exchange_config['api_key'],
                'secret': exchange_config['secret_key'],
                'password': exchange_config['passphrase'],  # OKX는 passphrase 사용
                'sandbox': self.sandbox,  # 테스트넷 사용
                'enableRateLimit': True,
            })
            
            self.logger.info(f"OKX 거래소 초기화 완료 (샌드박스: {self.sandbox})")
            
        except Exception as e:
            self.logger.error(f"OKX 거래소 초기화 실패: {e}")
            raise
    
    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            # 연결 체크
            if self.exchange is None:
                self.logger.error("거래소가 초기화되지 않았습니다")
                return False
                
            # 계좌 정보 조회로 연결 테스트
            balance = self.exchange.fetch_balance()
            self.logger.info("OKX 연결 테스트 성공")
            return True
        except Exception as e:
            self.logger.error(f"OKX 연결 테스트 실패: {e}")
            return False

# =============================================================================
# 로깅 설정
# =============================================================================

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    ) 

# =============================================================================
# Transaction/Position 모델
# =============================================================================

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
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """미실현 손익 계산 (BTC 기준)"""
        if self.is_closed:
            return 0.0
            
        price_change = (current_price - self.entry_price) / self.entry_price
        
        if self.side == PositionSide.LONG:
            return self.amount * price_change
        else:  # SHORT
            return -self.amount * price_change
    
    def close_transaction(self, exit_price: float, exit_time: datetime, reason: str, exit_fee: float = 0.0) -> float:
        """거래 청산"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.fees += exit_fee
        self.is_closed = True
        
        # 실현 손익 계산
        unrealized_pnl = self.calculate_unrealized_pnl(exit_price)
        self.realized_pnl = unrealized_pnl - self.fees
        
        return self.realized_pnl

@dataclass 
class Position:
    """포지션 집계 (같은 방향의 거래들 합계)"""
    side: PositionSide
    total_amount: float = 0.0  # 총 수량 (BTC)
    total_margin: float = 0.0  # 총 마진 (BTC)
    weighted_avg_price: float = 0.0  # 가중평균 진입가
    transaction_ids: List[str] = field(default_factory=list)
    
    def is_empty(self) -> bool:
        """포지션이 비어있는지 확인"""
        return self.total_amount == 0.0 or len(self.transaction_ids) == 0
    
    def add_transaction(self, transaction: Transaction):
        """거래 추가"""
        if transaction.side != self.side:
            raise ValueError(f"포지션 방향 불일치: {self.side} vs {transaction.side}")
        
        # 가중평균 진입가 계산
        if self.total_amount > 0:
            total_value = (self.weighted_avg_price * self.total_amount) + (transaction.entry_price * transaction.amount)
            new_total_amount = self.total_amount + transaction.amount
            self.weighted_avg_price = total_value / new_total_amount
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
            self.transaction_ids.clear()
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """미실현 손익 계산 (BTC 기준)"""
        if self.is_empty():
            return 0.0
            
        price_change = (current_price - self.weighted_avg_price) / self.weighted_avg_price
        
        if self.side == PositionSide.LONG:
            return self.total_amount * price_change
        else:  # SHORT
            return -self.total_amount * price_change

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
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """미실현 손익 계산 (BTC 기준)"""
        if self.is_closed:
            return 0.0
            
        price_change = (current_price - self.avg_entry_price) / self.avg_entry_price
        
        if self.side == PositionSide.LONG:
            return self.total_amount * price_change
        else:  # SHORT
            return -self.total_amount * price_change
    
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

class TransactionManager:
    """거래 관리자"""
    
    def __init__(self):
        self.transactions: Dict[str, Transaction] = {}
        self.next_id = 1
    
    def create_transaction(self, symbol: str, side: PositionSide, amount: float, 
                         entry_price: float, entry_time: datetime, margin: float, leverage: int = 1) -> Transaction:
        """새 거래 생성"""
        transaction_id = f"T{self.next_id:06d}"
        self.next_id += 1
        
        transaction = Transaction(
            id=transaction_id,
            symbol=symbol,
            side=side,
            amount=amount,
            entry_price=entry_price,
            entry_time=entry_time,
            margin=margin,
            leverage=leverage
        )
        
        self.transactions[transaction_id] = transaction
        return transaction
    
    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """거래 조회"""
        return self.transactions.get(transaction_id)
    
    def get_open_transactions(self) -> List[Transaction]:
        """열린 거래 목록"""
        return [t for t in self.transactions.values() if not t.is_closed]
    
    def get_closed_transactions(self) -> List[Transaction]:
        """닫힌 거래 목록"""
        return [t for t in self.transactions.values() if t.is_closed]
    
    def close_transaction(self, transaction_id: str, exit_price: float, exit_time: datetime, reason: str, exit_fee: float = 0.0) -> Optional[float]:
        """거래 청산"""
        transaction = self.transactions.get(transaction_id)
        if not transaction or transaction.is_closed:
            return None
        
        return transaction.close_transaction(exit_price, exit_time, reason, exit_fee)

class PositionManager:
    """포지션 관리자"""
    
    def __init__(self):
        self.positions: Dict[PositionSide, Position] = {
            PositionSide.LONG: Position(side=PositionSide.LONG),
            PositionSide.SHORT: Position(side=PositionSide.SHORT)
        }
        # PositionSet 관리 추가
        self.position_sets: List[PositionSet] = []
        self.next_set_id = 1
    
    def get_position(self, side: PositionSide) -> Position:
        """포지션 조회"""
        return self.positions[side]
    
    def add_transaction(self, transaction: Transaction):
        """거래를 포지션에 추가"""
        position = self.positions[transaction.side]
        position.add_transaction(transaction)
    
    def remove_transaction(self, transaction: Transaction):
        """거래를 포지션에서 제거"""
        position = self.positions[transaction.side]
        position.remove_transaction(transaction)
    
    def get_total_margin(self) -> float:
        """총 마진 계산"""
        return sum(pos.total_margin for pos in self.positions.values())

    def create_position_set(self, side: PositionSide, transactions: List[Transaction], 
                        trigger_size: float) -> PositionSet:
        """포지션 세트 생성"""
        set_id = f"SET_{self.next_set_id:06d}"
        self.next_set_id += 1
        
        # 가중평균 진입가 계산
        total_value = sum(t.entry_price * t.amount for t in transactions)
        total_amount = sum(t.amount for t in transactions)
        avg_entry_price = total_value / total_amount
        
        position_set = PositionSet(
            id=set_id,
            side=side,
            transaction_ids=[t.id for t in transactions],
            total_amount=trigger_size,  # 실제 분리된 크기
            avg_entry_price=avg_entry_price,
            total_margin=sum(t.margin for t in transactions),
            created_time=datetime.now()
        )
        
        self.position_sets.append(position_set)
        return position_set

    def get_open_sets(self) -> List[PositionSet]:
        """열린 포지션 세트 목록"""
        return [ps for ps in self.position_sets if not ps.is_closed]

    def get_set_by_id(self, set_id: str) -> Optional[PositionSet]:
        """ID로 포지션 세트 조회"""
        for ps in self.position_sets:
            if ps.id == set_id:
                return ps
        return None

    def close_position_set(self, set_id: str, exit_price: float, 
                        exit_time: datetime, reason: str = 'set_exit') -> Optional[float]:
        """포지션 세트 청산"""
        position_set = self.get_set_by_id(set_id)
        if position_set and not position_set.is_closed:
            return position_set.close_set(exit_price, exit_time, reason)
        return None

    def get_total_set_pnl(self, current_price: float) -> float:
        """모든 포지션 세트의 총 미실현 손익"""
        return sum(ps.calculate_unrealized_pnl(current_price) for ps in self.get_open_sets())
    
    def check_position_set_trigger(self, side: PositionSide, position_size: float, 
                                 position_set_size: int, max_sets: int) -> bool:
        """PositionSet 생성 조건 확인"""
        position = self.get_position(side)
        
        # 현재 포지션이 position_set_size * position_size에 도달했는지 확인
        trigger_amount = position_size * position_set_size
        
        # 이미 max_sets에 도달했는지 확인
        current_open_sets = len([ps for ps in self.get_open_sets() if ps.side == side])
        
        return (position.total_amount >= trigger_amount and 
                len(position.transaction_ids) >= position_set_size and
                current_open_sets < max_sets)
    
    def create_position_set_from_position(self, side: PositionSide, position_size: float, 
                                        position_set_size: int, transaction_manager) -> Optional[PositionSet]:
        """Position에서 PositionSet 생성 및 Position 정리"""
        position = self.get_position(side)
        
        if position.is_empty() or len(position.transaction_ids) < position_set_size:
            return None
        
        # 가장 오래된 거래들부터 position_set_size만큼 선택
        transactions_for_set = position.transaction_ids[:position_set_size]
        trigger_amount = position_size * position_set_size
        
        # 실제 transaction들에서 가중평균 진입가와 마진 계산
        total_value = 0.0
        total_margin = 0.0
        set_transactions = []
        
        for transaction_id in transactions_for_set:
            transaction = transaction_manager.get_transaction(transaction_id)
            if transaction:
                set_transactions.append(transaction)
                total_value += transaction.entry_price * transaction.amount
                total_margin += transaction.margin
        
        if not set_transactions:
            return None
        
        # 가중평균 진입가 계산
        avg_entry_price = total_value / trigger_amount
        
        # PositionSet 생성
        set_id = f"SET_{self.next_set_id:06d}"
        self.next_set_id += 1
        
        position_set = PositionSet(
            id=set_id,
            side=side,
            transaction_ids=transactions_for_set,
            total_amount=trigger_amount,
            avg_entry_price=avg_entry_price,
            total_margin=total_margin,
            created_time=datetime.now()
        )
        
        self.position_sets.append(position_set)
        
        # Position에서 해당 거래들 제거
        position.transaction_ids = position.transaction_ids[position_set_size:]
        position.total_amount -= trigger_amount
        position.total_margin -= total_margin
        
        # 남은 거래들로 가중평균 진입가 재계산
        if position.transaction_ids:
            remaining_value = 0.0
            for transaction_id in position.transaction_ids:
                transaction = transaction_manager.get_transaction(transaction_id)
                if transaction:
                    remaining_value += transaction.entry_price * transaction.amount
            position.weighted_avg_price = remaining_value / position.total_amount if position.total_amount > 0 else 0.0
        else:
            # 모든 거래가 PositionSet으로 이동한 경우
            position.weighted_avg_price = 0.0
        
        return position_set

# =============================================================================
# 전략 기본 클래스
# =============================================================================

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

class BaseStrategy:
    """전략 기본 클래스"""
    
    def __init__(self, name: str, config: Dict, config_manager=None):
        self.name = name
        self.config = config
        self.config_manager = config_manager
        self.logger = logging.getLogger(f"Strategy.{name}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """신호 생성 (하위 클래스에서 구현)"""
        raise NotImplementedError
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 계산 (하위 클래스에서 구현)"""
        raise NotImplementedError

# =============================================================================
# SMA 크로스오버 전략
# =============================================================================

class SMACrossoverStrategy(BaseStrategy):
    """SMA 크로스오버 전략 클래스"""
    
    def __init__(self, config: Dict, config_manager=None):
        super().__init__("SMA Crossover Strategy", config, config_manager)
        
        # 전략 파라미터
        self.symbol = config.get('symbol', 'BTC/USD:BTC')
        self.short_sma_period = config.get('short_sma_period', 24)
        self.long_sma_period = config.get('long_sma_period', 720)
        
        # 포지션 설정
        self.leverage = config.get('leverage', 5)
        self.margin_per_trade = config.get('margin_per_trade', 0.02)
        self.position_size = config.get('position_size', 0.1)
        
        # 익절/손절 설정
        self.take_profit_pct = config.get('take_profit_pct', 0.02)
        self.stop_loss_enabled = config.get('stop_loss_enabled', True)
        
        # 리스크 관리
        self.max_concurrent_positions = config.get('max_concurrent_positions', 5)
        self.daily_loss_limit_pct = config.get('daily_loss_limit_pct', 0.05)
        
        # Transaction/Position 관리자 
        self.transaction_manager = TransactionManager()
        self.position_manager = PositionManager()
        
        logging.info(f"SMA 크로스오버 전략 초기화: {self.short_sma_period}/{self.long_sma_period}")
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        if len(df) < self.long_sma_period:
            logging.warning(f"데이터 부족: {len(df)}봉, 최소 필요: {self.long_sma_period}봉")
            return df
            
        # SMA 계산
        df[f'sma_{self.short_sma_period}'] = df['close'].rolling(window=self.short_sma_period).mean()
        df[f'sma_{self.long_sma_period}'] = df['close'].rolling(window=self.long_sma_period).mean()
        
        # 추세 판단
        df['short_sma_trend'] = df[f'sma_{self.short_sma_period}'].diff() > 0
        df['long_sma_trend'] = df[f'sma_{self.long_sma_period}'].diff() > 0
        
        # 정배열/역배열 판단
        df['golden_cross'] = df[f'sma_{self.short_sma_period}'] > df[f'sma_{self.long_sma_period}']
        df['dead_cross'] = df[f'sma_{self.short_sma_period}'] < df[f'sma_{self.long_sma_period}']
        
        # 크로스오버 감지
        golden_cross_prev = df['golden_cross'].shift(1).fillna(False).astype(bool)
        dead_cross_prev = df['dead_cross'].shift(1).fillna(False).astype(bool)
        
        df['golden_cross_signal'] = (df['golden_cross'] & ~golden_cross_prev).fillna(False)
        df['dead_cross_signal'] = (df['dead_cross'] & ~dead_cross_prev).fillna(False)
        
        # 가격 변화
        df['price_decline'] = df['close'] < df['close'].shift(1)
        df['price_increase'] = df['close'] > df['close'].shift(1)
        
        return df
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """거래 신호 생성"""
        df = self.calculate_indicators(df)
        
        if len(df) < self.long_sma_period + 1:
            df['signal'] = 0
            df['signal_type'] = 'none'
            return df
            
        # 신호 초기화
        df['signal'] = 0  # 0: 신호없음, 1: 롱, -1: 숏, 2: 롱청산, -2: 숏청산
        df['signal_type'] = 'none'
        
        for i in range(len(df)):
            if i < self.long_sma_period:
                continue
                
            current = df.iloc[i]
            
            # 롱 진입 조건
            if (current['golden_cross'] and current['long_sma_trend'] and 
                current['price_decline'] and 
                len(self.transaction_manager.get_open_transactions()) < self.max_concurrent_positions):
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'signal_type'] = 'long_entry'
                
            # 숏 진입 조건
            elif (current['dead_cross'] and not current['long_sma_trend'] and 
                  current['price_increase'] and 
                  len(self.transaction_manager.get_open_transactions()) < self.max_concurrent_positions):
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'signal_type'] = 'short_entry'
                
            # 롱 청산 조건
            elif (current['dead_cross_signal'] and not current['long_sma_trend'] and
                  not self.position_manager.get_position(PositionSide.LONG).is_empty()):
                df.loc[df.index[i], 'signal'] = 2
                df.loc[df.index[i], 'signal_type'] = 'long_exit'
                
            # 숏 청산 조건
            elif (current['golden_cross_signal'] and current['long_sma_trend'] and
                  not self.position_manager.get_position(PositionSide.SHORT).is_empty()):
                df.loc[df.index[i], 'signal'] = -2
                df.loc[df.index[i], 'signal_type'] = 'short_exit'
                
        return df
    
    def calculate_position_size(self, current_balance: float, current_price: float) -> Dict:
        """포지션 크기 계산"""
        return {
            'amount': self.position_size,
            'margin': self.margin_per_trade,
            'leverage': self.leverage
        } 

# =============================================================================
# 백테스트 결과 클래스
# =============================================================================

@dataclass
class BacktestResult:
    """백테스트 결과를 저장하는 데이터 클래스"""
    
    # 기본 정보
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    
    # 수익률 지표
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    
    # 거래 통계
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # 상세 데이터
    sortino_ratio: float = 0.0
    trades: List[Dict] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    detailed_records: List[Dict] = field(default_factory=list)

# =============================================================================
# 백테스터 클래스
# =============================================================================

class Backtester:
    """백테스트 엔진"""
    
    def __init__(self, initial_balance: float = 1.0):
        """Backtester 초기화"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.transaction_manager = TransactionManager()
        self.position_manager = PositionManager()
        
        # 기록용 데이터
        self.trades = []
        self.equity_curve = []
        self.detailed_records = []  # 시점별 상세 기록 추가
        
        # 통계
        self.total_commission = 0.0
        self.total_funding_fee = 0.0
        
        # 거래 제한
        self.max_concurrent_positions = 5
        self.daily_loss_limit_pct = 0.05
        
        # 수수료 및 펀딩비 설정
        self.commission = 0.0001
        self.funding_rate = 0.0001
        
        # 로거
        self.logger = logging.getLogger("Backtester")

    def run_backtest(self, strategy, data: pd.DataFrame, 
                     commission: float = 0.0001, funding_rate: float = 0.0001) -> BacktestResult:
        """백테스트 실행"""
        
        # 초기화
        self._reset()
        self.commission = commission
        self.funding_rate = funding_rate
        
        # 신호 생성
        signals_data = strategy.generate_signals(data.copy())
        
        # 백테스트 실행
        for i, (timestamp, row) in enumerate(signals_data.iterrows()):
            # 신호 처리
            self._process_signal(row, strategy)
            
            # 펀딩비 적용 (매 8시간)
            if i % 8 == 0:
                self._apply_funding_fee(row)
            
            # 이익 실현 체크 (고가/저가 기준)
            self._check_take_profit(row['high'], row['low'], row['close'], timestamp, strategy)
            
            # 자산 곡선 업데이트
            self._update_equity_curve(row)
            
            # 상세 기록 추가
            self._record_detailed_data(timestamp, row, strategy)
        
        # 마지막에 모든 포지션 청산
        last_price = signals_data['close'].iloc[-1]
        self._close_all_positions(last_price)
        
        # 결과 계산 및 반환
        result = self._calculate_results(strategy.name, data)
        result.signals_data = signals_data
        result.detailed_records = self.detailed_records  # 상세 기록 추가
        
        return result

    def _reset(self):
        """백테스터 초기화"""
        self.current_balance = self.initial_balance
        self.total_commission = 0.0
        self.total_funding_fee = 0.0
        
        self.transaction_manager = TransactionManager()
        self.position_manager = PositionManager()
        self.trades.clear()
        self.equity_curve.clear()
        self.detailed_records.clear()
        
    def _process_signal(self, row, strategy):
        """신호 처리"""
        signal = row.get('signal', 0)
        timestamp = row.name
        price = row['close']
        
        # 디버깅: 신호 발생 로그 (첫 20개만)
        if len(self.detailed_records) < 20:
            self.logger.info(f"신호 처리 - timestamp: {timestamp}, signal: {signal}, price: {price}")
        
        if signal == 1:  # 롱 진입
            self.logger.info(f"롱 진입 신호 - timestamp: {timestamp}, price: {price}")
            self._open_position('long', price, timestamp, strategy)
        elif signal == -1:  # 숏 진입
            self.logger.info(f"숏 진입 신호 - timestamp: {timestamp}, price: {price}")
            self._open_position('short', price, timestamp, strategy)
        elif signal == 2:  # 롱 청산
            self._close_long_positions(price, timestamp, 'strategy_exit')
        elif signal == -2:  # 숏 청산
            self._close_short_positions(price, timestamp, 'strategy_exit')
            
    def _open_position(self, side: str, price: float, timestamp, strategy):
        """포지션 진입"""
        try:
            # 디버깅: 포지션 생성 시작 로그
            self.logger.info(f"포지션 생성 시작 - side: {side}, price: {price}, timestamp: {timestamp}")
            
            # 포지션 크기 계산
            position_info = strategy.calculate_position_size(self.current_balance, price)
            amount = position_info['amount']
            margin = position_info['margin']
            leverage = position_info.get('leverage', 1)
            
            # 디버깅: 포지션 크기 정보
            self.logger.info(f"포지션 크기 계산 - amount: {amount}, margin: {margin}, leverage: {leverage}")
            
            # 마진 확인
            if margin > self.current_balance:
                self.logger.warning(f"마진 부족: 필요 {margin:.6f} BTC, 보유 {self.current_balance:.6f} BTC")
                return
            
            # 수수료 계산 (BTC 기준)
            entry_fee = amount * self.commission
            
            # Transaction 생성
            position_side = PositionSide.LONG if side == 'long' else PositionSide.SHORT
            transaction = self.transaction_manager.create_transaction(
                symbol=strategy.symbol,
                side=position_side,
                amount=amount,
                entry_price=price,
                entry_time=timestamp,
                margin=margin,
                leverage=leverage
            )
            
            # 디버깅: Transaction 생성 확인
            self.logger.info(f"Transaction 생성 완료 - ID: {transaction.id}, timestamp: {transaction.entry_time}")
            
            # 수수료 적용
            transaction.fees = entry_fee
            self.current_balance -= entry_fee
            self.total_commission += entry_fee
            
            # 마진 차감
            self.current_balance -= margin
            
            # 포지션 관리자에 추가
            self.position_manager.add_transaction(transaction)
            
            # 디버깅: 포지션 관리자에 추가 후 확인
            self.logger.info(f"포지션 관리자에 추가 완료 - Transaction 수: {len(self.transaction_manager.transactions)}")
            
            # 디버깅: 포지션 추가 후 상태 확인 (첫 10개 거래만)
            position_after = self.position_manager.get_position(position_side)
            if len(self.transaction_manager.transactions) <= 10:
                self.logger.info(f"포지션 추가 후 - {side.upper()}: total_amount={position_after.total_amount}, transaction_ids={position_after.transaction_ids}")
            
            # PositionSet 생성 조건 확인 및 처리
            self._check_and_create_position_set(position_side, strategy)
            
            self.logger.info(f"{side.upper()} 진입: {amount:.6f} BTC @ ${price:.2f}")
            
        except Exception as e:
            self.logger.error(f"포지션 진입 실패: {e}")
    
    def _check_and_create_position_set(self, side: PositionSide, strategy):
        """PositionSet 생성 조건 확인 및 생성"""
        try:
            # config에서 PositionSet 설정 읽기
            if strategy.config_manager and hasattr(strategy.config_manager, 'config'):
                position_set_config = strategy.config_manager.config.get('position_set', {})
                position_set_size = position_set_config.get('position_set_size', 10)
                max_long_set = position_set_config.get('max_long_set', 3)
                max_short_set = position_set_config.get('max_short_set', 2)
            else:
                # 기본값 사용
                position_set_size = 10
                max_long_set = 3
                max_short_set = 2
            
            # 거래 크기
            position_size = strategy.config.get('position_size', 0.005)
            
            # 해당 side의 max_sets 결정
            max_sets = max_long_set if side == PositionSide.LONG else max_short_set
            
            # PositionSet 생성 조건 확인
            if self.position_manager.check_position_set_trigger(side, position_size, position_set_size, max_sets):
                # PositionSet 생성
                position_set = self.position_manager.create_position_set_from_position(
                    side, position_size, position_set_size, self.transaction_manager)
                if position_set:
                    self.logger.info(f"PositionSet 생성: {position_set.id} ({side.name}) - {position_set.total_amount:.6f} BTC")
                    
        except Exception as e:
            self.logger.error(f"PositionSet 체크 실패: {e}")
            
    def _close_position(self, transaction_id: str, price: float, timestamp, reason: str):
        """개별 거래 청산"""
        transaction = self.transaction_manager.get_transaction(transaction_id)
        if not transaction or transaction.is_closed:
            return
        
        # 수수료 계산
        exit_fee = transaction.amount * self.commission
        
        # 거래 청산
        realized_pnl = self.transaction_manager.close_transaction(
            transaction_id, price, timestamp, reason, exit_fee
        )
        
        if realized_pnl is not None:
            # 잔고 업데이트 (마진 반환 + 실현손익)
            self.current_balance += transaction.margin + realized_pnl
            self.total_commission += exit_fee
            
            # 포지션 관리자에서 제거
            self.position_manager.remove_transaction(transaction)
            
            # 거래 기록 저장
            trade_record = {
                'position_id': transaction.id,
                'side': transaction.side.value,
                'amount': transaction.amount,
                'entry_price': transaction.entry_price,
                'exit_price': transaction.exit_price,
                'entry_time': transaction.entry_time,
                'exit_time': transaction.exit_time,
                'margin': transaction.margin,
                'fees': transaction.fees,
                'gross_pnl': realized_pnl + transaction.fees,  # 수수료 제외 전
                'net_pnl': realized_pnl,  # 수수료 포함 실제 손익
                'exit_reason': reason
            }
            self.trades.append(trade_record)
            
            self.logger.info(f"{transaction.side.value.upper()} 청산: {transaction.amount:.6f} BTC @ ${price:.2f}, PnL: {realized_pnl:+.6f} BTC")
    
    def _close_long_positions(self, price: float, timestamp, reason: str):
        """모든 롱 포지션 청산"""
        open_transactions = self.transaction_manager.get_open_transactions()
        long_transactions = [t for t in open_transactions if t.side == PositionSide.LONG]
        
        for transaction in long_transactions:
            self._close_position(transaction.id, price, timestamp, reason)
    
    def _close_short_positions(self, price: float, timestamp, reason: str):
        """모든 숏 포지션 청산"""
        open_transactions = self.transaction_manager.get_open_transactions()
        short_transactions = [t for t in open_transactions if t.side == PositionSide.SHORT]
        
        for transaction in short_transactions:
            self._close_position(transaction.id, price, timestamp, reason)
    
    def _close_all_positions(self, price: float):
        """모든 포지션 청산"""
        open_transactions = self.transaction_manager.get_open_transactions()
        timestamp = datetime.now()
        
        for transaction in open_transactions:
            self._close_position(transaction.id, price, timestamp, 'backtest_end')
    
    def _check_take_profit(self, high_price: float, low_price: float, close_price: float, timestamp, strategy):
        """익절 확인 - 고가/저가 기준으로 Position과 PositionSet 익절 체크
        
        주의: 같은 봉에서 진입한 거래들은 체크에서 제외 (Look-ahead bias 방지)
        """
        # 1. PositionSet 익절 체크 (우선)
        self._check_position_set_take_profit(high_price, low_price, timestamp, strategy)
        
        # 2. Position 기준 익절 체크 (남은 포지션들)
        self._check_position_take_profit(high_price, low_price, timestamp, strategy)
    
    def _check_position_set_take_profit(self, high_price: float, low_price: float, timestamp, strategy):
        """PositionSet 익절 확인 - 같은 봉 생성 세트 제외"""
        open_sets = self.position_manager.get_open_sets().copy()  # 복사본으로 작업
        current_timestamp_str = str(timestamp)
        
        for position_set in open_sets:
            # 같은 봉에서 생성된 PositionSet은 익절 체크에서 제외
            if str(position_set.created_time) != current_timestamp_str:
                should_exit, exit_price = self._check_set_exit_conditions(position_set, high_price, low_price, strategy)
                if should_exit:
                    # PositionSet 전체 청산
                    realized_pnl = self.position_manager.close_position_set(position_set.id, exit_price, timestamp, 'take_profit')
                    if realized_pnl is not None:
                        # PositionSet에 포함된 모든 거래들도 실제로 청산
                        for transaction_id in position_set.transaction_ids:
                            self._close_position(transaction_id, exit_price, timestamp, 'position_set_take_profit')
                        self.logger.info(f"PositionSet 익절: {position_set.id} @ ${exit_price:.2f}, PnL: {realized_pnl:.6f} BTC")
    
    def _check_position_take_profit(self, high_price: float, low_price: float, timestamp, strategy):
        """Position (개별 포지션) 익절 확인 - 같은 봉 진입 거래 제외"""
        for side in [PositionSide.LONG, PositionSide.SHORT]:
            position = self.position_manager.get_position(side)
            if not position.is_empty():
                # 같은 봉에서 진입한 거래들 제외 체크
                eligible_transactions = []
                current_timestamp_str = str(timestamp)
                
                for transaction_id in position.transaction_ids:
                    transaction = self.transaction_manager.get_transaction(transaction_id)
                    if transaction and not transaction.is_closed:
                        # 같은 봉에서 진입한 거래는 제외
                        if str(transaction.entry_time) != current_timestamp_str:
                            eligible_transactions.append(transaction)
                
                # 체크 대상 거래가 있을 때만 익절 확인
                if eligible_transactions:
                    should_exit, exit_price = self._check_position_exit_conditions(position, high_price, low_price, strategy)
                    if should_exit:
                        # Position 전체 청산
                        if side == PositionSide.LONG:
                            self._close_long_positions(exit_price, timestamp, 'take_profit')
                        else:
                            self._close_short_positions(exit_price, timestamp, 'take_profit')
    
    def _check_set_exit_conditions(self, position_set: PositionSet, high_price: float, low_price: float, strategy) -> Tuple[bool, float]:
        """PositionSet 청산 조건 확인"""
        take_profit_pct = getattr(strategy, 'take_profit_pct', strategy.config.get('take_profit_pct', 0.04))
        
        if position_set.side == PositionSide.LONG:
            # 롱: 고가 기준으로 익절 확인
            profit_pct = (high_price - position_set.avg_entry_price) / position_set.avg_entry_price
            if profit_pct >= take_profit_pct:
                return True, high_price
        else:  # SHORT
            # 숏: 저가 기준으로 익절 확인
            profit_pct = (position_set.avg_entry_price - low_price) / position_set.avg_entry_price
            if profit_pct >= take_profit_pct:
                return True, low_price
        
        return False, 0.0
    
    def _check_position_exit_conditions(self, position: Position, high_price: float, low_price: float, strategy) -> Tuple[bool, float]:
        """Position 청산 조건 확인"""
        take_profit_pct = getattr(strategy, 'take_profit_pct', strategy.config.get('take_profit_pct', 0.04))
        
        if position.side == PositionSide.LONG:
            # 롱: 고가 기준으로 익절 확인
            profit_pct = (high_price - position.weighted_avg_price) / position.weighted_avg_price
            if profit_pct >= take_profit_pct:
                return True, high_price
        else:  # SHORT
            # 숏: 저가 기준으로 익절 확인
            profit_pct = (position.weighted_avg_price - low_price) / position.weighted_avg_price
            if profit_pct >= take_profit_pct:
                return True, low_price
        
        return False, 0.0
    
    def _check_transaction_exit_conditions(self, transaction: Transaction, current_price: float, strategy) -> Tuple[bool, str]:
        """개별 거래 청산 조건 확인"""
        entry_price = transaction.entry_price
        
        # 익절 조건 확인
        if transaction.side == PositionSide.LONG:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= strategy.take_profit_pct:
                return True, 'take_profit'
        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= strategy.take_profit_pct:
                return True, 'take_profit'
        
        return False, ''
    
    def _apply_funding_fee(self, row):
        """펀딩피 적용"""
        timestamp = row.name
        current_price = row['close']
        
        if hasattr(timestamp, 'hour'):
            hour = timestamp.hour
            if hour in [0, 8, 16]:  # UTC 기준 펀딩피 시각
                total_funding_fee_btc = 0.0
                
                for transaction in self.transaction_manager.get_open_transactions():
                    notional_value_usd = transaction.amount * current_price
                    funding_fee_usd = notional_value_usd * self.funding_rate
                    funding_fee_btc = funding_fee_usd / current_price
                    total_funding_fee_btc += funding_fee_btc
                
                if total_funding_fee_btc > 0:
                    self.current_balance -= total_funding_fee_btc
                    self.total_funding_fee += total_funding_fee_btc
    
    def _calculate_total_equity(self, current_price: float) -> float:
        """총 자산 계산 (잔고 + 포지션 평가손익)"""
        total_equity = self.current_balance
        
        for transaction in self.transaction_manager.get_open_transactions():
            unrealized_pnl = transaction.calculate_unrealized_pnl(current_price)
            total_equity += unrealized_pnl
            
        return total_equity
    
    def _update_equity_curve(self, row):
        """자산 곡선 업데이트"""
        current_price = row['close']
        timestamp = row.name
        total_equity = self._calculate_total_equity(current_price)
        
        open_transactions = self.transaction_manager.get_open_transactions()
        long_position_amount = sum(t.amount for t in open_transactions if t.side == PositionSide.LONG)
        short_position_amount = sum(t.amount for t in open_transactions if t.side == PositionSide.SHORT)
        
        equity_point = {
            'timestamp': timestamp,
            'price': current_price,
            'balance': self.current_balance,
            'total_equity': total_equity,
            'open_positions': len(open_transactions),
            'long_position_amount': long_position_amount,
            'short_position_amount': short_position_amount
        }
        
        self.equity_curve.append(equity_point)
        
    def _calculate_results(self, strategy_name: str, data: pd.DataFrame) -> BacktestResult:
        """백테스트 결과 계산"""
        
        # 날짜 타입 안전 변환
        start_date = pd.to_datetime(data.index[0]).to_pydatetime() if hasattr(data.index[0], 'to_pydatetime') else pd.to_datetime(data.index[0])
        end_date = pd.to_datetime(data.index[-1]).to_pydatetime() if hasattr(data.index[-1], 'to_pydatetime') else pd.to_datetime(data.index[-1])
        
        if not self.trades:
            return BacktestResult(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                initial_balance=self.initial_balance,
                final_balance=self.current_balance,
                total_return=0.0,
                annual_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                trades=[],
                equity_curve=pd.DataFrame(self.equity_curve)
            )
            
        # 기본 통계
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # 수익률 계산
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        
        # 기간 계산  
        period_days = (end_date - start_date).days
        annual_return = (1 + total_return) ** (365 / period_days) - 1 if period_days > 0 else 0
        
        # 최대 낙폭 계산
        equity_df['peak'] = equity_df['total_equity'].cummax()
        equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = float(equity_df['drawdown'].min())
        
        # 샤프 비율 계산
        equity_df['returns'] = equity_df['total_equity'].pct_change()
        if equity_df['returns'].std() > 0:
            sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252 * 24)
        else:
            sharpe_ratio = 0.0
            
        # 거래 통계
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        win_rate = winning_trades / len(trades_df) if len(trades_df) > 0 else 0
        
        avg_win = float(trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean()) if winning_trades > 0 else 0.0
        avg_loss = float(trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean()) if losing_trades > 0 else 0.0
        
        total_gains = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum() if winning_trades > 0 else 0
        total_losses = abs(trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].sum()) if losing_trades > 0 else 0
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=self.current_balance,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=0.0,  # 간단히 0으로 설정
            total_trades=len(trades_df),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=self.trades,
            equity_curve=equity_df
        )

    def _record_detailed_data(self, timestamp, row, strategy):
        """시점별 상세 데이터 기록"""
        current_price = row['close']
        
        # 현재 포지션 정보
        long_position = self.position_manager.get_position(PositionSide.LONG)
        short_position = self.position_manager.get_position(PositionSide.SHORT)
        
        # 미실현 손익 계산
        unrealized_pnl = 0.0
        open_transactions = self.transaction_manager.get_open_transactions()
        for transaction in open_transactions:
            unrealized_pnl += transaction.calculate_unrealized_pnl(current_price)
        
        # 총 자산 계산
        total_equity = self._calculate_total_equity(current_price)
        
        # 수익률 계산
        current_return = (total_equity - self.initial_balance) / self.initial_balance * 100
        
        # 누적 수익률 계산
        cumulative_return = current_return
        
        # 신호 정보
        signal = row.get('signal', 0)
        buy_signal = ''
        sell_signal = ''
        trading_position = ''
        
        if signal == 1:
            buy_signal = '롱진입'
            trading_position = '롱'
        elif signal == -1:
            buy_signal = '숏진입'
            trading_position = '숏'
        elif signal == 2:
            sell_signal = '롱청산'
        elif signal == -2:
            sell_signal = '숏청산'
        
        # 거래 정보 (실제 거래가 발생한 경우에만 기록)
        position_id = None
        trade_direction = None
        trade_price = None
        trade_size = None  # 개별 거래 크기
        
        # 실제 포지션 변화가 있는 경우에만 거래 정보 기록
        # 현재 timestamp에서 생성된 최신 거래들 확인
        current_transactions = []
        all_transactions = self.transaction_manager.get_open_transactions()
        
        # 디버깅: 전체 거래 상태 확인 (첫 10개 기록만)
        if len(self.detailed_records) < 10:
            self.logger.info(f"거래 기록 체크 - timestamp: {timestamp}")
            self.logger.info(f"  전체 열린 거래 수: {len(all_transactions)}")
            for i, t in enumerate(all_transactions[-3:]):  # 최근 3개만
                self.logger.info(f"  거래 {i}: ID={t.id}, timestamp={t.entry_time}, side={t.side}")
        
        # timestamp 비교 (pandas Timestamp 객체 직접 비교)
        for t in all_transactions:
            if pd.Timestamp(t.entry_time) == pd.Timestamp(timestamp):
                current_transactions.append(t)
        
        if current_transactions:
            # 가장 최근 생성된 거래 정보 사용
            latest_transaction = current_transactions[-1]
            position_id = latest_transaction.id
            trade_direction = '롱' if latest_transaction.side == PositionSide.LONG else '숏'
            trade_price = latest_transaction.entry_price
            trade_size = latest_transaction.amount
            
            # 디버깅: 첫 10개 거래 정보 로그
            if len(self.detailed_records) < 10:
                self.logger.info(f"거래 기록 발견! - timestamp: {timestamp}")
                self.logger.info(f"  거래 ID: {position_id}, 방향: {trade_direction}")
                self.logger.info(f"  가격: {trade_price}, 크기: {trade_size}")
        else:
            # 디버깅: 거래가 발견되지 않은 경우
            if len(self.detailed_records) < 10:
                self.logger.info(f"거래 기록 없음 - timestamp: {timestamp}")
        
        # 롱 포지션 상세 정보
        long_size = long_position.total_amount if not long_position.is_empty() else 0.0
        long_sets_info = "0"
        long_avg_price = 0.0
        long_pnl = 0.0
        long_value = 0.0
        
        if not long_position.is_empty():
            # 롱 포지션 Set 정보 계산
            long_open_sets = [ps for ps in self.position_manager.get_open_sets() if ps.side == PositionSide.LONG]
            long_sets_count = len(long_open_sets)
            long_sets_total = sum(ps.total_amount for ps in long_open_sets)
            long_sets_info = f"{long_sets_count}({long_sets_total:.6f})" if long_sets_count > 0 else f"0({long_position.total_amount:.6f})"
            
            long_avg_price = long_position.weighted_avg_price
            long_pnl = long_position.calculate_unrealized_pnl(current_price)
            # 롱 가치 = 롱크기 + 롱 PnL
            long_value = long_size + long_pnl
        
        # 숏 포지션 상세 정보
        short_size = short_position.total_amount if not short_position.is_empty() else 0.0
        short_sets_info = "0"
        short_avg_price = 0.0
        short_pnl = 0.0
        short_value = 0.0
        
        # 디버깅: 숏 포지션 정보 로그 (첫 10개 레코드만)
        if len(self.detailed_records) < 10:
            self.logger.info(f"SHORT 포지션 체크 - timestamp: {timestamp}")
            self.logger.info(f"  short_position.is_empty(): {short_position.is_empty()}")
            self.logger.info(f"  short_position.total_amount: {short_position.total_amount}")
            self.logger.info(f"  short_position.transaction_ids: {short_position.transaction_ids}")
            self.logger.info(f"  short_position.weighted_avg_price: {short_position.weighted_avg_price}")
        
        if not short_position.is_empty():
            # 숏 포지션 Set 정보 계산
            short_open_sets = [ps for ps in self.position_manager.get_open_sets() if ps.side == PositionSide.SHORT]
            short_sets_count = len(short_open_sets)
            short_sets_total = sum(ps.total_amount for ps in short_open_sets)
            short_sets_info = f"{short_sets_count}({short_sets_total:.6f})" if short_sets_count > 0 else f"0({short_position.total_amount:.6f})"
            
            short_avg_price = short_position.weighted_avg_price
            short_pnl = short_position.calculate_unrealized_pnl(current_price)
            # 숏 가치 = 숏크기 + 숏 PnL
            short_value = short_size + short_pnl
            
            if len(self.detailed_records) < 10:
                self.logger.info(f"  계산된 값들 - 크기: {short_size}, Set: {short_sets_info}, 진입가: {short_avg_price}, PnL: {short_pnl}, 가치: {short_value}")
        
        # 전체 포지션 크기 계산
        position_btc = long_size + short_size
        selected_btc = position_btc
        
        # 상세 기록 저장
        record = {
            'timestamp': timestamp,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'sma_24': row.get('sma_24', ''),
            'sma_720': row.get('sma_720', ''),
            'trading_position': trading_position,
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'position_id': position_id,
            'trade_direction': trade_direction,
            'trade_price': trade_price,
            'trade_size': trade_size,
            'long_size': long_size,
            'long_sets': long_sets_info,
            'long_avg_price': long_avg_price,
            'long_value': long_value,
            'short_size': short_size,
            'short_sets': short_sets_info,
            'short_avg_price': short_avg_price,
            'short_value': short_value,
            'buy_signal_trade': buy_signal,
            'sell_signal_trade': sell_signal,
            'position_btc': position_btc,
            'selected_btc': selected_btc,
            'current_return': current_return,
            'cumulative_return': cumulative_return,
            'trade_balance': self.current_balance,
            'total_margin': self.position_manager.get_total_margin(),
            'total_equity': total_equity
        }
        
        self.detailed_records.append(record)

# =============================================================================
# 백테스트 리포트 생성 클래스
# =============================================================================

class BacktestRecorder:
    """백테스트 결과 기록 및 분석 클래스"""
    
    def __init__(self, initial_balance: float, output_dir: str = "logs", report_start_date: str = "2025-01-01 00:00:00"):
        self.output_dir = output_dir
        self.report_start_date = report_start_date
        self.initial_balance = initial_balance
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """출력 디렉토리 확인 및 생성"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def _create_ohlc_signals_sheet(self, original_data: pd.DataFrame, signals_data: pd.DataFrame, detailed_records: List[Dict] = None) -> pd.DataFrame:
        """OHLC + 지표 + 신호 + 상세 거래 정보 시트 생성 (첨부 이미지 형태)"""
        
        df = signals_data.copy()
        df_filtered = df[df.index >= self.report_start_date].copy()
        
        if len(df_filtered) == 0:
            # 빈 DataFrame 반환
            empty_columns = ['시간', '시가', '고가', '저가', '종가', 'SMA_24', 'SMA_720', '매수신호', '매도신호',
                           '거래ID', '거래방향', '거래가격', '거래크기', '롱크기', '롱Set', '롱진입가', '롱가치',
                           '숏크기', '숏Set', '숏진입가', '숏가치', '포지션크기', 
                           '포지션수익율(%)', '누적수익률(%)', '거래잔고(BTC)', '총마진(BTC)', '총자산(BTC)']
            return pd.DataFrame(columns=empty_columns)
        
        # 상세 기록 데이터프레임으로 변환
        if detailed_records:
            detailed_df = pd.DataFrame(detailed_records)
            detailed_df['timestamp'] = pd.to_datetime(detailed_df['timestamp'])
            detailed_df.set_index('timestamp', inplace=True)
            detailed_df = detailed_df[detailed_df.index >= self.report_start_date]
        else:
            detailed_df = pd.DataFrame()
        
        # 기본 OHLC 데이터 - 인덱스가 DatetimeIndex인지 확인하고 처리
        if hasattr(df_filtered.index, 'strftime'):
            time_formatted = df_filtered.index.strftime('%Y-%m-%d %H:%M:%S')
        else:
            time_formatted = [str(t) for t in df_filtered.index]
        
        result_df = pd.DataFrame({
            '시간': time_formatted,
            '시가': df_filtered['open'].round(2),
            '고가': df_filtered['high'].round(2),
            '저가': df_filtered['low'].round(2),
            '종가': df_filtered['close'].round(2),
        })
        
        # SMA 지표
        if 'sma_24' in df_filtered.columns:
            result_df['SMA_24'] = df_filtered['sma_24'].round(2)
        if 'sma_720' in df_filtered.columns:
            result_df['SMA_720'] = df_filtered['sma_720'].round(2)
        
        # 상세 기록이 있는 경우 추가 정보 포함
        if not detailed_df.empty:
            # 매수신호, 매도신호
            result_df['매수신호'] = detailed_df['buy_signal'].fillna('')
            result_df['매도신호'] = detailed_df['sell_signal'].fillna('')
            
            # 트랜잭션 정보
            result_df['거래ID'] = detailed_df['position_id'].fillna('')
            result_df['거래방향'] = detailed_df['trade_direction'].fillna('')
            
            # 거래가격 처리 (문자열과 숫자 구분)
            def format_trade_price(x):
                if not x or str(x) == '' or str(x) == 'nan':
                    return ''
                try:
                    # 숫자인 경우
                    if isinstance(x, (int, float)):
                        return f"{float(x):.2f}"
                    # 문자열인 경우 (여러 가격이 콤마로 구분된 경우)
                    elif isinstance(x, str) and ',' in str(x):
                        prices = str(x).split(',')
                        formatted_prices = []
                        for price in prices:
                            try:
                                formatted_prices.append(f"{float(price.strip()):.2f}")
                            except (ValueError, TypeError):
                                formatted_prices.append(price.strip())
                        return ','.join(formatted_prices)
                    else:
                        # 단일 문자열 숫자인 경우
                        return f"{float(x):.2f}"
                except (ValueError, TypeError):
                    return str(x)
            
            result_df['거래가격'] = detailed_df['trade_price'].apply(format_trade_price).fillna('')
            
            # 거래크기 처리 (실제 거래 발생시만 표시)
            def format_trade_size(x):
                if not x or str(x) == '' or str(x) == 'nan':
                    return ''
                try:
                    return safe_format_float(x, 6)
                except:
                    return ''
            
            result_df['거래크기'] = detailed_df['trade_size'].apply(format_trade_size).fillna('')
            
            # 안전한 수치 변환 함수
            def safe_format_float(x, decimals=6, default='0'):
                try:
                    if x is None or str(x) == 'nan' or str(x) == '':
                        return default + '.' + '0' * decimals
                    return f"{float(x):.{decimals}f}"
                except (ValueError, TypeError):
                    return default + '.' + '0' * decimals
            
            # 롱 포지션 정보
            result_df['롱크기'] = detailed_df['long_size'].apply(lambda x: safe_format_float(x, 6) if x and x > 0 else '0.000000').fillna('0.000000')
            result_df['롱Set'] = detailed_df['long_sets'].fillna('0')
            result_df['롱진입가'] = detailed_df['long_avg_price'].apply(lambda x: safe_format_float(x, 2) if x and x > 0 else '0.00').fillna('0.00')
            result_df['롱가치'] = detailed_df['long_value'].apply(lambda x: safe_format_float(x, 6) if x and x != 0 else '0.000000').fillna('0.000000')
            
            # 숏 포지션 정보
            result_df['숏크기'] = detailed_df['short_size'].apply(lambda x: safe_format_float(x, 6) if x and x > 0 else '0.000000').fillna('0.000000')
            result_df['숏Set'] = detailed_df['short_sets'].fillna('0')
            result_df['숏진입가'] = detailed_df['short_avg_price'].apply(lambda x: safe_format_float(x, 2) if x and x > 0 else '0.00').fillna('0.00')
            result_df['숏가치'] = detailed_df['short_value'].apply(lambda x: safe_format_float(x, 6) if x and x != 0 else '0.000000').fillna('0.000000')
            
            # 거래 신호 (중복이지만 이미지에 맞게)
            result_df['매수신호'] = detailed_df['buy_signal_trade'].fillna('')
            result_df['매도신호'] = detailed_df['sell_signal_trade'].fillna('')
            
            # 포지션 크기
            result_df['포지션크기'] = detailed_df['position_btc'].apply(lambda x: f"{x:.6f}" if x > 0 else '').fillna('')
            
            # 수익률
            result_df['포지션수익율(%)'] = detailed_df['current_return'].apply(lambda x: f"{x:.6f}" if x != 0 else '0.000000').fillna('0.000000')
            result_df['누적수익률(%)'] = detailed_df['cumulative_return'].apply(lambda x: f"{x:.6f}" if x != 0 else '0.000000').fillna('0.000000')
            
            # 계좌 정보
            result_df['거래잔고(BTC)'] = detailed_df['trade_balance'].apply(lambda x: f"{x:.6f}").fillna(f"{self.initial_balance:.6f}")
            result_df['총마진(BTC)'] = detailed_df['total_margin'].apply(lambda x: f"{x:.6f}" if x > 0 else '0.000000').fillna('0.000000')
            result_df['총자산(BTC)'] = detailed_df['total_equity'].apply(lambda x: f"{x:.6f}").fillna(f"{self.initial_balance:.6f}")
        else:
            # 상세 기록이 없는 경우 기본값으로 채우기
                
            if 'signal' in df_filtered.columns:
                # 매수신호
                buy_signal_map = {1: '롱진입', -1: '숏진입'}
                result_df['매수신호'] = df_filtered['signal'].map(buy_signal_map).fillna('')
                
                # 매도신호
                sell_signal_map = {2: '롱청산', -2: '숏청산'}
                result_df['매도신호'] = df_filtered['signal'].map(sell_signal_map).fillna('')
            else:
                result_df['매수신호'] = ''
                result_df['매도신호'] = ''
            
            # 빈 컬럼들 추가
            empty_cols = ['거래ID', '거래방향', '거래가격', '거래크기', '롱크기', '롱Set', '롱진입가', '롱가치',
                         '숏크기', '숏Set', '숏진입가', '숏가치', '포지션크기', 
                         '포지션수익율(%)', '누적수익률(%)', '거래잔고(BTC)', '총마진(BTC)', '총자산(BTC)']
            for col in empty_cols:
                if col in ['포지션수익율(%)', '누적수익률(%)', '총마진(BTC)', '롱크기', '숏크기', '롱가치', '숏가치']:
                    result_df[col] = '0.000000'
                elif col in ['롱진입가', '숏진입가']:
                    result_df[col] = '0.00'
                elif col in ['롱Set', '숏Set']:
                    result_df[col] = '0'
                elif col in ['거래잔고(BTC)', '총자산(BTC)']:
                    result_df[col] = f"{self.initial_balance:.6f}"
                else:
                    result_df[col] = ''
        
        return result_df
        
    def _save_to_excel(self, reports: Dict[str, pd.DataFrame], base_filename: str) -> str:
        """Excel 파일로 저장"""
        
        excel_path = os.path.join(self.output_dir, f"{base_filename}_backtest_report.xlsx")
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                sheet_names = {'ohlc_signals': '시세분석'}
                
                for key, df in reports.items():
                    sheet_name = sheet_names.get(key, key)
                    if not df.empty:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
            print(f"📊 백테스트 리포트가 저장되었습니다: {excel_path}")
            return excel_path
            
        except Exception as e:
            print(f"❌ Excel 저장 실패: {e}")
            csv_path = os.path.join(self.output_dir, f"{base_filename}_backtest_report.csv")
            reports['ohlc_signals'].to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"📄 리포트가 CSV로 저장되었습니다: {csv_path}")
            return csv_path

# =============================================================================
# 메인 실행 함수들
# =============================================================================

def fetch_okx_data(exchange: OKXExchange, symbol: str, timeframe: str, limit: int, start_date=None) -> pd.DataFrame:
    """OKX에서 OHLCV 데이터 가져오기"""
    
    print(f"📊 OKX 데이터 수집: {symbol} {timeframe} {limit}봉")
    if start_date:
        print(f"   시작일: {start_date}")
    
    try:
        all_data = []
        max_per_request = 300
        
        if start_date:
            current_since = int(pd.to_datetime(start_date).timestamp() * 1000)
            
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
                    if not any(existing[0] == timestamp for existing in all_data):
                        new_data.append(candle)
                
                all_data.extend(new_data)
                
                if ohlcv:
                    latest_timestamp = max(candle[0] for candle in ohlcv)
                    current_since = latest_timestamp + (60 * 60 * 1000)
                
                time.sleep(0.2)
                
        else:
            current_time = None
            
            while len(all_data) < limit:
                remaining = min(max_per_request, limit - len(all_data))
                print(f"  요청: {remaining}개 (총 {len(all_data)}/{limit})")
                
                params = {'limit': remaining}
                if current_time:
                    params['since'] = int(current_time.timestamp() * 1000)
                
                ohlcv = exchange.exchange.fetch_ohlcv(symbol, timeframe, **params)
                
                if not ohlcv:
                    break
                    
                new_data = []
                for candle in ohlcv:
                    timestamp = candle[0]
                    if not any(existing[0] == timestamp for existing in all_data):
                        new_data.append(candle)
                
                all_data.extend(new_data)
                
                if ohlcv:
                    oldest_timestamp = min(candle[0] for candle in ohlcv)
                    current_time = pd.to_datetime(oldest_timestamp, unit='ms') - pd.Timedelta(hours=1)
                
                time.sleep(0.1)
        
        # 시간순 정렬
        all_data.sort(key=lambda x: x[0])
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        df = df.astype(float)
        
        print(f"✅ 데이터 수집 완료: {len(df)}봉")
        print(f"   기간: {df.index[0]} ~ {df.index[-1]}")
        
        return df
        
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        raise

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
        'margin_per_trade': position_size / leverage,  # 포지션 크기를 레버리지로 나누어 마진 계산
        'position_size': position_size,
        'take_profit_pct': take_profit_pct,
        'stop_loss_pct': stop_loss_pct,
        'stop_loss_enabled': True,
        'max_concurrent_positions': 5,
        'daily_loss_limit_pct': 0.05
    }

def run_backtest(data: pd.DataFrame, strategy_config: dict, 
                 initial_balance: float, report_start_date: str, backtest_config: dict, config_manager) -> None:
    """백테스트 실행"""
    
    print(f"\n🔄 백테스트 실행")
    print(f"   데이터: {len(data)}봉")
    print(f"   초기자본: {initial_balance:.6f} BTC")
    
    if len(data) < 720:
        print(f"❌ 데이터 부족: {len(data)}봉 (최소 720봉 필요)")
        return
    
    # 전략 및 백테스터 초기화
    strategy = SMACrossoverStrategy(strategy_config, config_manager)
    backtester = Backtester(initial_balance=initial_balance)
    
    # 백테스트 실행
    result = backtester.run_backtest(
        strategy=strategy,
        data=data,
        commission=backtest_config['fees']['commission'],
        funding_rate=backtest_config['fees']['funding_rate']
    )
    
    print(f"✅ 백테스트 완료!")
    print(f"   총 거래수: {result.total_trades}")
    print(f"   승률: {result.win_rate:.1%}")
    print(f"   총 수익률: {result.total_return:.2%}")
    print(f"   최대 낙폭: {result.max_drawdown:.2%}")
    print(f"   샤프 비율: {result.sharpe_ratio:.2f}")
    
    # 리포트 생성 (상세 정보 포함)
    print(f"\n📊 리포트 생성 중...")
    
    recorder = BacktestRecorder(
        initial_balance=backtest_config['initial_balance_btc'],
        output_dir=f"./{backtest_config['report']['output_dir']}",
        report_start_date=report_start_date
    )
    
    signals_data = strategy.generate_signals(data.copy())
    detailed_records = getattr(result, 'detailed_records', [])  # 기본값으로 빈 리스트 사용
    ohlc_signals = recorder._create_ohlc_signals_sheet(data, signals_data, detailed_records)
    
    reports = {'ohlc_signals': ohlc_signals}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = recorder._save_to_excel(reports, f"SMACrossoverStrategy_{timestamp}")
    
    print(f"✅ 리포트 저장: {excel_path}")

def calculate_data_requirements(backtest_config: dict, timeframe: str) -> dict:
    """백테스트에 필요한 데이터량 계산"""
    
    trading_period = backtest_config['trading_period']
    sma_period = 720
    
    if trading_period.get('start_date') and trading_period.get('end_date'):
        start_date = pd.to_datetime(trading_period['start_date'])
        end_date = pd.to_datetime(trading_period['end_date'])
        
        if timeframe == '1h':
            hours_diff = int((end_date - start_date).total_seconds() / 3600)
        else:
            raise ValueError(f"지원하지 않는 timeframe: {timeframe}")
        
        trading_hours = hours_diff
        report_start_date = trading_period['start_date']
        data_start_date = start_date - pd.Timedelta(hours=sma_period)
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
    parser = argparse.ArgumentParser(description='올인원 백테스트 스크립트')
    
    parser.add_argument('--symbol', default=None, help='거래 심볼')
    parser.add_argument('--timeframe', default='1h', help='시간봉')
    parser.add_argument('--hours', type=int, default=None, help='거래할 시간봉 개수')
    parser.add_argument('--initial-balance', type=float, default=None, help='초기 자본 BTC')
    parser.add_argument('--report-start', default=None, help='리포트 시작 날짜')
    parser.add_argument('--sandbox', action='store_true', default=True, help='샌드박스 모드')
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("🚀 올인원 백테스트 스크립트")
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
        
        print(f"📋 백테스트 설정:")
        print(f"  - 심볼: {symbol}")
        print(f"  - 거래기간: {data_req['trading_hours']}시간")
        print(f"  - 초기자본: {initial_balance:.6f} BTC")
        print(f"  - 리포트 시작: {report_start_date}")
        
        # OKX 연결
        print("\n🔗 OKX 연결 중...")
        exchange = OKXExchange(config_manager=config_manager, sandbox=args.sandbox)
        
        if not exchange.test_connection():
            print("❌ OKX 연결 실패")
            print("API 키 환경변수를 확인하세요:")
            print("  - OKX_API_KEY")
            print("  - OKX_SECRET_KEY") 
            print("  - OKX_PASSPHRASE")
            print("  - OKX_SANDBOX")
            return
        
        print("✅ OKX 연결 성공")
        
        print(f"\n📊 필요 데이터:")
        print(f"  - 거래 기간: {data_req['trading_hours']}봉")
        print(f"  - SMA 계산용: {data_req['sma_period']}봉")
        print(f"  - 총 필요량: {data_req['total_data_needed']}봉")
        
        # 데이터 수집
        data = fetch_okx_data(exchange, symbol, args.timeframe, data_req['total_data_needed'], data_req['data_start_date'])
        
        # 백테스트 실행
        strategy_config = create_strategy_config(symbol, config_manager)
        run_backtest(data, strategy_config, initial_balance, report_start_date, backtest_config, config_manager)
        
        print("\n🎉 완료!")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 