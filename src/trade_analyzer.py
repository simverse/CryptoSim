import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models import Position, PositionSet, PositionSide, Transaction
from src.strategy import BaseStrategy
from src.trading_manager import TradingManager # PositionManager 대신 TradingManager 임포트
from src.virtual_exchange import VirtualExchange # 가상 거래소 임포트
# from src.transaction_manager import TransactionManager # 더 이상 필요 없음


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


class TradeAnalyzer:
    """거래 분석 엔진"""
    
    def __init__(self, config, initial_balance: float = 1.0):
        """TradeAnalyzer 초기화"""
        self.config = config # config 저장
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # TradingManager 초기화 (백테스트에서는 실제 exchange가 필요 없으므로 VirtualExchange 전달)
        self.trading_manager = TradingManager(config, VirtualExchange(config, initial_balance)) 
        
        # 기록용 데이터
        self.trades = []
        self.equity_curve = []
        self.detailed_records = []
        self.current_tp_signal = None # 익절 신호 임시 저장
        self.last_closed_trades = []
        
        # 통계
        self.total_commission = 0.0
        self.total_funding_fee = 0.0
        
        # 거래 제한
        self.max_concurrent_positions = 5
        self.daily_loss_limit_pct = 0.05
        
        # 수수료 및 펀딩비 설정 (TradingManager에서 관리하도록 변경 예정)
        self.commission = 0.0001
        self.funding_rate = 0.0001
        
        # 로거
        self.logger = logging.getLogger("TradeAnalyzer") # 로거 이름 변경

    def run_backtest(self, strategy: BaseStrategy, all_data: pd.DataFrame, backtest_data: pd.DataFrame,
                     commission: float = 0.0001, funding_rate: float = 0.0001) -> BacktestResult:
        """백테스트 실행"""
        
        self._reset()
        self.commission = commission
        self.funding_rate = funding_rate
        
        # 신호 생성은 전체 데이터를 기반으로 수행
        signals_data = strategy.generate_signals(all_data.copy())
        
        # 실제 백테스트 루프는 backtest_data(필터링된 데이터)를 기반으로 수행
        for i, (timestamp, row) in enumerate(backtest_data.iterrows()):
            # 현재 캔들의 신호 정보를 signals_data에서 가져옴
            current_signal_row = signals_data.loc[timestamp]

            self.logger.info(f"Processing candle at: {timestamp}")
            self.last_closed_trades.clear()
            self.current_tp_signal = None

            # TradingManager를 통해 신호 처리
            self.trading_manager.process_signal(current_signal_row)
            self.trading_manager.check_open_orders(row)
            
            if i % 8 == 0:
                pass

            self._update_equity_curve(row)
            
            # 상세 기록 시에도 현재 캔들의 신호 정보를 전달
            self._record_detailed_data(timestamp, current_signal_row, strategy)
        
        result = self._calculate_results(strategy.name, backtest_data)
        result.signals_data = signals_data[signals_data.index >= backtest_data.index[0]] # 리포트용 신호 데이터도 필터링
        result.detailed_records = self.detailed_records
        
        return result

    def _reset(self):
        """TradeAnalyzer 초기화"""
        self.current_balance = self.initial_balance
        self.total_commission = 0.0
        self.total_funding_fee = 0.0
        
        # TradingManager 초기화
        self.trading_manager = TradingManager(self.config, VirtualExchange(self.config, self.initial_balance)) # config 전달
        self.trades.clear()
        self.equity_curve.clear()
        self.detailed_records.clear()
        self.current_tp_signal = None
        self.last_closed_trades = []
        
    def _process_signal(self, row, strategy: BaseStrategy):
        """신호 처리 (TradingManager로 위임) """
        # 이 메서드는 더 이상 TradeAnalyzer에서 직접 호출되지 않고,
        # run_backtest 내에서 self.trading_manager.process_signal(row)로 대체됨
        pass
            
    def _open_position(self, side: str, price: float, timestamp, strategy: BaseStrategy):
        """포지션 진입 (TradingManager로 위임) """
        pass
    
    def _check_and_create_position_set(self, side: PositionSide, strategy: BaseStrategy):
        """PositionSet 생성 조건 확인 및 생성 (TradingManager로 위임) """
        pass

    def _close_position(self, transaction_id: str, price: float, timestamp, reason: str):
        """개별 거래 청산 (TradingManager로 위임) """
        pass
    
    def _close_long_positions(self, price: float, timestamp, reason: str):
        """모든 롱 포지션 청산 (TradingManager로 위임) """
        pass
    
    def _close_short_positions(self, price: float, timestamp, reason: str):
        """모든 숏 포지션 청산 (TradingManager로 위임) """
        pass
    
    def _close_all_positions(self, price: float):
        """모든 포지션 청산 (TradingManager로 위임) """
        pass
    
    def _check_take_profit(self, high_price: float, low_price: float, timestamp, strategy: BaseStrategy):
        """익절 확인 (TradingManager로 위임) """
        pass
    
    def _check_position_set_take_profit(self, high_price: float, low_price: float, timestamp, strategy: BaseStrategy):
        """PositionSet 익절 확인 (TradingManager로 위임) """
        pass

    def _check_position_take_profit(self, high_price: float, low_price: float, timestamp, strategy: BaseStrategy):
        """Position 전체를 기준으로 익절 확인 (TradingManager로 위임) """
        pass

    def _check_set_exit_conditions(self, position_set: PositionSet, high_price: float, low_price: float, strategy: BaseStrategy) -> Tuple[bool, float]:
        """PositionSet 청산 조건 확인 및 생성 (TradingManager로 위임) """
        pass
    
    def _apply_funding_fee(self, row):
        """펀딩피 적용 (TradingManager로 위임) """
        pass

    def _update_equity_curve(self, row):
        """자산 곡선 업데이트"""
        price = row['close']
        # TradingManager에서 현재 열려있는 트랜잭션 가져오기
        positions, open_orders, transaction_manager, _, _ = self.trading_manager.get_current_state()
        open_transactions = [tx for tx in transaction_manager.values() if not tx.is_closed]

        unrealized_pnl = sum(t.calculate_unrealized_pnl(price) for t in open_transactions)
        
        # 현재 잔고는 TradingManager의 상태를 기반으로 계산
        # TradingManager의 positions에서 total_margin을 가져와야 함
        total_margin = positions[PositionSide.LONG].total_margin + positions[PositionSide.SHORT].total_margin
        
        # current_balance는 TradingManager의 PnL을 반영해야 함
        # TradingManager에서 실현 손익을 가져와 current_balance에 반영
        realized_pnl_sum = sum(tx.realized_pnl for tx in transaction_manager.values() if tx.is_closed and tx.realized_pnl is not None)
        
        # 백테스트 초기 잔고 + 실현 손익 - 현재 마진 + 미실현 손익
        equity = self.trading_manager.exchange.balance + unrealized_pnl
        
        self.equity_curve.append({'timestamp': row.name, 'equity': equity})
        
    def _calculate_results(self, strategy_name: str, data: pd.DataFrame) -> BacktestResult:
        """백테스트 결과 계산"""
        equity_df = pd.DataFrame(self.equity_curve).set_index('timestamp')
        total_return = (equity_df['equity'].iloc[-1] - self.initial_balance) / self.initial_balance
        
        # TradingManager에서 최종 거래 내역 가져오기
        positions, open_orders, transaction_manager, long_sets, short_sets = self.trading_manager.get_current_state()
        
        # 모든 트랜잭션을 trades 리스트에 추가 (실현 손익이 있는 것만)
        self.trades.clear() # 기존 trades 초기화
        for tx in transaction_manager.values():
            if tx.is_closed and tx.realized_pnl is not None:
                self.trades.append({
                    'position_id': tx.id, 'side': tx.side.value,
                    'amount': tx.amount, 'entry_price': tx.entry_price,
                    'exit_price': tx.exit_price, 'entry_time': tx.entry_time,
                    'exit_time': tx.exit_time, 'margin': tx.margin, 'fees': tx.fees,
                    'net_pnl': tx.realized_pnl, 'exit_reason': tx.exit_reason
                })

        if not self.trades:
            return BacktestResult(strategy_name, data.index[0], data.index[-1], self.initial_balance, self.current_balance, 0,0,0,0,0,0,0,0,0,0,[],equity_df,data,[])

        trades_df = pd.DataFrame(self.trades)
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        losing_trades = trades_df[trades_df['net_pnl'] <= 0]
        
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=data.index[0], end_date=data.index[-1],
            initial_balance=self.initial_balance, final_balance=self.current_balance,
            total_return=total_return,
            annual_return=(1 + total_return) ** (365 / (data.index[-1] - data.index[0]).days) - 1 if (data.index[-1] - data.index[0]).days > 0 else 0,
            max_drawdown=(equity_df['equity'] / equity_df['equity'].cummax() - 1).min(),
            sharpe_ratio=(equity_df['equity'].pct_change().mean() / equity_df['equity'].pct_change().std()) * np.sqrt(252*24) if equity_df['equity'].pct_change().std() > 0 else 0,
            total_trades=len(trades_df),
            winning_trades=len(winning_trades), losing_trades=len(losing_trades),
            win_rate=len(trades_df) / len(trades_df) if len(trades_df) > 0 else 0,
            avg_win=winning_trades['net_pnl'].mean() if not winning_trades.empty else 0,
            avg_loss=losing_trades['net_pnl'].mean() if not losing_trades.empty else 0,
            profit_factor=winning_trades['net_pnl'].sum() / abs(losing_trades['net_pnl'].sum()) if not losing_trades.empty and losing_trades['net_pnl'].sum() != 0 else float('inf'),
            trades=self.trades, equity_curve=equity_df, signals_data=data, detailed_records=self.detailed_records
        )

    def _record_detailed_data(self, timestamp, row, strategy: BaseStrategy):
        """시점별 상세 데이터 기록"""
        record = {'timestamp': timestamp}

        # TradingManager에서 현재 상태 가져오기 (맨 위로 이동)
        positions, open_orders, transaction_manager, long_position_sets, short_position_sets = self.trading_manager.get_current_state()

        candle_datetime = timestamp.to_pydatetime()

        # 전략 신호
        signal = row.get('signal', 0)
        record['buy_signal'] = ''
        record['sell_signal'] = ''
        if signal == 1: record['buy_signal'] = '롱진입'
        elif signal == -1: record['buy_signal'] = '숏진입'
                
        # 익절 신호 또는 전략 청산 신호
        # 현재 캔들에서 청산된 거래를 확인하여 매도 신호 기록
        closed_transactions_in_candle = [
            tx for tx in transaction_manager.values()
            if tx.is_closed and tx.exit_time and tx.exit_time.replace(tzinfo=None) == candle_datetime.replace(tzinfo=None)
        ]

        if closed_transactions_in_candle:
            for tx in closed_transactions_in_candle:
                if tx.side == PositionSide.LONG:
                    record['sell_signal'] = '롱청산'
                elif tx.side == PositionSide.SHORT:
                    record['sell_signal'] = '숏청산'
                break
        elif signal == 2: record['sell_signal'] = '롱청산'
        elif signal == -2: record['sell_signal'] = '숏청산'

        # 거래 정보 (현재 캔들에서 발생한 신규 거래 또는 청산)
        candle_datetime = timestamp.to_pydatetime()

        # TradingManager에서 전체 청산 주문 정보를 가져옴
        full_exit_order_info = self.trading_manager.last_full_exit_order_info
        
        if full_exit_order_info and full_exit_order_info['side'] == position_side.value and full_exit_order_info['id'] == filled_order['id']:
            # 전체 포지션 청산인 경우
            record['position_id'] = full_exit_order_info['id']
            record['trade_direction'] = 'close_' + full_exit_order_info['side']
            record['trade_price'] = filled_order['average']
            record['trade_size'] = full_exit_order_info['amount']
            record['realized_pnl'] = full_exit_order_info['realized_pnl']
            self.trading_manager.last_full_exit_order_info = None # 사용 후 초기화
        else:
            # 개별 거래 또는 진입 주문인 경우
            current_candle_transactions = [
                tx for tx in transaction_manager.values()
                if (tx.entry_time == candle_datetime) or \
                   (tx.is_closed and tx.exit_time and tx.exit_time == candle_datetime)
            ]

            if current_candle_transactions:
                latest_tx = max(current_candle_transactions, key=lambda tx: tx.entry_time if tx.entry_time == timestamp.to_pydatetime() else tx.exit_time)
                record['position_id'] = latest_tx.id
                record['trade_direction'] = latest_tx.side.name.lower() if latest_tx.entry_time == timestamp else 'close_' + latest_tx.side.name.lower()
                record['trade_price'] = latest_tx.entry_price if latest_tx.entry_time == timestamp else latest_tx.exit_price
                record['trade_size'] = sum(tx.amount for tx in current_candle_transactions)
                total_realized_pnl = sum(tx.realized_pnl if tx.realized_pnl is not None else 0 for tx in current_candle_transactions)
                record['realized_pnl'] = total_realized_pnl if total_realized_pnl != 0 else None
            else:
                record['position_id'] = None
                record['trade_direction'] = None
                record['trade_price'] = None
                record['trade_size'] = None
                record['realized_pnl'] = None
        # 포지션 정보 (전체 기준)
        total_long_size = 0.0
        total_short_size = 0.0

        for side in [PositionSide.LONG, PositionSide.SHORT]:
            prefix = side.name.lower()
            
            # Position과 PositionSet의 정보를 모두 합산하여 계산
            current_position = positions[side]
            sets_for_side = long_position_sets if side == PositionSide.LONG else short_position_sets

            # 1. 총 포지션 크기 (Position + 모든 PositionSets)
            total_size = current_position.total_amount + sum(p_set.total_amount for p_set in sets_for_side)

            # 2. 총 포지션 가치 (Position + 모든 PositionSets)
            total_value = (current_position.total_amount * current_position.weighted_avg_price) + \
                          sum(p_set.total_amount * p_set.avg_entry_price for p_set in sets_for_side)

            # 3. 총 평균 진입가
            total_avg_price = total_value / total_size if total_size > 0 else 0

            record[f'total_{prefix}_size'] = total_size
            record[f'total_{prefix}_avg_price'] = total_avg_price
            if side == PositionSide.LONG:
                # 롱 포지션 가치 (미실현 손익)
                record[f'total_{prefix}_value'] = total_size * (row['close'] - total_avg_price) / total_avg_price if total_avg_price > 0 else 0
                total_long_size = total_size
            else: # PositionSide.SHORT
                # 숏 포지션 가치 (미실현 손익)
                record[f'total_{prefix}_value'] = total_size * (total_avg_price - row['close']) / total_avg_price if total_avg_price > 0 else 0
                total_short_size = total_size

            # PositionSet 정보 기록 (기존 로직 유지)
            if not sets_for_side:
                record[f'{prefix}_sets'] = '0()'
            else:
                num_sets = len(sets_for_side)
                avg_of_set_prices = sum(p_set.avg_entry_price for p_set in sets_for_side) / num_sets
                record[f'{prefix}_sets'] = f"{num_sets}({avg_of_set_prices:.6f})"

        # 계좌 정보
        # TradingManager의 상태를 기반으로 일관성 있게 계산
        current_balance = self.trading_manager.exchange.balance
        total_margin = positions[PositionSide.LONG].total_margin + positions[PositionSide.SHORT].total_margin
        # Position 객체를 통한 미실현 손익 계산
        total_unrealized_pnl_from_positions = positions[PositionSide.LONG].calculate_unrealized_pnl(row['close']) + \
                                              positions[PositionSide.SHORT].calculate_unrealized_pnl(row['close'])

        # PositionSet의 미실현 손익 추가
        for p_set in long_position_sets:
            total_unrealized_pnl_from_positions += p_set.calculate_unrealized_pnl(row['close'])
        for p_set in short_position_sets:
            total_unrealized_pnl_from_positions += p_set.calculate_unrealized_pnl(row['close'])

        # 개별 트랜잭션을 통한 미실현 손익 계산 (비교용)
        open_transactions = [tx for tx in transaction_manager.values() if not tx.is_closed]
        total_unrealized_pnl_from_transactions = sum(t.calculate_unrealized_pnl(row['close']) for t in open_transactions)

        # 두 값의 차이 기록 (디버깅 및 검증용)
        unrealized_pnl_diff = total_unrealized_pnl_from_positions - total_unrealized_pnl_from_transactions

        #delme
        if unrealized_pnl_diff != 0:
            print(f"unrealized_pnl_diff: {unrealized_pnl_diff}")

        record['unrealized_pnl_diff'] = unrealized_pnl_diff
        
        # 최종적으로 total_equity 계산에 사용할 값은 Position 객체를 통한 값
        total_unrealized_pnl = total_unrealized_pnl_from_positions
        
        # 총 투입 포지션 크기 합 계산
        total_position_size_sum = total_long_size + total_short_size

        # 총자산 = 현재 잔고 + 미실현 손익
        # (현재 잔고는 거래 수수료 등이 이미 반영된 상태)
        total_equity = current_balance + total_unrealized_pnl
        
        # 거래잔고 = 현재 잔고 - 총 투입 마진
        trade_balance = current_balance - total_margin

        # 포지션 수익률 (%) = 미실현손익 / 총 투입 포지션 크기 합
        if total_position_size_sum > 0:
            record['current_return'] = (total_unrealized_pnl / total_position_size_sum) * 100
        else:
            record['current_return'] = 0.0
        
        # 누적 수익률 (%) = (현재 총자산 - 초기자본) / 초기자본
        record['cumulative_return'] = (total_equity / self.initial_balance - 1) * 100 if self.initial_balance != 0 else 0.0
        
        record['trade_balance'] = trade_balance
        record['total_margin'] = total_margin
        record['total_equity'] = total_equity

        self.detailed_records.append(record)
