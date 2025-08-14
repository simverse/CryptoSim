import logging
from typing import Dict
import pandas as pd
import numpy as np

from src.config_manager import ConfigManager
from src.models import PositionSide


class BaseStrategy:
    """전략 기본 클래스"""
    
    def __init__(self, name: str, config: Dict, config_manager: ConfigManager = None):
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

class SMACrossoverStrategy(BaseStrategy):
    """SMA 크로스오버 전략 클래스"""
    
    def __init__(self, config: Dict, config_manager: ConfigManager = None):
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
        
        # Transaction/Position 관리자는 더 이상 strategy에서 직접 관리하지 않음
        
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
        df['golden_cross_signal'] = (df[f'sma_{self.short_sma_period}'] > df[f'sma_{self.long_sma_period}']) &                                   (df[f'sma_{self.short_sma_period}'].shift(1) <= df[f'sma_{self.long_sma_period}'].shift(1))
        df['dead_cross_signal'] = (df[f'sma_{self.short_sma_period}'] < df[f'sma_{self.long_sma_period}']) &                                 (df[f'sma_{self.short_sma_period}'].shift(1) >= df[f'sma_{self.long_sma_period}'].shift(1))
        
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
        df['signal'] = 0  # 0: 신호없음, 1: 롱, -1: 숏
        df['signal_type'] = 'none'

        # 롱 진입 조건: 상승 추세(정배열) + 장기 이평선 상승 + 가격 하락
        long_entry_condition = (
            df['golden_cross'] &
            df['long_sma_trend'] &
            df['price_decline']
        )

        # 숏 진입 조건: 하락 추세(역배열) + 장기 이평선 하락 + 가격 상승
        short_entry_condition = (
            df['dead_cross'] &
            (~df['long_sma_trend']) &
            df['price_increase']
        )

        df['signal'] = np.where(long_entry_condition, 1, 0)
        df['signal'] = np.where(short_entry_condition, -1, df['signal'])

        df['signal_type'] = np.where(long_entry_condition, 'long_entry', 'none')
        df['signal_type'] = np.where(short_entry_condition, 'short_entry', df['signal_type'])

        return df
    
    def calculate_position_size(self, current_balance: float, current_price: float) -> Dict:
        """포지션 크기 계산"""
        return {
            'amount': self.position_size,
            'margin': self.margin_per_trade,
            'leverage': self.leverage
        }
