import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

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
            'sandbox': self._get_env_bool('OKX_SANDBOX', self.get('exchange.sandbox', True)),
            'trade_mode': self.get('exchange.trade_mode', 'demo') # trade_mode 추가
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
            'use_tp': self.get('trading.use_tp', True),
            'use_sl': self.get('trading.use_sl', True),
            'stop_loss_pct': self.get('trading.stop_loss_pct', 0.02),
            'take_profit_pct': self.get('trading.take_profit_pct', 0.04)
        }
    
    def get_backtest_config(self) -> Dict[str, Any]:
        """백테스트 설정 조회"""
        initial_balance = self.get('backtest.initial_balance')
        if initial_balance is None:
            raise ValueError("config.yaml에 'backtest.initial_balance' 설정이 없습니다.")
        
        return {
            'initial_balance': initial_balance,
            'trading_period': self.get('backtest.trading_period', {'hours': 2000}),
            'fees': self.get('backtest.fees', {'commission': 0.0001, 'funding_rate': 0.0001}),
            'report': self.get('backtest.report', {'output_dir': 'logs'})
        }

    def get_recorder_config(self) -> Dict[str, Any]:
        """리코더 설정 조회"""
        return self.get('backtest.report.recorder', {'excel': {'enabled': True}, 'google_sheet': {'enabled': False}})
