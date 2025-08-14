# CryptoSim v2.0 - 개발 룰 및 가이드라인

## 🎯 개발 원칙

### 1. 핵심 원칙
- **단순성 추구**: 복잡한 구조보다는 명확하고 이해하기 쉬운 코드
- **전문가 수준**: 30년 경력 개발자에게 적합한 고급 기능과 최적화
- **한국어 우선**: 모든 주석, 로그, 문서는 한국어로 작성
- **단일 파일**: 모든 핵심 기능을 하나의 파일에 통합

### 2. 코드 품질 기준
- **가독성**: 변수명, 함수명, 클래스명은 목적이 명확하게 드러나야 함
- **일관성**: 동일한 패턴과 스타일을 프로젝트 전체에 적용
- **효율성**: 메모리와 CPU 사용량을 최적화
- **안정성**: 모든 예외 상황에 대한 적절한 처리

## 📝 코딩 스타일 가이드

### 1. 파일 구조
```python
#!/usr/bin/env python3
"""
CryptoSim v2.0 - 올인원 백테스팅 시스템
모든 기능을 하나의 파일에 통합한 간편한 암호화폐 백테스팅 솔루션

사용법:
    python cryptosim_v2.py --symbol BTC/USDT:USDT --timeframe 1h --hours 2000

작성자: CryptoSim 개발팀
버전: 2.0
날짜: 2025-01-17
"""

# 표준 라이브러리 임포트
import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# 서드파티 라이브러리 임포트
import pandas as pd
import numpy as np
import yaml
import logging
import ccxt

# 설정 및 상수
DEFAULT_CONFIG = {...}
SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d']
```

### 2. 네이밍 컨벤션

#### 변수명
```python
# 좋은 예 - 목적이 명확한 한국어 주석
initial_balance = 1.0  # 초기 자본 (BTC)
trading_symbol = "BTC/USDT:USDT"  # 거래 심볼
sma_short_period = 24  # 단기 이동평균 기간
current_position_size = 0.0  # 현재 포지션 크기

# 나쁜 예
x = 1.0
data = "BTC/USDT:USDT"
p1 = 24
pos = 0.0
```

#### 함수명
```python
# 좋은 예 - 동작이 명확한 동사형
def fetch_ohlcv_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """OHLCV 데이터를 수집합니다."""
    pass

def calculate_sma_signals(df: pd.DataFrame) -> pd.DataFrame:
    """SMA 크로스오버 신호를 계산합니다."""
    pass

def generate_excel_report(backtest_result: dict) -> str:
    """Excel 형태의 백테스트 결과 리포트를 생성합니다."""
    pass
```

#### 클래스명
```python
# 좋은 예 - 목적이 명확한 명사형
class ConfigManager:
    """설정 파일 관리 클래스"""
    pass

class DataFetcher:
    """데이터 수집 클래스"""
    pass

class BacktestEngine:
    """백테스팅 엔진 클래스"""
    pass
```

### 3. 주석 작성 규칙

#### 클래스 주석
```python
class BacktestEngine:
    """
    백테스팅 엔진 클래스
    
    암호화폐 거래 전략의 백테스팅을 수행하고 성과를 분석합니다.
    SMA 크로스오버 전략을 기본으로 지원하며, 확장 가능한 구조를 제공합니다.
    
    Attributes:
        initial_balance (float): 초기 자본 (BTC 기준)
        commission_rate (float): 거래 수수료율
        funding_rate (float): 펀딩피율
        leverage (int): 레버리지 배수
    
    Example:
        >>> engine = BacktestEngine(initial_balance=1.0)
        >>> result = engine.run_backtest(data, strategy_config)
    """
```

#### 함수 주석
```python
def calculate_position_size(current_balance: float, price: float, risk_ratio: float = 0.02) -> float:
    """
    포지션 크기를 계산합니다.
    
    Args:
        current_balance (float): 현재 잔고 (BTC)
        price (float): 현재 가격 (USDT)
        risk_ratio (float): 리스크 비율 (기본값: 0.02)
    
    Returns:
        float: 계산된 포지션 크기 (BTC)
    
    Raises:
        ValueError: 잔고가 음수이거나 가격이 0 이하인 경우
    
    Example:
        >>> position_size = calculate_position_size(1.0, 50000.0)
        >>> print(f"포지션 크기: {position_size:.6f} BTC")
    """
```

#### 인라인 주석
```python
# 단기/장기 이동평균 계산
df['sma_short'] = df['close'].rolling(window=short_period).mean()
df['sma_long'] = df['close'].rolling(window=long_period).mean()

# 골든 크로스/데드 크로스 감지
df['golden_cross'] = (df['sma_short'] > df['sma_long']) & (df['sma_short'].shift(1) <= df['sma_long'].shift(1))
df['dead_cross'] = (df['sma_short'] < df['sma_long']) & (df['sma_short'].shift(1) >= df['sma_long'].shift(1))

# 거래 신호 생성 (1: 매수, -1: 매도, 0: 보유)
df['signal'] = np.where(df['golden_cross'], 1, np.where(df['dead_cross'], -1, 0))
```

## 🔧 기술적 가이드라인

### 1. 에러 처리

#### 네트워크 에러
```python
def fetch_data_with_retry(self, symbol: str, timeframe: str, limit: int, max_retries: int = 3) -> pd.DataFrame:
    """재시도 로직을 포함한 데이터 수집"""
    for attempt in range(max_retries):
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except ccxt.NetworkError as e:
            if attempt == max_retries - 1:
                self.logger.error(f"네트워크 오류로 데이터 수집 실패: {e}")
                raise
            self.logger.warning(f"네트워크 오류 발생 (재시도 {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)  # 지수 백오프
```

#### 데이터 검증
```python
def validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 데이터 유효성 검증"""
    if df.empty:
        raise ValueError("데이터가 비어있습니다.")
    
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
    
    # 가격 데이터 이상치 검사
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if (df[col] <= 0).any():
            self.logger.warning(f"{col} 컬럼에 0 이하의 값이 있습니다.")
            df = df[df[col] > 0]  # 이상치 제거
    
    return df
```

### 2. 성능 최적화

#### 메모리 효율적인 데이터 처리
```python
def process_large_dataset(self, df: pd.DataFrame, chunk_size: int = 1000) -> pd.DataFrame:
    """대용량 데이터를 청크 단위로 처리"""
    results = []
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        processed_chunk = self.process_chunk(chunk)
        results.append(processed_chunk)
        
        # 메모리 사용량 모니터링
        if i % (chunk_size * 10) == 0:
            self.logger.info(f"처리 진행률: {(i / len(df)) * 100:.1f}%")
    
    return pd.concat(results, ignore_index=True)
```

#### 벡터화 연산 활용
```python
def calculate_signals_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
    """벡터화된 연산을 사용한 신호 계산"""
    # 반복문 대신 벡터화 연산 사용
    df['returns'] = df['close'].pct_change()
    df['sma_short'] = df['close'].rolling(window=self.short_period).mean()
    df['sma_long'] = df['close'].rolling(window=self.long_period).mean()
    
    # 조건부 로직도 벡터화
    conditions = [
        (df['sma_short'] > df['sma_long']) & (df['sma_short'].shift(1) <= df['sma_long'].shift(1)),
        (df['sma_short'] < df['sma_long']) & (df['sma_short'].shift(1) >= df['sma_long'].shift(1))
    ]
    choices = [1, -1]
    df['signal'] = np.select(conditions, choices, default=0)
    
    return df
```

### 3. 로깅 시스템

#### 로거 설정
```python
def setup_logger(self, log_level: str = "INFO", log_file: str = "logs/cryptosim.log") -> logging.Logger:
    """로깅 시스템 설정"""
    # 로그 디렉토리 생성
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('cryptosim')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 포매터 설정 (한국어 로그)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
```

#### 로그 메시지 작성
```python
# 정보성 로그
self.logger.info(f"백테스트 시작: {symbol} / {timeframe} / {start_date} ~ {end_date}")
self.logger.info(f"초기 자본: {initial_balance:.6f} BTC")

# 경고 로그
self.logger.warning(f"데이터 누락 발견: {missing_count}개 봉")
self.logger.warning(f"네트워크 지연 감지: {delay_ms}ms")

# 에러 로그
self.logger.error(f"API 호출 실패: {error_message}")
self.logger.error(f"백테스트 중단: {exception}")

# 거래 로그
self.logger.info(f"포지션 진입: {side} / 가격: {price:,.2f} / 수량: {amount:.6f}")
self.logger.info(f"포지션 청산: 수익률: {pnl_pct:.2f}% / 실현손익: {realized_pnl:.6f} BTC")
```

## 📊 테스트 가이드라인

### 1. 단위 테스트
```python
def test_sma_calculation(self):
    """SMA 계산 함수 테스트"""
    # 테스트 데이터 준비
    test_data = pd.DataFrame({
        'close': [100, 102, 101, 103, 105, 104, 106, 108]
    })
    
    # SMA 계산
    result = self.strategy.calculate_sma(test_data, period=3)
    
    # 기대값과 비교
    expected = [np.nan, np.nan, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    np.testing.assert_array_almost_equal(result['sma'].values, expected, decimal=2)
```

### 2. 통합 테스트
```python
def test_full_backtest_workflow(self):
    """전체 백테스트 워크플로우 테스트"""
    # 실제 데이터로 백테스트 실행
    result = self.run_sample_backtest()
    
    # 결과 검증
    assert result['final_balance'] > 0
    assert result['total_trades'] > 0
    assert 'sharpe_ratio' in result
    assert len(result['equity_curve']) > 0
```

## 🚀 배포 가이드라인

### 1. 버전 관리
```python
# 파일 상단에 버전 정보 명시
__version__ = "2.0.0"
__author__ = "CryptoSim 개발팀"
__email__ = "dev@cryptosim.kr"
__status__ = "Production"
```

### 2. 의존성 관리
```python
# requirements.txt 최신 상태 유지
def check_dependencies(self):
    """필수 의존성 확인"""
    required_packages = {
        'ccxt': '>=4.4.90',
        'pandas': '>=2.3.0',
        'numpy': '>=2.3.1',
        'PyYAML': '>=6.0'
    }
    
    for package, version in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            raise ImportError(f"필수 패키지가 설치되지 않았습니다: {package} {version}")
```

### 3. 설정 파일 검증
```python
def validate_config(self, config: dict) -> bool:
    """설정 파일 유효성 검증"""
    required_keys = [
        'exchange.name',
        'trading.symbol',
        'backtest.initial_balance'
    ]
    
    for key in required_keys:
        if not self.get_nested_value(config, key):
            self.logger.error(f"필수 설정이 누락되었습니다: {key}")
            return False
    
    return True
```

---

**작성일**: 2025년 1월 17일  
**버전**: 1.0  
**담당자**: CryptoSim 개발팀

이 문서는 CryptoSim v2.0 개발 시 준수해야 할 모든 규칙과 가이드라인을 포함합니다.
모든 개발자는 이 문서를 숙지하고 일관된 코드 품질을 유지해야 합니다. 