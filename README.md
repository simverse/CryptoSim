<<<<<<< HEAD
# CryptoSim
=======
# CryptoSim - 올인원 백테스트 시뮬레이션

## 📋 프로젝트 개요

CryptoBot에서 검증된 `run_backtest_standalone.py` 스크립트를 기반으로 한 독립적인 백테스트 시뮬레이션 환경입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 활성화 (conda 환경)
conda activate crypto

# 또는 새 가상환경 생성
conda create -n cryptosim python=3.8
conda activate cryptosim

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정

`config/config.yaml` 파일에서 OKX API 키를 설정하거나 환경변수를 사용하세요:

```bash
# 환경변수 설정 (권장)
set OKX_API_KEY=your_api_key
set OKX_SECRET_KEY=your_secret_key  
set OKX_PASSPHRASE=your_passphrase
set OKX_SANDBOX=true
```

### 3. 백테스트 실행

```bash
# 기본 실행 (BTC/USDT:USDT, 1시간봉, 2000시간)
python run_backtest_standalone.py --symbol BTC/USDT:USDT --timeframe 1h --hours 2000

# 다른 설정으로 실행
python run_backtest_standalone.py --symbol BTC/USDT:USDT --timeframe 1h --hours 1000 --initial-balance 2.0
```

## ⚙️ 주요 옵션

- `--symbol`: 거래 심볼 (기본값: BTC/USDT:USDT)
- `--timeframe`: 시간프레임 (기본값: 1h)
- `--hours`: 거래 기간 (시간 단위)
- `--initial-balance`: 초기 자본 (BTC 단위, 기본값: 1.0)
- `--report-start`: 리포트 시작 날짜
- `--sandbox`: 샌드박스 모드 사용

## 📊 결과 확인

백테스트 완료 후 `logs/` 디렉토리에 Excel 보고서가 자동 생성됩니다:
- 거래 내역
- 자산 곡선  
- 성과 분석
- OHLC 데이터 및 신호

## 🔧 설정 파일

`config/config.yaml`에서 다음 설정을 변경할 수 있습니다:
- 거래 파라미터 (레버리지, 포지션 크기 등)
- SMA 전략 설정 (단기/장기 이동평균 기간)
- 수수료 설정
- 초기 자본

## 📈 전략 정보

현재 구현된 전략: **SMA 크로스오버 전략**
- 단기 SMA: 24봉
- 장기 SMA: 720봉  
- 골든 크로스 시 롱 포지션 진입
- 데드 크로스 시 숏 포지션 진입
- 2% 익절 설정

## 🔒 보안 주의사항

- API 키는 반드시 환경변수로 설정하세요
- 테스트용으로는 OKX 샌드박스 환경을 사용하세요
- 실제 거래 전 충분한 백테스팅을 권장합니다

## 📋 요구사항

- Python 3.8+
- OKX API 키 (샌드박스 또는 실거래)
- 인터넷 연결 (실시간 데이터 수집용)

## 🎯 다음 단계

1. 다양한 시장 조건에서 백테스트 실행
2. 전략 파라미터 최적화
3. 새로운 전략 개발 및 테스트
4. 실제 거래 시스템으로 발전

---

**📞 문의**: CryptoBot 프로젝트에서 분리된 독립 실행 환경입니다.
>>>>>>> 553ca90 (Initial commit)
