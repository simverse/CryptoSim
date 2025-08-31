import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np # numpy 임포트 추가

from src.strategy import BaseStrategy # BaseStrategy 임포트

class BacktestRecorder:
    """백테스트 결과 기록 및 분석 클래스"""
    
    def __init__(self, initial_balance: float, output_dir: str , report_start_date: str ):
        self.output_dir = output_dir
        self.report_start_date = pd.to_datetime(report_start_date).tz_localize('Asia/Seoul')
        self.initial_balance = initial_balance
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """출력 디렉토리 확인 및 생성"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def _create_ohlc_signals_sheet(self, original_data: pd.DataFrame, signals_data: pd.DataFrame, detailed_records: List[Dict] = None, strategy: BaseStrategy = None) -> pd.DataFrame:
        """OHLC + 지표 + 신호 + 상세 거래 정보 시트 생성 (전략별 동적 컬럼)"""
        
        # 1. 기준 데이터프레임 생성
        df = signals_data[signals_data.index >= self.report_start_date].copy()
        
        # 2. 상세 기록을 데이터프레임으로 변환하고 인덱스 타입 통일
        if not detailed_records:
            return pd.DataFrame() # 상세 기록 없으면 빈 DF 반환

        detailed_df = pd.DataFrame(detailed_records)
        detailed_df['timestamp'] = pd.to_datetime(detailed_df['timestamp'])
        detailed_df.set_index('timestamp', inplace=True)

        # 3. 최종 리포트 생성 및 기본 컬럼 할당
        report_df = pd.DataFrame(index=df.index)
        
        report_df['시간'] = df.index.to_series().dt.strftime('%Y-%m-%d %H:%M:%S')
        report_df['시가'] = df['open']
        report_df['고가'] = df['high']
        report_df['저가'] = df['low']
        report_df['종가'] = df['close']

        # 4. 전략별 지표 컬럼 동적 추가
        if strategy:
            if strategy.name == "SMA Crossover Strategy":
                report_df[f'SMA_{strategy.short_sma_period}'] = df.get(f'sma_{strategy.short_sma_period}')
                report_df[f'SMA_{strategy.mid_sma_period}'] = df.get(f'sma_{strategy.mid_sma_period}')
                report_df[f'SMA_{strategy.long_sma_period}'] = df.get(f'sma_{strategy.long_sma_period}')
                report_df['MA_DLSMA'] = df.get('long_sma_change')
                report_df['RSI'] = df.get('rsi')
            elif strategy.name == "Parabolic SAR Strategy":
                report_df['PSAR'] = df.get('psar')
                report_df['h-PSAR'] = df.get('h-psar')
                report_df[f'SMA_{strategy.short_sma_period}'] = df.get(f'sma_{strategy.short_sma_period}')
                report_df[f'SMA_{strategy.long_sma_period}'] = df.get(f'sma_{strategy.long_sma_period}')
                report_df['RSI'] = df.get('rsi')
        
        # 신호 정보
        report_df['매수신호'] = detailed_df['buy_signal'].reindex(df.index).fillna('')
        report_df['매도신호'] = detailed_df['sell_signal'].reindex(df.index).fillna('')
        
        # 거래 정보
        report_df['거래ID'] = detailed_df['position_id'].reindex(df.index).fillna('')
        report_df['거래방향'] = detailed_df['trade_direction'].reindex(df.index).fillna('')
        report_df['거래가격'] = detailed_df['trade_price'].reindex(df.index).apply(lambda x: f"{x:.2f}" if pd.notna(x) else '')
        report_df['거래크기'] = detailed_df['trade_size'].reindex(df.index).apply(lambda x: f"{x:.6f}" if pd.notna(x) else '')
        report_df['거래수익'] = detailed_df['realized_pnl'].reindex(df.index).apply(lambda x: f"{x:.8f}" if pd.notna(x) else '')

        # 매수신호는 있으나 거래ID가 없는 경우 'X'로 표시 (최대치 도달)
        report_df.loc[(report_df['매수신호'] == '롱진입') & (report_df['거래ID'] == ''), '거래ID'] = 'XL'        
        report_df.loc[(report_df['매수신호'] == '숏진입') & (report_df['거래ID'] == ''), '거래ID'] = 'XS'

        # 포지션 정보
        import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np
try:
    import gspread
    from gspread_dataframe import set_with_dataframe
except ImportError:
    gspread = None
    set_with_dataframe = None


from src.strategy import BaseStrategy
from src.config_manager import ConfigManager

class BacktestRecorder:
    """백테스트 결과 기록 및 분석 클래스"""
    
    def __init__(self, initial_balance: float, output_dir: str, report_start_date: str, config_manager: ConfigManager):
        self.output_dir = output_dir
        self.report_start_date = pd.to_datetime(report_start_date).tz_localize('Asia/Seoul')
        self.initial_balance = initial_balance
        self.config_manager = config_manager
        self.recorder_config = self.config_manager.get_recorder_config()
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """출력 디렉토리 확인 및 생성"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_report(self, reports: Dict[str, pd.DataFrame], base_filename: str):
        """리포트를 설정에 따라 저장"""
        if self.recorder_config.get('excel', {}).get('enabled', False):
            excel_path = self._save_to_excel(reports, base_filename)
            print(f"백테스트 리포트가 Excel 파일로 저장되었습니다: {excel_path}")

        if self.recorder_config.get('google_sheet', {}).get('enabled', False):
            self._save_to_google_sheet(reports, base_filename)

    def _save_to_google_sheet(self, reports: Dict[str, pd.DataFrame], base_filename: str):
        """Google Sheet으로 저장"""
        if gspread is None or set_with_dataframe is None:
            print("Google Sheet 저장을 위해 'gspread'와 'gspread-dataframe' 라이브러리를 설치해주세요.")
            return
            
        try:
            gc = gspread.service_account(filename=self.recorder_config['google_sheet']['credentials_path'])
            spreadsheet = gc.open(self.recorder_config['google_sheet']['spreadsheet_name'])
            
            sheet_names = {
                'ohlc_signals': '시세분석',
                'trade_history': '거래내역'
            }

            for key, df in reports.items():
                sheet_name = f"{base_filename}_{sheet_names.get(key, key)}"
                try:
                    worksheet = spreadsheet.worksheet(sheet_name)
                    worksheet.clear()
                except gspread.WorksheetNotFound:
                    worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="1", cols="1")
                
                if not df.empty:
                    set_with_dataframe(worksheet, df)
            print(f"백테스트 리포트가 Google Sheet에 저장되었습니다: {self.recorder_config['google_sheet']['spreadsheet_name']}")

        except Exception as e:
            print(f"Google Sheet 저장 실패: {e}")
            
    def _create_ohlc_signals_sheet(self, original_data: pd.DataFrame, signals_data: pd.DataFrame, detailed_records: List[Dict] = None, strategy: BaseStrategy = None) -> pd.DataFrame:
        """OHLC + 지표 + 신호 + 상세 거래 정보 시트 생성 (전략별 동적 컬럼)"""
        
        # 1. 기준 데이터프레임 생성
        df = signals_data[signals_data.index >= self.report_start_date].copy()
        
        # 2. 상세 기록을 데이터프레임으로 변환하고 인덱스 타입 통일
        if not detailed_records:
            return pd.DataFrame() # 상세 기록 없으면 빈 DF 반환

        detailed_df = pd.DataFrame(detailed_records)
        detailed_df['timestamp'] = pd.to_datetime(detailed_df['timestamp'])
        detailed_df.set_index('timestamp', inplace=True)

        # 3. 최종 리포트 생성 및 기본 컬럼 할당
        report_df = pd.DataFrame(index=df.index)
        
        report_df['시간'] = df.index.to_series().dt.strftime('%Y-%m-%d %H:%M:%S')
        report_df['시가'] = df['open']
        report_df['고가'] = df['high']
        report_df['저가'] = df['low']
        report_df['종가'] = df['close']

        # 4. 전략별 지표 컬럼 동적 추가
        if strategy:
            if strategy.name == "SMA Crossover Strategy":
                report_df[f'SMA_{strategy.short_sma_period}'] = df.get(f'sma_{strategy.short_sma_period}')
                report_df[f'SMA_{strategy.mid_sma_period}'] = df.get(f'sma_{strategy.mid_sma_period}')
                report_df[f'SMA_{strategy.long_sma_period}'] = df.get(f'sma_{strategy.long_sma_period}')
                report_df['MA_DLSMA'] = df.get('long_sma_change')
                report_df['RSI'] = df.get('rsi')
            elif strategy.name == "Parabolic SAR Strategy":
                report_df['PSAR'] = df.get('psar')
                report_df['h-PSAR'] = df.get('h-psar')
                report_df[f'SMA_{strategy.short_sma_period}'] = df.get(f'sma_{strategy.short_sma_period}')
                report_df[f'SMA_{strategy.long_sma_period}'] = df.get(f'sma_{strategy.long_sma_period}')
                report_df['RSI'] = df.get('rsi')
        
        # 신호 정보
        report_df['매수신호'] = detailed_df['buy_signal'].reindex(df.index).fillna('')
        report_df['매도신호'] = detailed_df['sell_signal'].reindex(df.index).fillna('')
        
        # 거래 정보
        report_df['거래ID'] = detailed_df['position_id'].reindex(df.index).fillna('')
        report_df['거래방향'] = detailed_df['trade_direction'].reindex(df.index).fillna('')
        report_df['거래가격'] = detailed_df['trade_price'].reindex(df.index).apply(lambda x: f"{x:.2f}" if pd.notna(x) else '')
        report_df['거래크기'] = detailed_df['trade_size'].reindex(df.index).apply(lambda x: f"{x:.6f}" if pd.notna(x) else '')
        report_df['거래수익'] = detailed_df['realized_pnl'].reindex(df.index).apply(lambda x: f"{x:.8f}" if pd.notna(x) else '')

        # 매수신호는 있으나 거래ID가 없는 경우 'X'로 표시 (최대치 도달)
        report_df.loc[(report_df['매수신호'] == '롱진입') & (report_df['거래ID'] == ''), '거래ID'] = 'XL'        
        report_df.loc[(report_df['매수신호'] == '숏진입') & (report_df['거래ID'] == ''), '거래ID'] = 'XS'

        # 포지션 정보
        report_df['총 롱크기'] = detailed_df['total_long_size'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")
        report_df['롱Set'] = detailed_df['long_sets'].reindex(df.index).ffill().fillna('0(0.000000)')
        report_df['총 롱진입가'] = detailed_df['total_long_avg_price'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.2f}")
        report_df['총 롱가치'] = detailed_df['total_long_value'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")
        
        report_df['총 숏크기'] = detailed_df['total_short_size'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")
        report_df['숏Set'] = detailed_df['short_sets'].reindex(df.index).ffill().fillna('0(0.000000)')
        report_df['총 숏진입가'] = detailed_df['total_short_avg_price'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.2f}")
        report_df['총 숏가치'] = detailed_df['total_short_value'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")

        report_df['포지션크기'] = (report_df['총 롱크기'].astype(float) + report_df['총 숏크기'].astype(float)).apply(lambda x: f"{x:.6f}" if x > 1e-9 else '')

        # 수익률 및 계좌 정보 (사용자 요청 순서)
        report_df['포지션수익율(%)'] = detailed_df['current_return'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")
        report_df['총마진(BTC)'] = detailed_df['total_margin'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")
        report_df['거래잔고(BTC)'] = detailed_df['trade_balance'].reindex(df.index).ffill().fillna(self.initial_balance).apply(lambda x: f"{x:.6f}")
        report_df['총자산(BTC)'] = detailed_df['total_equity'].reindex(df.index).ffill().fillna(self.initial_balance).apply(lambda x: f"{x:.6f}")
        report_df['누적수익률(%)'] = detailed_df['cumulative_return'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")
        report_df['미실현손익_차이'] = detailed_df['unrealized_pnl_diff'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.8f}")

        return report_df.reset_index(drop=True)

    def _create_trade_history_sheet(self, trades: List[Dict]) -> pd.DataFrame:
        """완료된 거래 및 미청산 포지션 내역 시트 생성"""
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        
        # 수익률 계산 (투입 마진 대비)
        # margin이 0인 경우를 대비하여 오류 방지
        df['수익률(%)'] = np.where(df['margin'] > 0, (df['net_pnl'] / df['margin']) * 100, 0)

        # 컬럼 선택 및 이름 변경
        report_df = df[[
            'entry_time', 'exit_time', 'side', 'amount', 'entry_price', 
            'exit_price', 'margin', 'net_pnl', '수익률(%)', 'exit_reason'
        ]].copy()
        
        report_df.columns = [
            '진입시간', '청산시간', '포지션', '수량(BTC)', '진입가격', 
            '청산가격', '투입마진(BTC)', '손익(BTC)', '수익률(%)', '청산이유'
        ]

        # 시간 형식 변경 (NaT 값을 빈 문자열로 처리)
        report_df['진입시간'] = pd.to_datetime(report_df['진입시간']).dt.strftime('%Y-%m-%d %H:%M:%S')
        report_df['청산시간'] = pd.to_datetime(report_df['청산시간'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

        return report_df
        
    def _save_to_excel(self, reports: Dict[str, pd.DataFrame], base_filename: str) -> str:
        """Excel 파일로 저장"""
        
        excel_path = os.path.join(self.output_dir, f"{base_filename}_backtest_report.xlsx")
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                sheet_names = {
                    'ohlc_signals': '시세분석',
                    'trade_history': '거래내역'
                }
                
                for key, df in reports.items():
                    sheet_name = sheet_names.get(key, key)
                    if not df.empty:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
            return excel_path
            
        except Exception as e:
            print(f"Excel 저장 실패: {e}")
            # CSV 저장은 주 시트만 저장
            if 'ohlc_signals' in reports:
                csv_path = os.path.join(self.output_dir, f"{base_filename}_ohlc_report.csv")
                reports['ohlc_signals'].to_csv(csv_path, index=False, encoding='utf-8-sig')
                return csv_path
            return ""

        report_df['롱Set'] = detailed_df['long_sets'].reindex(df.index).ffill().fillna('0(0.000000)')
        report_df['총 롱진입가'] = detailed_df['total_long_avg_price'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.2f}")
        report_df['총 롱가치'] = detailed_df['total_long_value'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")
        
        report_df['총 숏크기'] = detailed_df['total_short_size'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")
        report_df['숏Set'] = detailed_df['short_sets'].reindex(df.index).ffill().fillna('0(0.000000)')
        report_df['총 숏진입가'] = detailed_df['total_short_avg_price'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.2f}")
        report_df['총 숏가치'] = detailed_df['total_short_value'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")

        report_df['포지션크기'] = (report_df['총 롱크기'].astype(float) + report_df['총 숏크기'].astype(float)).apply(lambda x: f"{x:.6f}" if x > 1e-9 else '')

        # 수익률 및 계좌 정보 (사용자 요청 순서)
        report_df['포지션수익율(%)'] = detailed_df['current_return'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")
        report_df['총마진(BTC)'] = detailed_df['total_margin'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")
        report_df['거래잔고(BTC)'] = detailed_df['trade_balance'].reindex(df.index).ffill().fillna(self.initial_balance).apply(lambda x: f"{x:.6f}")
        report_df['총자산(BTC)'] = detailed_df['total_equity'].reindex(df.index).ffill().fillna(self.initial_balance).apply(lambda x: f"{x:.6f}")
        report_df['누적수익률(%)'] = detailed_df['cumulative_return'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.6f}")
        report_df['미실현손익_차이'] = detailed_df['unrealized_pnl_diff'].reindex(df.index).ffill().fillna(0).apply(lambda x: f"{x:.8f}")

        return report_df.reset_index(drop=True)

    def _create_trade_history_sheet(self, trades: List[Dict]) -> pd.DataFrame:
        """완료된 거래 및 미청산 포지션 내역 시트 생성"""
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        
        # 수익률 계산 (투입 마진 대비)
        # margin이 0인 경우를 대비하여 오류 방지
        df['수익률(%)'] = np.where(df['margin'] > 0, (df['net_pnl'] / df['margin']) * 100, 0)

        # 컬럼 선택 및 이름 변경
        report_df = df[[
            'entry_time', 'exit_time', 'side', 'amount', 'entry_price', 
            'exit_price', 'margin', 'net_pnl', '수익률(%)', 'exit_reason'
        ]].copy()
        
        report_df.columns = [
            '진입시간', '청산시간', '포지션', '수량(BTC)', '진입가격', 
            '청산가격', '투입마진(BTC)', '손익(BTC)', '수익률(%)', '청산이유'
        ]

        # 시간 형식 변경 (NaT 값을 빈 문자열로 처리)
        report_df['진입시간'] = pd.to_datetime(report_df['진입시간']).dt.strftime('%Y-%m-%d %H:%M:%S')
        report_df['청산시간'] = pd.to_datetime(report_df['청산시간'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

        return report_df
        
    def _save_to_excel(self, reports: Dict[str, pd.DataFrame], base_filename: str) -> str:
        """Excel 파일로 저장"""
        
        excel_path = os.path.join(self.output_dir, f"{base_filename}_backtest_report.xlsx")
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                sheet_names = {
                    'ohlc_signals': '시세분석',
                    'trade_history': '거래내역' # 시트 이름 추가
                }
                
                for key, df in reports.items():
                    sheet_name = sheet_names.get(key, key)
                    if not df.empty:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
            print(f"백테스트 리포트가 저장되었습니다: {excel_path}")
            return excel_path
            
        except Exception as e:
            print(f"Excel 저장 실패: {e}")
            # CSV 저장은 주 시트만 저장
            if 'ohlc_signals' in reports:
                csv_path = os.path.join(self.output_dir, f"{base_filename}_ohlc_report.csv")
                reports['ohlc_signals'].to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"시세분석 리포트가 CSV로 저장되었습니다: {csv_path}")
                return csv_path
            return ""