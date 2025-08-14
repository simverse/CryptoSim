import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

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
            
    def _create_ohlc_signals_sheet(self, original_data: pd.DataFrame, signals_data: pd.DataFrame, detailed_records: List[Dict] = None) -> pd.DataFrame:
        """OHLC + 지표 + 신호 + 상세 거래 정보 시트 생성 (안정화된 버전)"""
        
        # 1. 기준 데이터프레임 생성
        df = signals_data[signals_data.index >= self.report_start_date].copy()
        
        # 2. 상세 기록을 데이터프레임으로 변환하고 인덱스 타입 통일
        if not detailed_records:
            return pd.DataFrame() # 상세 기록 없으면 빈 DF 반환

        detailed_df = pd.DataFrame(detailed_records)
        detailed_df['timestamp'] = pd.to_datetime(detailed_df['timestamp'])
        detailed_df.set_index('timestamp', inplace=True)

        # 3. 최종 리포트 생성 및 컬럼 할당
        report_df = pd.DataFrame(index=df.index)
        
        # 기본 정보
        report_df['시간'] = df.index.to_series().dt.strftime('%Y-%m-%d %H:%M:%S')
        report_df['시가'] = df['open']
        report_df['고가'] = df['high']
        report_df['저가'] = df['low']
        report_df['종가'] = df['close']
        report_df['SMA_24'] = df.get('sma_24')
        report_df['SMA_720'] = df.get('sma_720')
        
        # 신호 정보
        report_df['매수신호'] = detailed_df['buy_signal'].reindex(df.index).fillna('')
        report_df['매도신호'] = detailed_df['sell_signal'].reindex(df.index).fillna('')
        
        # 거래 정보
        report_df['거래ID'] = detailed_df['position_id'].reindex(df.index).fillna('')
        report_df['거래방향'] = detailed_df['trade_direction'].reindex(df.index).fillna('')
        report_df['거래가격'] = detailed_df['trade_price'].reindex(df.index).apply(lambda x: f"{x:.2f}" if pd.notna(x) else '')
        report_df['거래크기'] = detailed_df['trade_size'].reindex(df.index).apply(lambda x: f"{x:.6f}" if pd.notna(x) else '')
        report_df['거래수익'] = detailed_df['realized_pnl'].reindex(df.index).apply(lambda x: f"{x:.8f}" if pd.notna(x) else '')

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
                        
            print(f"백테스트 리포트가 저장되었습니다: {excel_path}")
            return excel_path
            
        except Exception as e:
            print(f"Excel 저장 실패: {e}")
            csv_path = os.path.join(self.output_dir, f"{base_filename}_backtest_report.csv")
            reports['ohlc_signals'].to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"리포트가 CSV로 저장되었습니다: {csv_path}")
            return csv_path