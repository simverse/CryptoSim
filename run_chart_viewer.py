import glob
import argparse
import json
import pandas as pd
import gspread
import numpy as np
from src.config_manager import ConfigManager

def fetch_data_from_gsheet(spreadsheet_name: str, sheet_name: str, creds_path: str) -> pd.DataFrame:
    """Google Sheet에서 데이터를 가져와 DataFrame으로 변환"""
    print(f"'{spreadsheet_name}' 스프레드시트의 '{sheet_name}' 시트에서 데이터를 가져옵니다...")
    try:
        gc = gspread.service_account(filename=creds_path)
        spreadsheet = gc.open(spreadsheet_name)
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        print(f"'{sheet_name}' 시트에서 데이터를 성공적으로 가져왔습니다.")
        return df
    except Exception as e:
        print(f"'{sheet_name}' 시트 조회 중 오류 발생: {e}")
        return pd.DataFrame()

def process_data_for_chart(df_ohlc: pd.DataFrame, df_trades: pd.DataFrame, strategy_name: str) -> str:
    """DataFrame을 Lightweight Charts 형식의 JSON으로 변환"""
    if df_ohlc.empty:
        return json.dumps({'ohlc': [], 'markers': [], 'position_sizes': [], 'strategy': 'sma'})


    # --- OHLC 및 지표 데이터 처리 ---
    column_map = {'시간': 'time', '시가': 'open', '고가': 'high', '저가': 'low', '종가': 'close', '총 롱크기': 'total_long_size','총 숏크기': 'total_short_size'}
    if strategy_name == 'psar':
        column_map.update({'PSAR': 'psar', 'h-PSAR': 'h_psar', 'RSI': 'rsi'})
        sma_cols = [col for col in df_ohlc.columns if col.startswith('SMA_')]
        for col in sma_cols: column_map[col] = col.lower()
    elif strategy_name == 'sma':
        column_map.update({'RSI': 'rsi'}) # RSI 컬럼 추가
        sma_cols = [col for col in df_ohlc.columns if col.startswith('SMA_')]
        for col in sma_cols: column_map[col] = col.lower()

    df_ohlc.rename(columns=column_map, inplace=True)
    df_ohlc.replace({np.nan: None}, inplace=True)
    
    final_cols = ['time', 'open', 'high', 'low', 'close']
    final_cols.extend([v for k, v in column_map.items() 
                        if k not in ['시간', '시가', '고가', '저가', '종가'] and v in df_ohlc.columns])
    
    chart_data = df_ohlc[[col for col in final_cols if col in df_ohlc.columns]].copy()
    
    if 'time' not in chart_data:
        print(500, description="'시세분석' 시트에 '시간' 컬럼이 없습니다.")
        return
        
    chart_data['time'] = pd.to_datetime(chart_data['time']).astype(np.int64) // 10**9


    # --- 포지션 크기 데이터 처리 ---
    position_size_data = []
    if not df_ohlc.empty and 'total_long_size' in df_ohlc.columns and 'total_short_size' in df_ohlc.columns:
        pos_cols = ['time', 'total_long_size', 'total_short_size']
        
        position_data = df_ohlc[[col for col in pos_cols if col in df_ohlc.columns]].copy()
        position_data['time'] = pd.to_datetime(position_data['time']).astype(np.int64) // 10**9
        
        if 'time' not in position_data:
            print(500, description="'시세분석' 시트에 '시간' 컬럼이 없습니다.")
            return
            
        position_size_data = position_data.to_dict(orient='records')        


    # --- 마커 데이터 처리 (거래내역 시트 기반) ---
    markers = []

    if not df_trades.empty:
        # 차트에 표시될 시간 범위(타임스탬프) 가져오기
        chart_start_time = chart_data['time'].iloc[0]
        chart_end_time = chart_data['time'].iloc[-1]

        # 거래내역의 시간을 UTC 타임스탬프로 변환 (OHLC 데이터와 동일한 방식)
        df_trades['진입시간_unix'] = pd.to_datetime(df_trades['진입시간']).astype(np.int64) // 10**9
        df_trades['청산시간_unix'] = pd.to_datetime(df_trades['청산시간'], errors='coerce').astype(np.int64) // 10**9
        
        entry_map = {
            'long': {'shape': 'arrowUp', 'color': '#2962FF', 'open_color': '#00BCD4'},
            'short': {'shape': 'arrowDown', 'color': '#FF0400', 'open_color': '#FF9800'}
        }

        for _, row in df_trades.iterrows():
            # 차트 시간 범위 밖의 진입은 건너뜀
            if not (chart_start_time <= row['진입시간_unix'] <= chart_end_time):
                continue

            side_info = entry_map.get(row['포지션'])
            if not side_info: continue
            
            is_open = row['청산이유'] == '미청산'
            color = side_info['open_color'] if is_open else side_info['color']
            position = 'belowBar' if row['포지션'] == 'long' else 'aboveBar'

            markers.append({
                'time': row['진입시간_unix'],
                'position': position, 'color': color, 'shape': side_info['shape']
            })

        exit_map = { 'long': {'shape': 'circle', 'color': '#2962FF'}, 'short': {'shape': 'circle', 'color': '#FF0400'} }
        closed_trades = df_trades[df_trades['청산이유'] != '미청산'].copy()
        if not closed_trades.empty:
            closed_trades.dropna(subset=['청산시간_unix'], inplace=True)
            
            # # 차트 시간 범위 밖의 청산은 필터링
            # closed_trades = closed_trades[
            #     (closed_trades['청산시간_unix'] >= chart_start_time) & (closed_trades['청산시간_unix'] <= chart_end_time)
            # ]

            if not closed_trades.empty:
                unique_exits = closed_trades.groupby(['청산시간_unix', '포지션']).first().reset_index()
                for _, row in unique_exits.iterrows():
                    side_info = exit_map.get(row['포지션'])
                    if not side_info: continue
                    
                    position = 'belowBar' if row['포지션'] == 'long' else 'aboveBar'
                    markers.append({
                        'time': row['청산시간_unix'],
                        'position': position, 'color': side_info['color'], 'shape': side_info['shape']
                    })

            # 'X' 마커 추가 (매수신호는 있으나 거래되지 않은 경우)                                
            if '거래ID' in df_ohlc.columns:
                # short 
                missed_trades = df_ohlc[df_ohlc['거래ID'] == 'XS'].copy()
                if not missed_trades.empty:
                    missed_trades['time_unix'] = pd.to_datetime(missed_trades['time']).astype(np.int64) // 10**9
                    for _, row in missed_trades.iterrows():
                        markers.append({
                            'time': row['time_unix'],
                            'position': 'aboveBar',
                            'color': '#808080',  # Gray
                            'shape': 'arrowDown'
                        })
                # long
                missed_trades = df_ohlc[df_ohlc['거래ID'] == 'XL'].copy()
                if not missed_trades.empty:
                    missed_trades['time_unix'] = pd.to_datetime(missed_trades['time']).astype(np.int64) // 10**9
                    for _, row in missed_trades.iterrows():
                        markers.append({
                            'time': row['time_unix'],
                            'position': 'belowBar',
                            'color': '#808080',  # Gray
                            'shape': 'arrowUp'
                        })

    markers.sort(key=lambda x: x['time'])

    # --- 최종 데이터 구성 ---
    final_data = {
        'strategy': strategy_name,
        'ohlc': chart_data.to_dict(orient='records'),
        'markers': markers,
        'position_sizes': position_size_data
    }
    return json.dumps(final_data)

def create_chart_html(template_path: str, data_json: str) -> str:
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    fetch_str = "const response = await fetch('/api/data');"
    replace_str = f"const data = {data_json};"
    if fetch_str in template:
        template = template.replace(fetch_str, "/* fetch call replaced by embedded data */")
        template = template.replace("if (!response.ok) throw new Error(`Failed to load data: ${response.status} ${await response.text()}`);", "")
        template = template.replace("const data = await response.json();", replace_str)
    return template

def main():
    parser = argparse.ArgumentParser(description="Google Sheet 데이터를 읽어 차트 웹페이지를 생성합니다.")
    parser.add_argument('--ohlc_sheet', default='시세분석', help='OHLC 데이터를 읽어올 시트 이름')
    parser.add_argument('--trades_sheet', default='거래내역', help='거래내역 데이터를 읽어올 시트 이름')
    args = parser.parse_args()

    config_manager = ConfigManager()
    recorder_config = config_manager.get_recorder_config().get('google_sheet', {})
    spreadsheet_name = recorder_config.get('spreadsheet_name')
    creds_path = recorder_config.get('credentials_path')
    strategy = config_manager.get('strategy.name')

    if not spreadsheet_name or not creds_path:
        print("오류: config.yaml 파일에 Google Sheet 설정('spreadsheet_name', 'credentials_path')이 필요합니다.")
        return

    df_ohlc = fetch_data_from_gsheet(spreadsheet_name, args.ohlc_sheet, creds_path)
    df_trades = fetch_data_from_gsheet(spreadsheet_name, args.trades_sheet, creds_path)

    if df_ohlc.empty:
        print(f"'{args.ohlc_sheet}' 시트에 데이터가 없어 차트를 생성할 수 없습니다.")
        return
        
    chart_data_json = process_data_for_chart(df_ohlc, df_trades,strategy)

    template_path = 'templates/chart.html'
    chart_html_content = create_chart_html(template_path, chart_data_json)
    
    output_html_path = 'gsheet_chart.html'
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(chart_html_content)
        
    print(f"성공! '{output_html_path}' 파일이 생성되었습니다.")
    print("해당 파일을 웹 브라우저에서 직접 열어 차트를 확인하세요.")

if __name__ == '__main__':
    main()