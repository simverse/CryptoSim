import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import random
import multiprocessing

import os
from dotenv import load_dotenv

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False



# .env 파일에서 환경변수 로드
load_dotenv()

# 환경변수에서 API 키 정보 읽어오기
API_KEY = os.getenv('OKX_API_KEY')
API_SECRET = os.getenv('OKX_SECRET_KEY') 
API_PASSPHRASE = os.getenv('OKX_PASSPHRASE')

class SubAccount:
    def __init__(self, initial_value, avg_entry_price, position, margin, depth = 0, entry_index=None ):
        self.initial_value = initial_value
        self.avg_entry_price = avg_entry_price
        self.position = position
        self.margin = margin
        self.depth  = depth
        self.entry_index = entry_index

        self.active = True     
        self.amount = position * avg_entry_price

class Account:                
    def __init__(self, initial_value, buy_unit, leverage=1,  is_long=True, max_account=1):
        
        self.is_long = is_long  # True: 롱, False: 숏
        self.initial_asset = initial_value
        self.buy_unit = buy_unit
        self.leverage = leverage

        # 하위 계좌 관리
        self.sub_accounts = []

        self.max_account = max_account

    def set_account(self, df,  profit_rate=0.02, loss_rate=0.02,  trading_fee_rate=0.0001, funding_fee_rate=0.0001):

        # 수동 입력 필요한 부분
        self.df = df
        data_size = len(self.df)

        # OKX VIP1 기준 Maker fee 0.01% , Taker fee 0.03%  -- 주로 지정가 주문이므로 Maker fee 적용
        self.trading_fee_rate = trading_fee_rate
        # 8시간마다 지불되는 펀딩피 : 실제로는 유동적으로 변동됨     
        self.funding_fee_rate = funding_fee_rate

        self.profit_rate = profit_rate
        self.loss_rate = loss_rate    

        # 봉별 기록용 리스트
        self.balances = [0] * data_size                     # 실현자산
        self.asset_values = [0] * data_size                 # 평가자산
        self.positions = [0] * data_size                    # 각 봉별 포지션(수량)
        self.amounts = [0] * data_size                      # 각 봉별 투자규모(진입금액)
        self.margins = [0] * data_size                      # 각 봉별 투자금(마진)
        self.returns = [0] * data_size                      # 각 봉별 수익률
        self.avg_entry_prices = [0] * data_size             # 각 봉별 평균 진입가

        self.sell_amounts = [0] * data_size                 # 각 봉별 매도 금액
        self.buy_amounts = [0] * data_size                  # 각 봉별 매수 금액

        self.funding_fees = [0] * data_size                 # 각 봉별 펀딩피

        self.total_amounts = [0] * data_size                # 전체 투자규모         
        self.total_margins = [0] * data_size                # 전체 증거금
        self.total_asset_values = [0] * data_size           # 평가자산

        self.sub_account_cnts = [0] * data_size            # 추가 계좌 수

        self.total_asset_values[0] = 0

        self.balances[0] = 0
        self.asset_values[0] = 0


    def set_filename(self, filename):

        name, ext = os.path.splitext(filename)

        self.data_filename = f"{name}_long.xlsx" if self.is_long else f"{name}_short.xlsx"
        self.img_filename = self.data_filename.replace('.xlsx', '.png')

    def check_open_position(self, index):

        if self.initial_asset - self.margins[index] < self.buy_unit:
            return
        
        buy_price = self.df['close'].iloc[index] * (1 + self.trading_fee_rate if self.is_long else 1 - self.trading_fee_rate)

        self.buy_amounts[index] = self.buy_unit * self.leverage

        self.amounts[index] +=  self.buy_amounts[index]
        cur_position = self.buy_amounts[index] / buy_price
        self.positions[index] += cur_position
        self.margins[index] +=  self.buy_unit

        if self.positions[index-1] > 0:
            self.avg_entry_prices[index] = (
                (self.positions[index-1] * self.avg_entry_prices[index-1] + cur_position * buy_price) /
                (self.positions[index-1] + cur_position)
            )
        else:
            self.avg_entry_prices[index] = buy_price
        
        self.total_amounts[index] = self.total_amounts[index-1] + self.buy_amounts[index]
        self.total_margins[index] = self.total_margins[index-1] + (self.margins[index] - self.margins[index-1])  

        if self.is_long:
            self.returns[index] = (buy_price / self.avg_entry_prices[index] - 1) * self.leverage
        else:
            self.returns[index] = (1 - buy_price / self.avg_entry_prices[index] ) * self.leverage
    
    def check_margin_full(self, index, total_asset_value, total_balance, total_margin):
        
        # sub account 추가 조건
        # 계좌 평가 금액이 계좌 초기 금액보다 작으면 생성 금지
        if total_asset_value < self.initial_asset  or len(self.sub_accounts) >= self.max_account :
            return

        if self.positions[index] > 0 and (self.initial_asset - self.margins[index] < self.buy_unit):

            # 현재 계좌에 여유가 될때만 추가 진행
            if total_balance - total_margin > self.initial_asset :

                sub_account = SubAccount(self.initial_asset, self.avg_entry_prices[index], self.positions[index], self.margins[index], len(self.sub_accounts), index)
                self.sub_accounts.append(sub_account)

                self.margins[index] = 0
                self.positions[index] = 0
                self.amounts[index] = 0

    def check_close_position(self, index ):
        """
        포지션 청산(전량) 함수
        index: 봉 인덱스
        price: 청산 가격
        trading_fee_rate: 수수료율
        leverage: 레버리지
        """
        if self.positions[index] == 0:
            return 0

        pnl = 0
        sell_price = 0

        price_high = self.df['high'].iloc[index]
        price_low = self.df['low'].iloc[index]

        # 수익 계산
        if self.is_long:
            # 롱 익절
            if price_high >= self.avg_entry_prices[index] * (1 + self.profit_rate):
                sell_price = price_high * (1 + self.profit_rate)
                sell_price_after_fee = sell_price * (1 - self.trading_fee_rate)
                pnl = (sell_price_after_fee - self.avg_entry_prices[index]) * self.positions[index]
            # 롱 손절
            elif price_low <= self.avg_entry_prices[index] * (1 - self.loss_rate):   
                sell_price = price_low * (1 - self.loss_rate)
                sell_price_after_fee = sell_price * (1 - self.trading_fee_rate)
                pnl = (sell_price_after_fee - self.avg_entry_prices[index]) * self.positions[index]
        else:
            # 숏 익절
            if price_low <= self.avg_entry_prices[index] * (1 - self.profit_rate):
                sell_price = price_low * (1 - self.profit_rate)
                sell_price_after_fee = sell_price * (1 - self.trading_fee_rate)
                pnl = (self.avg_entry_prices[index] - sell_price_after_fee) * self.positions[index]                
            # 숏 손절
            elif price_high >= self.avg_entry_prices[index] * (1 + self.loss_rate):
                sell_price = price_high * (1 + self.loss_rate)
                sell_price_after_fee = sell_price * (1 - self.trading_fee_rate)
                pnl = (self.avg_entry_prices[index] - sell_price_after_fee) * self.positions[index]
        
        if sell_price != 0:
            sell_mount = sell_price * self.positions[index]
        
            self.sell_amounts[index] += sell_mount
            self.returns[index] = pnl/self.margins[index]
        

            self.total_amounts[index] -= sell_mount
            self.total_margins[index] -= self.margins[index]
            self.balances[index] += pnl

            self.amounts[index] = 0
            self.margins[index] = 0
            self.positions[index] = 0
            self.avg_entry_prices[index] = 0

            #익절 처리시 추가 계좌 청산
            self.clear_sub_account(index)

            return 1
        else:
            return 0


    def check_sub_account (self, index ):
        """
        하위 계좌 청산 함수
        index: 봉 인덱스
        cur_price: 현재가
        """
        pnl = 0

        for sub_account in self.sub_accounts:
            if sub_account.position > 0 and sub_account.active:
                # 현재가로 계산한 수익률이 목표 수익률에 도달했는지 확인                
                if self.is_long:
                    sell_price = sub_account.avg_entry_price * (1 + self.profit_rate)

                    if self.df['high'].iloc[index] >= sell_price:
                        # 익절 실행                    
                        sell_price_after_fee = sell_price * (1 - self.trading_fee_rate)
                        pnl += sub_account.position * (sell_price_after_fee - sub_account.avg_entry_price)

                        sub_account.active = False 

                else:
                    sell_price = sub_account.avg_entry_price * (1 - self.profit_rate)

                    if self.df['low'].iloc[index] <= sell_price:
                        sell_price_after_fee = sell_price * (1 + self.trading_fee_rate)
                        pnl += sub_account.position * (sub_account.avg_entry_price - sell_price_after_fee )

                        sub_account.active = False 
                if not sub_account.active :
                    self.sell_amounts[index] += sub_account.amount
                    self.total_amounts[index] -= sub_account.amount
                    self.total_margins[index] -= sub_account.margin
                    self.balances[index] += (pnl )

        if pnl != 0:
            self.sub_accounts = [sub_account for sub_account in self.sub_accounts if sub_account.active]

        return pnl
    
    def clear_sub_account (self, index ):
        """
        하위 계좌 청산 함수
        index: 봉 인덱스
        """
        pnl = 0

        for sub_account in self.sub_accounts:
            if sub_account.position > 0 and sub_account.active:
                # 현재가로 계산한 수익률이 목표 수익률에 도달했는지 확인                
                if self.is_long:
                    # 현재 가격으로 종결                    
                    sell_price_after_fee = self.df['close'].iloc[index] * (1 - self.trading_fee_rate)
                    pnl += sub_account.position * (sell_price_after_fee - sub_account.avg_entry_price)
                    sub_account.active = False 
                else:
                    # 현재 가격으로 종결                    
                    sell_price_after_fee = self.df['close'].iloc[index] * (1 + self.trading_fee_rate)
                    pnl += sub_account.position * (sub_account.avg_entry_price - sell_price_after_fee )
                    sub_account.active = False 

                if not sub_account.active :
                    self.sell_amounts[index] += sub_account.amount
                    self.total_amounts[index] -= sub_account.amount
                    self.total_margins[index] -= sub_account.margin
                    self.balances[index] += (pnl )

        if pnl != 0:
            self.sub_accounts = [sub_account for sub_account in self.sub_accounts if sub_account.active]

        return pnl
    
    def clear_all_position(self, index):
        """
        모든 포지션 청산
        """
        pnl = 0

        pnl += self.clear_sub_account(index)

        if self.positions[index] == 0:
            return 0

        sell_price = self.df['close'].iloc[index]


        # 수익 계산
        if self.is_long:
            sell_price_after_fee = sell_price * (1 - self.trading_fee_rate)
            pnl = (sell_price_after_fee - self.avg_entry_prices[index]) * self.positions[index]
        else:
            sell_price_after_fee = sell_price * (1 - self.trading_fee_rate)
            pnl = (self.avg_entry_prices[index] - sell_price_after_fee) * self.positions[index]                


        sell_mount = sell_price * self.positions[index]
        
        self.sell_amounts[index] += sell_mount
        self.total_margins[index] -= self.margins[index]

        self.returns[index] = pnl/ abs(self.total_amounts[index-1]  - self.total_amounts[index]) 
        
        self.total_amounts[index] -= sell_mount        
        
        self.balances[index] += pnl

        self.amounts[index] = 0
        self.margins[index] = 0
        self.positions[index] = 0
        self.avg_entry_prices[index] = 0        
        
    
    def apply_funding_fee(self, index):

        funding_fee = 0

        # 메인 계좌에 펀딩피 적용
        if self.positions[index] > 0:
            funding_fee = self.positions[index] * self.avg_entry_prices[index] * self.funding_fee_rate
            
            self.balances[index] -= funding_fee
            self.funding_fees[index] = funding_fee

        # 분리 계좌에도 펀딩피 적용
        for sub_account in self.sub_accounts:   

            if sub_account.position > 0 and sub_account.active:
                funding_fee = sub_account.position * sub_account.avg_entry_price * self.funding_fee_rate
                self.balances[index] -= funding_fee                
                
                self.funding_fees[index] += funding_fee
        
        return self.funding_fees[index] 
    

    def update_state(self, index):

        self.sub_account_cnts[index] = len(self.sub_accounts)

        self.total_amounts[index] =  self.amounts[index] + sum(sub_account.amount for sub_account in self.sub_accounts)
        self.total_margins[index] = self.margins[index] + sum(sub_account.margin for sub_account in self.sub_accounts)

        pnl = self.get_total_pnl(index)

        self.total_asset_values[index] = self.balances[index] + pnl

        if self.sell_amounts[index] == 0:
            self.returns[index] = pnl / self.total_amounts[index]  if  self.total_amounts[index] > 0 else 0


    def get_total_pnl(self, index):
        pnl = 0
        for sub_account in self.sub_accounts:   
            pnl += sub_account.position * (self.df['close'].iloc[index] - sub_account.avg_entry_price)
        
        pnl += self.positions[index] * (self.df['close'].iloc[index] - self.avg_entry_prices[index])

        if not self.is_long:
            pnl = - pnl

        return pnl
    
    def get_sma_cross(self, index):
        # 장기 이평선이 단기 이평선을 위로 교차하면 매수
        if self.df['sma_long'].iloc[index] < self.df['sma_short'].iloc[index] and self.df['sma_short'].iloc[index] < self.df['close'].iloc[index]  and self.df['sma_long'].iloc[index] > self.df['sma_long'].iloc[index-1] :            
            return 1
        elif self.df['sma_long'].iloc[index] > self.df['sma_short'].iloc[index] and self.df['sma_short'].iloc[index] > self.df['close'].iloc[index] and self.df['sma_long'].iloc[index] < self.df['sma_long'].iloc[index-1] :            
            return -1
        else:
            return 0
        
    def is_golden_cross(self, index):

        prev_index = index-1
        while prev_index >= 0:
            if self.df['sma_long'].iloc[prev_index] == self.df['sma_short'].iloc[prev_index] :
                prev_index -= 1
            else:
                break
        next_index = index
        while next_index < len(self.df):
            if self.df['sma_long'].iloc[next_index] == self.df['sma_short'].iloc[next_index] :
                next_index += 1
            else:
                break
     
        #golden cross 체크
        if self.df['sma_long'].iloc[prev_index] < self.df['sma_short'].iloc[prev_index] and self.df['sma_long'].iloc[next_index] > self.df['sma_short'].iloc[next_index] :
            return 1
        #dead cross 체크
        elif self.df['sma_long'].iloc[prev_index] > self.df['sma_short'].iloc[prev_index] and self.df['sma_long'].iloc[next_index] < self.df['sma_short'].iloc[next_index] :
            return -1
        else:   
            return 0
        

    def simulate_iteration(self, index, total_balance, total_asset_value,total_margin):
 
        # 기본적으로 이전 상태 유지
        self.amounts[index] = self.amounts[index-1]
        self.margins[index] = self.margins[index-1]
        self.positions[index] = self.positions[index-1]
        self.avg_entry_prices[index] = self.avg_entry_prices[index-1]
        self.balances[index] = self.balances[index-1]
        # self.asset_values[index] = self.asset_values[index-1]

        # self.total_amounts[index] = self.total_amounts[index-1]
        # self.total_margins[index] = self.total_margins[index-1]
        self.balances[index] = self.balances[index-1]
        # self.total_asset_values[index] = self.total_asset_values[index-1]

        if self.positions[index] > 0:
            self.returns[index] = self.returns[index-1] 
        else:
            self.returns[index] = 0


        bSell = 0           
        
        # 추가 계좌 매도 여부 체크
        self.check_sub_account(index)

        bSell = self.check_close_position(index)

        # 이평선 크로스시 이전 서브 계좌 청산
        if self.total_margins[index-1] > 0 and self.is_golden_cross(index) != 0 :
            self.clear_all_position(index)

        # 매도한 시점에서는 당일 매수 하지 않음
        if bSell == 0:
            if self.is_long:
                # 롱 매수 체크
                if self.df['close'].iloc[index] < self.df['close'].iloc[index-1]  and self.get_sma_cross(index) == 1:
                    # 추가 계좌 수에 따라서 매수 간격을 증가시킴
                    # --- 마진소진 체크 ---            
                    self.check_margin_full(index, total_balance, total_asset_value, total_margin)

                    self.check_open_position(index)
            else:
                # 숏 매수 체크
                if self.df['close'].iloc[index] > self.df['close'].iloc[index-1]  and self.get_sma_cross(index) == -1:
                    # 추가 계좌 수에 따라서 매수 간격을 증가시킴
                    # --- 마진소진 체크 ---            
                    self.check_margin_full(index, total_balance, total_asset_value, total_margin)
                    
                    self.check_open_position(index)            


        #펀딩피 적용 
        if(index-1) % 8 == 0:
            self.funding_fees[index] = self.apply_funding_fee(index)    
            self.balances[index] -= self.funding_fees[index]

        self.update_state(index)

            

    def save_results(self):
        results_df = pd.DataFrame({
            'Date': self.df.index.date,
            'Open': self.df['open'],
            'High': self.df['high'],
            'Low': self.df['low'],
            'Close': self.df['close'],
            'Buy': self.buy_amounts,
            'Sell': self.sell_amounts,
            'Amount': self.amounts,
            'margin_used': self.margins,
            'Position': self.positions,
            'Average_Price': self.avg_entry_prices,
            'Return (%)': self.returns,            
            'margin': self.total_margins,
            'account_depth': self.sub_account_cnts,
            'Balance': self.balances,
            'Asset_values': self.total_asset_values,
            'Funding_fees': self.funding_fees
        })

        results_df.to_excel(self.data_filename, index=False)

        if self.is_long:            
            print(f"백테스트 Long Account : '{self.data_filename}'로 저장되었습니다.")
        else:
            print(f"백테스트 Short Account : '{self.data_filename}'로 저장되었습니다.")

    def plot_results(self, df, symbol=None, timeframe=None):    
        """
        계좌별 백테스트 결과를 명확하게 시각화하는 함수 (Account 전용)
        - 가격, 실현/미실현 자산, 계좌 추가/정리 등 시각화
        - 주요 변수는 Account 클래스의 속성에 맞게 수정
        """
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # 1. 왼쪽 Y축: 가격
        ax1.set_xlabel('시간')
        ax1.set_ylabel('가격', color='black')
        price_line, = ax1.plot(df.index, df['close'], label='가격', color='black', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)

        # SMA 추가 (24, 164, 720)
        sma_periods = [24, 164, 720]
        sma_colors = ['#1976D2', '#FF9800', '#D32F2F']
        for period, color in zip(sma_periods, sma_colors):
            sma = df['close'].rolling(window=period, min_periods=1).mean()
            ax1.plot(df.index, sma, label=f'SMA {period}', color=color, linewidth=1.5)

        # 매수
        buy_idx = [i for i, v in enumerate(self.buy_amounts) if v > 0]
        ax1.scatter(df.index[buy_idx], df['close'].iloc[buy_idx], marker='^', color='green', label='매수', s=80, zorder=5)

        # 매도(수익/손실)
        sell_profit_idx = [i for i in range(1, len(self.sell_amounts)) if self.sell_amounts[i] > 0 and  self.returns[i] > 0]
        sell_loss_idx = [i for i in range(1, len(self.sell_amounts)) if self.sell_amounts[i] > 0 and  self.returns[i] < 0]
        ax1.scatter(df.index[sell_profit_idx], df['close'].iloc[sell_profit_idx], marker='v', color='red', label='매도(수익)', s=80, zorder=5)
        ax1.scatter(df.index[sell_loss_idx], df['close'].iloc[sell_loss_idx], marker='v', color='blue', label='매도(손실)', s=80, zorder=5)

        # 2. 오른쪽 Y축: 실현/미실현 자산 
        ax2 = ax1.twinx()
        ax2.set_ylabel('자산', color='#0072B2')
        min_balance = min(min(self.total_balances), min(self.total_asset_values))
        max_balance = max(max(self.total_balances), max(self.total_asset_values))
        y_margin = (max_balance - min_balance) * 0.05 if max_balance > min_balance else 1
        y_min = min_balance - y_margin
        y_max = max_balance + y_margin
        ax2.fill_between(df.index, self.total_balances, color='#56B4E9', alpha=0.18, label='실현 자산(영역)')
        asset_line = ax2.plot(df.index, self.total_asset_values, label='평가 자산', color='orange', linestyle='-.', linewidth=1.8)
        ax2.axhline(y=self.initial_asset, color='#009E73', linestyle=':', alpha=0.7, label='초기 자본금')
        ax2.tick_params(axis='y', labelcolor='#0072B2')
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax2.set_ylim(y_min, y_max)

        # 3. 마진소진 카운트(별도 보조축, 빨강계열)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_ylabel('추가 계좌수', color='#CC0000')
        ax3.plot(df.index, self.long_account.sub_account_cnts, label='롱 계좌수', color='#CC0000', linestyle='-.', linewidth=1.0)
        ax3.plot(df.index, self.short_account.sub_account_cnts, label='숏 계좌수', color='#0000CC', linestyle='-.', linewidth=1.0)
        ax3.tick_params(axis='y', labelcolor='#CC0000')
        ax3.set_ylim(bottom=0)        

        # 5. 범례 정리 (중복 제거)
        handles, labels = [], []
        for ax in [ax1, ax2, ax3]:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
        ax1.legend(handles, labels, loc='upper left', fontsize=11)

        # 6. 제목, 레이아웃
        title1 = f'Account 백테스트 결과'
        if self.symbol and self.timeframe:
            title1 += f' [심볼: {self.symbol} | 간격: {self.timeframe} | 레버리지: {self.leverage}]'
        title2 = f'초기 자본금: {self.initial_asset} | 단위투자금: {self.buy_unit}'
        plt.title(f'{title1}\n{title2}', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()


        # --- 성과 요약 텍스트 박스 ---
        # 먼저 성과 계산 (속성 없으면 계산)
        if not hasattr(self, 'cumulative_return'):
            self.evaluate_performance()
        perf_text = (
            f"누적수익률: {self.cumulative_return*100:.2f}%\n"
            f"최대낙폭(MDD): {self.mdd*100:.2f}%\n"
            f"샤프지수(연환산): {self.sharpe:.3f}\n"
            f"연환산수익률(CAGR): {self.cagr*100:.2f}%\n"
            f"매수횟수: {self.buy_count}  매도횟수: {self.sell_count}\n"
            f"손익거래비: {self.sell_count_profit / (self.sell_count_loss + self.sell_count_profit) *100. :.2f} %, (수익: {self.sell_count_profit}, 손실: {self.sell_count_loss})\n"
        )

        # 범례(upper left) 바로 오른쪽에, 각 항목별로 세로로 출력
        ax1.text(0.17, 0.98, perf_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontfamily='Malgun Gothic')

        # 그래프를 파일로 저장
        # plt.savefig(self.img_filename)
        # print(f"그래프를 파일로 저장: {self.img_filename}")
        if bshow:
            plt.show()
        # plt.show()
        # plt.close()

        return fig


    def plot_results_interactive(self, df, symbol=None, timeframe=None, bshow=False):
        """
        Plotly를 사용해 웹에서 인터랙티브하게 백테스트 결과를 시각화하는 함수 (Account 전용)
        - 가격, 실현/미실현 자산, 계좌 추가/정리 등 시각화
        - 주요 변수는 Account 클래스의 속성에 맞게 수정
        """
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
        import numpy as np

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 1. 가격 (왼쪽 Y축)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['close'],
            name='가격', line=dict(color='black', width=1.5)
        ), secondary_y=False)

        # SMA 추가 (164, 720)
        sma_colors = ['#FF9800', '#D32F2F']

        fig.add_trace(go.Scatter(
            x=df.index, y=self.df['sma_long'],
            name='SMA_long(720)', line=dict(color=sma_colors[0], width=1.5)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df.index, y=self.df['sma_short'],
            name='SMA_short(24)', line=dict(color=sma_colors[1], width=1.5)
        ), secondary_y=False)

        # 매수/매도 시그널
        buy_index_long = [i for i in range(1, len(self.long_account.buy_amounts)) if self.long_account.buy_amounts[i] > 0]
        buy_index_short = [i for i in range(1, len(self.short_account.buy_amounts)) if self.short_account.buy_amounts[i] > 0]

        fig.add_trace(go.Scatter(
            x=df.index[buy_index_long], y=df['close'].iloc[buy_index_long],
            mode='markers', marker=dict(symbol='triangle-up', color='green', size=10),
            name='매수(롱)'
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df.index[buy_index_short], y=df['close'].iloc[buy_index_short],
            mode='markers', marker=dict(symbol='triangle-up', color='purple', size=10),
            name='매수(숏)'
        ), secondary_y=False)

        sell_index_long = [i for i in range(1, len(self.long_account.sell_amounts)) if self.long_account.sell_amounts[i] > 0]
        sell_index_short = [i for i in range(1, len(self.short_account.sell_amounts)) if self.short_account.sell_amounts[i] > 0]

        fig.add_trace(go.Scatter(
            x=df.index[sell_index_long], y=df['close'].iloc[sell_index_long],
            mode='markers', marker=dict(symbol='triangle-down', color='blue', size=10),
            name='매도(롱)'
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df.index[sell_index_short], y=df['close'].iloc[sell_index_short],
            mode='markers', marker=dict(symbol='triangle-down', color='red', size=10),
            name='매도(숏)'
        ), secondary_y=False)

        # 2. 실현 자산(영역), 평가 자산(라인), 초기 자본금(라인) (오른쪽 Y축)
        fig.add_trace(go.Scatter(
            x=df.index, y=self.total_balances,
            name='실현 자산(영역)', line=dict(color='#56B4E9', width=2),
            fill='tozeroy', opacity=0.18
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=df.index, y=self.total_asset_values,
            name='평가 자산', line=dict(color='orange', dash='dot', width=1.8)
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=df.index, y=[self.initial_asset]*len(df),
            name='초기 자본금', line=dict(color='#009E73', dash='dot', width=1)
        ), secondary_y=True)

        # 3. 마진소진 카운트(별도 보조축, 빨강계열)
        fig.add_trace(go.Scatter(
            x=df.index, y=self.long_account.sub_account_cnts,
            name='롱 계좌수', line=dict(color='#CC0000', dash='dot', width=1),
        ), secondary_y=True)

        fig.add_trace(go.Scatter(
            x=df.index, y=self.short_account.sub_account_cnts,
            name='숏 계좌수', line=dict(color='#0000CC', dash='dot', width=1),
        ), secondary_y=True)

        # 텍스트 박스 생성을 위한 정보
        if not hasattr(self, 'cumulative_return'):
            self.evaluate_performance()

        perf_text = (
            f"누적수익률: {self.cumulative_return*100:.2f}%\n"
            f"최대낙폭(MDD): {self.mdd*100:.2f}%\n"
            f"샤프지수(연환산): {self.sharpe:.3f}\n"
            f"연환산수익률(CAGR): {self.cagr*100:.2f}%\n"
            f"매수횟수: {self.buy_count}  매도횟수: {self.sell_count}\n"
            f"손익거래비: {self.sell_count_profit / (self.sell_count_loss + self.sell_count_profit) *100. :.2f} %, (수익: {self.sell_count_profit}, 손실: {self.sell_count_loss})\n"
        )

        fig.update_layout(
            title={'text': f'Account 백테스트 결과 (인터랙티브)' + (f' | 심볼: {symbol} | 간격: {timeframe} | 레버리지: {self.leverage}' if symbol and timeframe else '') + f' | 초기 자본금: {self.initial_asset} | 단위투자금: {self.buy_unit}', 'font': {'size': 12}},
            xaxis_title='시간',
            yaxis_title='가격',
            legend=dict(orientation='h'),
            hovermode='x unified',
            height=800,
            annotations=[
                dict(
                    x=0.17,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=perf_text,
                    showarrow=False,
                    align="left",
                    bordercolor="rgba(0, 0, 0, 0.8)",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    font=dict(family='Malgun Gothic', size=10)
                )
            ]
        )
        fig.update_yaxes(title_text='가격', secondary_y=False)
        fig.update_yaxes(title_text='자산 / 계좌수', secondary_y=True)

        if bshow:
            fig.show()
        return fig


class AccountManager :
    def __init__(self, symbol, timeframe, total_asset, initial_asset, buy_unit, leverage, target_profit, target_loss = 0):

        self.symbol = symbol
        self.timeframe = timeframe
        self.total_asset = total_asset
        self.initial_asset = initial_asset
        self.buy_unit = buy_unit
        self.leverage = leverage
        self.target_profit = target_profit
        self.target_loss = target_loss

        # 수동 입력 필요한 부분
        # OKX VIP1 기준 Maker fee 0.01% , Taker fee 0.03%  -- 주로 지정가 주문이므로 Maker fee 적용
        self.trading_fee_rate = 0.0001
        # 8시간마다 지불되는 펀딩피 : 실제로는 유동적으로 변동됨     
        self.funding_fee_rate = 0.0001

        # default파일명 생성
        filename = 'cryto_backtest.png'
        str_symbol = self.symbol.replace('/', '')
        split_count = self.initial_asset /  self.buy_unit

        name, ext = os.path.splitext(filename)
        split_part = f'_split{split_count}' if split_count is not None else ''
        self.img_filename = f"{name}_{str_symbol}_{self.timeframe}_lev{self.leverage}{split_part}{ext}"

        # 평가 지표
        self.mdd = 0 

    # 데이터 가져오기 함수
    def fetch_cryto_data(self, start_date, end_date):

        exchange = ccxt.binance()  # Binance 거래소 예시

        since_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        ohlcv = []

        while since_timestamp < end_timestamp:
            ohlcv_batch = exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since_timestamp, limit=1000)
            if len(ohlcv_batch) == 0:
                break
            ohlcv += ohlcv_batch
            since_timestamp = ohlcv_batch[-1][0] + 1

        self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.set_index('timestamp', inplace=True)

        data_size = len(self.df)
        
        #전체 계좌 기준 정보
        # 총 자산 배열 생성 (거래된 총 자산 : 포지션 평가금액 제외)
        self.total_balances = [0] * data_size
        # 총 자산 가치 배열 생성 (포지션 평가금액 포함)
        self.total_asset_values = [0] * data_size
        # 총 매수 규모 배열 생성 (리버리지 적용)
        self.total_amounts = [0] * data_size
        # 총 원금 규모 배열 생성 (리버리지 미적용)
        self.total_margins = [0] * data_size

        self.sub_account_cnts = [0] * data_size            # 추가 계좌 수

        # 봉기준으로 현재 계좌 정보
        # 매수 금액 배열 생성
        self.buy_amount = [0] * data_size
        self.sell_amount = [0] * data_size
        
        self.amounts = [0] * data_size
        self.margins = [0] * data_size
        self.positions = [0] * data_size
        self.balances = [0] * data_size

        # 현계좌 관련  계산 항목
        self.returns = [0] * data_size        
        self.avg_buy_prices = [0] * data_size

        self.funding_fees = [0] * data_size # 수수료 배열


        self.total_asset_values[0] = self.total_asset
        self.total_balances[0] = self.total_asset
        self.sub_account_cnts[0] = 0

        # 현재 계좌 관리 변수
        self.data_size = data_size

        # 장기 이평선 - 월간 (24*30 = 720)
        self.df['sma_long'] = self.add_moving_average(column='close', window=720, ma_type='SMA')
        # 단기 이평선 - 일간간
        self.df['sma_short'] = self.add_moving_average(column='close', window=24, ma_type='SMA')

        # 계좌 정의        
        self.long_account = Account(self.initial_asset, self.buy_unit, self.leverage, is_long=True, max_account = 5)
        self.long_account.set_account(self.df,self.target_profit-1, 1-self.target_loss, self.trading_fee_rate, self.funding_fee_rate )

        self.long_account.set_filename(self.img_filename)
        
        # short 용 계좌        
        self.short_account = Account(self.initial_asset, self.buy_unit, self.leverage, is_long=False, max_account = 1)
        self.short_account.set_account(self.df, self.target_profit-1, 1-self.target_loss, self.trading_fee_rate, -self.funding_fee_rate/2. )

        self.short_account.set_filename(self.img_filename)

        # 초기 데이터 가져오기
        self.fill_sma_data(720, start_date)
        
    def fill_sma_data(self, sma_long, start_date):

        exchange = ccxt.binance()  # Binance 거래소 예시

        # 초기 데이터 가져오기
        since_timestamp = int((start_date.timestamp() -sma_long*60*60)* 1000)
        end_timestamp = int((start_date.timestamp() + (sma_long + 1)*60*60)* 1000)

        ohlcv = []

        while since_timestamp < end_timestamp:
            ohlcv_batch = exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since_timestamp, limit=1000)
            if len(ohlcv_batch) == 0:
                break
            ohlcv += ohlcv_batch
            since_timestamp = ohlcv_batch[-1][0] + 1

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)


        sma_long_data  = df['close'].rolling(window=720).mean()
        sma_short_data = df['close'].rolling(window=24).mean()
        # 단기 이평선 - 일간간
        start_index = 0
        for i in range(len(df)):
            if(self.df.index[0] == df.index[i]):
                start_index = i
                break

        for i in range(0, len(sma_long_data)):            
            index_value = self.df.index[i]
            if pd.isna (self.df['sma_long'].iloc[i]):
                self.df.loc[index_value, 'sma_long'] = sma_long_data.iloc[i + start_index]
                if pd.isna (self.df['sma_short'].iloc[i]):
                    self.df.loc[index_value, 'sma_short'] = sma_short_data.iloc[i + start_index]
            else:
                break
                

    def simulate_trading(self):        

        # 초기값 정리
        self.total_balances[0] =  self.total_asset
        self.total_asset_values[0] = self.total_asset

        # 총 매수 규모 배열 생성 (리버리지 적용)
        self.total_amounts[0] = 0
        # 총 원금 규모 배열 생성 (리버리지 미적용)
        self.total_margins[0] = 0

        for i in range(1, self.data_size):            
            # 기본적으로 이전 상태 유지
            
            self.total_amounts[i] = self.total_amounts[i-1]
            self.total_margins[i] = self.total_margins[i-1]
            self.total_balances[i] = self.total_balances[i-1]
            self.total_asset_values[i] = self.total_asset_values[i-1]

            if self.positions[i] > 0:
                self.returns[i] = self.returns[i-1] 
            else:
                self.returns[i] = 0
        
            self.long_account.simulate_iteration(i, self.total_balances[i], self.total_asset_values[i], self.total_margins[i])
            self.short_account.simulate_iteration(i, self.total_balances[i], self.total_asset_values[i], self.total_margins[i])

            
            self.total_amounts[i] = self.long_account.total_amounts[i] + self.short_account.total_amounts[i]
            self.total_margins[i] = self.long_account.total_margins[i] + self.short_account.total_margins[i]
            self.total_balances[i] = self.total_asset + self.long_account.balances[i] + self.short_account.balances[i]
            self.total_asset_values[i] = self.total_balances[i] + self.long_account.get_total_pnl(i) + self.short_account.get_total_pnl(i)
            

    def evaluate_performance(self, bshow=True):
        """
        시뮬레이션 결과를 바탕으로 주요 성과 지표를 계산/출력
        - 누적 수익률, 최대낙폭(MDD), 샤프지수, 연환산수익률, 매수/매도 수, 하위계좌 생성/정리 수
        - 타이틀에 초기 자본금, 계좌자본금, 단위투자금 표시
        """
        # 누적 수익률
        initial = self.total_asset_values[0]
        final = self.total_asset_values[-1]
        self.cumulative_return = (final - initial) / initial

        # 최대 낙폭(MDD)
        self.mdd = self.calculate_mdd()

        # 수익률 시계열 (일별)
        asset_series = pd.Series(self.total_asset_values, index=self.df.index)
        daily_return = asset_series.pct_change().fillna(0)

        # 샤프지수 (무위험수익률 0 가정, 연환산)
        self.sharpe = (daily_return.mean() / daily_return.std()) * np.sqrt(365*24) if daily_return.std() > 0 else np.nan

        # 연환산수익률 (CAGR)
        days = (self.df.index[-1] - self.df.index[0]).days
        years = days / 365.0 if days > 0 else 1
        self.cagr = (final / initial) ** (1/years) - 1 if years > 0 else np.nan

        # 매수/매도 수
        self.buy_count = sum(1 for v in self.long_account.buy_amounts if v > 0) + sum(1 for v in self.short_account.buy_amounts if v > 0)
        self.sell_count = sum(1 for v in self.long_account.sell_amounts if v > 0) + sum(1 for v in self.short_account.sell_amounts if v > 0)
        
        # 매도시 수익/손실 건수 집계
        self.sell_count_profit = sum(1 for i in range(1, len(self.long_account.sell_amounts)) if self.long_account.sell_amounts[i]>0 and self.long_account.returns[i] > 0)
        self.sell_count_profit += sum(1 for i in range(1, len(self.short_account.sell_amounts)) if self.short_account.sell_amounts[i]>0 and self.short_account.returns[i] > 0)
        
        self.sell_count_loss = sum(1 for i in range(1, len(self.long_account.sell_amounts)) if self.long_account.sell_amounts[i] and self.long_account.returns[i] < 0)
        self.sell_count_loss += sum(1 for i in range(1, len(self.short_account.sell_amounts)) if self.short_account.sell_amounts[i] and self.short_account.returns[i] < 0)

        # 하위계좌 생성/정리 수
        self.sub_account_cnts = [self.long_account.sub_account_cnts[i] + self.short_account.sub_account_cnts[i] for i in range(0, len(self.df))]
        
        # 정합성 보정: 생성수 - 정리수 = 현재 계좌수
        current_cnt = self.sub_account_cnts[-1]

        if bshow:
            print("==== 시뮬레이션 성과 평가 ====")
            print(f"초기 자본금: {self.total_asset} | 계좌자본금: {self.initial_asset} | 단위투자금: {self.buy_unit}")
            print(f"누적 수익률: {self.cumulative_return*100:.2f}%")
            print(f"최대 낙폭(MDD): {self.mdd*100:.2f}%")
            print(f"샤프지수(연환산): {self.sharpe:.3f}")
            print(f"연환산수익률(CAGR): {self.cagr*100:.2f}%")
            print(f"매수 횟수: {self.buy_count}")
            print(f"매도 횟수: {self.sell_count}")
            print(f"손익횟수 비율: {self.sell_count_profit / (self.sell_count_loss + self.sell_count_profit) if self.sell_count_loss + self.sell_count_profit > 0 else 1}, (수익: {self.sell_count_profit}, 손실: {self.sell_count_loss})")
            print("==========================")

    def calculate_mdd(self, asset_curve=None):
        """
        최대 낙폭(MDD, Maximum Drawdown) 계산 함수
        asset_curve: 평가자산 곡선 (기본값: self.total_asset_values)
        return: 최대 낙폭 (음수, 예: -0.35)
        """
        if asset_curve is None:
            asset_curve = self.total_asset_values
        cummax = np.maximum.accumulate(asset_curve)
        drawdown = (asset_curve - cummax) / cummax
        mdd = drawdown.min()
        return mdd



    def add_moving_average(self, column='close', window=5, ma_type='SMA'):
        if ma_type == 'SMA':
            return self.df[column].rolling(window=window).mean()
        elif ma_type == 'EMA':
            return self.df[column].ewm(span=window, adjust=False).mean()
        

    def save_results(self):
        results_df = pd.DataFrame({
            'Date': self.df.index.date,
            'Close': self.df['close'],
            'Amount': self.total_amounts,
            'margin_used': self.total_margins,
            'account_depth': self.sub_account_cnts,
            'Long_Balance': self.long_account.balances,
            'Long_Asset_values': self.long_account.total_asset_values,
            'Short_Balance': self.short_account.balances,
            'Short_Asset_values': self.short_account.total_asset_values,
            'Total_Balance': self.total_balances,
            'Total_Asset_values': self.total_asset_values,
        })

        results_df.to_excel(self.img_filename.replace('.png', '.xlsx'), index=False)

        self.long_account.save_results()
        self.short_account.save_results()


    
    def plot_results(self, bshow=True):    
        """
        계좌별 백테스트 결과를 명확하게 시각화하는 함수 (Account 전용)
        - 가격, 실현/미실현 자산, 계좌 추가/정리 등 시각화
        - 주요 변수는 Account 클래스의 속성에 맞게 수정
        """
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # 1. 왼쪽 Y축: 가격
        ax1.set_xlabel('시간')
        ax1.set_ylabel('가격', color='black')
        price_line, = ax1.plot(self.df.index, self.df['close'], label='가격', color='black', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)

        # SMA 추가 (164, 720)
        
        sma_colors = ['#FF9800', '#D32F2F']
        
        ax1.plot(self.df.index, self.df['sma_long'], label='SMA_long(720)', color=sma_colors[0], linewidth=1.5)
        ax1.plot(self.df.index, self.df['sma_short'], label='SMA_short(24)', color=sma_colors[1], linewidth=1.5)    
        
        # 매수
        buy_index_long = [i for i in range(1, len(self.long_account.buy_amounts)) if self.long_account.buy_amounts[i] > 0 ]
        buy_index_short = [i for i in range(1, len(self.short_account.buy_amounts)) if self.short_account.buy_amounts[i] > 0]

        ax1.scatter(self.df.index[buy_index_long], self.df['close'].iloc[buy_index_long], marker='^', color='green', label='매수(롱)', s=50, zorder=5)
        ax1.scatter(self.df.index[buy_index_short], self.df['close'].iloc[buy_index_short], marker='^', color='purple', label='매수(숏)', s=50, zorder=5)

        # 매도(수익/손실)
        sell_index_long = [i for i in range(1, len(self.long_account.sell_amounts)) if self.long_account.sell_amounts[i] > 0 ]
        sell_index_short = [i for i in range(1, len(self.short_account.sell_amounts)) if self.short_account.sell_amounts[i] > 0]
        ax1.scatter(self.df.index[sell_index_long], self.df['close'].iloc[sell_index_long], marker='v', color='blue', label='매도(롱)', s=70, zorder=5)
        ax1.scatter(self.df.index[sell_index_short], self.df['close'].iloc[sell_index_short], marker='v', color='red', label='매도(숏)', s=70, zorder=5)

        # 2. 오른쪽 Y축: 실현/미실현 자산 
        ax2 = ax1.twinx()
        ax2.set_ylabel('자산', color='#0072B2')
        min_balance = min(min(self.total_balances), min(self.total_asset_values))
        max_balance = max(max(self.total_balances), max(self.total_asset_values))
        y_margin = (max_balance - min_balance) * 0.05 if max_balance > min_balance else 1
        y_min = min_balance - y_margin
        y_max = max_balance + y_margin
        ax2.fill_between(self.df.index, self.total_balances, color='#56B4E9', alpha=0.18, label='실현 자산(영역)')
        asset_line = ax2.plot(self.df.index, self.total_asset_values, label='평가 자산', color='orange', linestyle='-.', linewidth=1.8)
        ax2.axhline(y=self.initial_asset, color='#009E73', linestyle=':', alpha=0.7, label='초기 자본금')
        ax2.tick_params(axis='y', labelcolor='#0072B2')
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax2.set_ylim(y_min, y_max)

        # 3. 마진소진 카운트(별도 보조축, 빨강계열)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_ylabel('추가 계좌수', color='#CC0000')
        ax3.plot(self.df.index, self.long_account.sub_account_cnts, label='롱 계좌수', color='#CC0000', linestyle='-.', linewidth=1.0)
        ax3.plot(self.df.index, self.short_account.sub_account_cnts, label='숏 계좌수', color='#0000CC', linestyle='-.', linewidth=1.0)
        ax3.tick_params(axis='y', labelcolor='#CC0000')
        ax3.set_ylim(bottom=0)        

        # 5. 범례 정리 (중복 제거)
        handles, labels = [], []
        for ax in [ax1, ax2, ax3]:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
        ax1.legend(handles, labels, loc='upper left', fontsize=11)

        # 6. 제목, 레이아웃
        title1 = f'Account 백테스트 결과'
        if self.symbol and self.timeframe:
            title1 += f' [심볼: {self.symbol} | 간격: {self.timeframe} | 레버리지: {self.leverage}]'
        title2 = f'초기 자본금: {self.initial_asset} | 단위투자금: {self.buy_unit}'
        plt.title(f'{title1}\n{title2}', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()


        # --- 성과 요약 텍스트 박스 ---
        # 먼저 성과 계산 (속성 없으면 계산)
        if not hasattr(self, 'cumulative_return'):
            self.evaluate_performance()
        perf_text = (
            f"누적수익률: {self.cumulative_return*100:.2f}%\n"
            f"최대낙폭(MDD): {self.mdd*100:.2f}%\n"
            f"샤프지수(연환산): {self.sharpe:.3f}\n"
            f"연환산수익률(CAGR): {self.cagr*100:.2f}%\n"
            f"매수횟수: {self.buy_count}  매도횟수: {self.sell_count}\n"
            f"손익거래비: {self.sell_count_profit / (self.sell_count_loss + self.sell_count_profit) *100. :.2f} %, (수익: {self.sell_count_profit}, 손실: {self.sell_count_loss})\n"
        )

        # 범례(upper left) 바로 오른쪽에, 각 항목별로 세로로 출력
        ax1.text(0.17, 0.98, perf_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontfamily='Malgun Gothic')

        # 그래프를 파일로 저장
        # plt.savefig(self.img_filename)
        # print(f"그래프를 파일로 저장: {self.img_filename}")
        if bshow:
            plt.show()
        # plt.show()
        # plt.close()

        return fig


    def plot_results_interactive(self, df, symbol=None, timeframe=None, bshow=False):
        """
        Plotly를 사용해 웹에서 인터랙티브하게 백테스트 결과를 시각화하는 함수 (Account 전용)
        - 가격, 실현/미실현 자산, 계좌 추가/정리 등 시각화
        - 주요 변수는 Account 클래스의 속성에 맞게 수정
        """
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
        import numpy as np

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 1. 가격 (왼쪽 Y축)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['close'],
            name='가격', line=dict(color='black', width=1.5)
        ), secondary_y=False)

        # SMA 추가 (164, 720)
        sma_colors = ['#FF9800', '#D32F2F']

        fig.add_trace(go.Scatter(
            x=df.index, y=self.df['sma_long'],
            name='SMA_long(720)', line=dict(color=sma_colors[0], width=1.5)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df.index, y=self.df['sma_short'],
            name='SMA_short(24)', line=dict(color=sma_colors[1], width=1.5)
        ), secondary_y=False)

        # 매수/매도 시그널
        buy_index_long = [i for i in range(1, len(self.long_account.buy_amounts)) if self.long_account.buy_amounts[i] > 0]
        buy_index_short = [i for i in range(1, len(self.short_account.buy_amounts)) if self.short_account.buy_amounts[i] > 0]

        fig.add_trace(go.Scatter(
            x=df.index[buy_index_long], y=df['close'].iloc[buy_index_long],
            mode='markers', marker=dict(symbol='triangle-up', color='green', size=10),
            name='매수(롱)'
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df.index[buy_index_short], y=df['close'].iloc[buy_index_short],
            mode='markers', marker=dict(symbol='triangle-up', color='purple', size=10),
            name='매수(숏)'
        ), secondary_y=False)

        sell_index_long = [i for i in range(1, len(self.long_account.sell_amounts)) if self.long_account.sell_amounts[i] > 0]
        sell_index_short = [i for i in range(1, len(self.short_account.sell_amounts)) if self.short_account.sell_amounts[i] > 0]

        fig.add_trace(go.Scatter(
            x=df.index[sell_index_long], y=df['close'].iloc[sell_index_long],
            mode='markers', marker=dict(symbol='triangle-down', color='blue', size=10),
            name='매도(롱)'
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df.index[sell_index_short], y=df['close'].iloc[sell_index_short],
            mode='markers', marker=dict(symbol='triangle-down', color='red', size=10),
            name='매도(숏)'
        ), secondary_y=False)

        # 2. 실현 자산(영역), 평가 자산(라인), 초기 자본금(라인) (오른쪽 Y축)
        fig.add_trace(go.Scatter(
            x=df.index, y=self.total_balances,
            name='실현 자산(영역)', line=dict(color='#56B4E9', width=2),
            fill='tozeroy', opacity=0.18
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=df.index, y=self.total_asset_values,
            name='평가 자산', line=dict(color='orange', dash='dot', width=1.8)
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=df.index, y=[self.initial_asset]*len(df),
            name='초기 자본금', line=dict(color='#009E73', dash='dot', width=1)
        ), secondary_y=True)

        # 3. 마진소진 카운트(별도 보조축, 빨강계열)
        fig.add_trace(go.Scatter(
            x=df.index, y=self.long_account.sub_account_cnts,
            name='롱 계좌수', line=dict(color='#CC0000', dash='dot', width=1),
        ), secondary_y=True)

        fig.add_trace(go.Scatter(
            x=df.index, y=self.short_account.sub_account_cnts,
            name='숏 계좌수', line=dict(color='#0000CC', dash='dot', width=1),
        ), secondary_y=True)

        # 텍스트 박스 생성을 위한 정보
        if not hasattr(self, 'cumulative_return'):
            self.evaluate_performance()

        perf_text = (
            f"누적수익률: {self.cumulative_return*100:.2f}%\n"
            f"최대낙폭(MDD): {self.mdd*100:.2f}%\n"
            f"샤프지수(연환산): {self.sharpe:.3f}\n"
            f"연환산수익률(CAGR): {self.cagr*100:.2f}%\n"
            f"매수횟수: {self.buy_count}  매도횟수: {self.sell_count}\n"
            f"손익거래비: {self.sell_count_profit / (self.sell_count_loss + self.sell_count_profit) *100. :.2f} %, (수익: {self.sell_count_profit}, 손실: {self.sell_count_loss})\n"
        )

        fig.update_layout(
            title={'text': f'Account 백테스트 결과 (인터랙티브)' + (f' | 심볼: {symbol} | 간격: {timeframe} | 레버리지: {self.leverage}' if symbol and timeframe else '') + f' | 초기 자본금: {self.initial_asset} | 단위투자금: {self.buy_unit}', 'font': {'size': 12}},
            xaxis_title='시간',
            yaxis_title='가격',
            legend=dict(orientation='h'),
            hovermode='x unified',
            height=800,
            annotations=[
                dict(
                    x=0.17,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=perf_text,
                    showarrow=False,
                    align="left",
                    bordercolor="rgba(0, 0, 0, 0.8)",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    font=dict(family='Malgun Gothic', size=10)
                )
            ]
        )
        fig.update_yaxes(title_text='가격', secondary_y=False)
        fig.update_yaxes(title_text='자산 / 계좌수', secondary_y=True)

        if bshow:
            fig.show()
        return fig



class StrategyOptimizer:
    def __init__(self, symbol='BTC/USDT', timeframe='1h', total_asset=10000, initial_asset=2000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.total_asset = total_asset
        self.initial_asset = initial_asset
        
        # 유전 알고리즘 설정
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()


        # buy_unit 값을 지정된 값들로만 제한
        self.buy_unit_choices = [50, 100, 300, 500, 1000, 2000]
        
        # 매개변수 범위 설정
        self.toolbox.register("buy_unit", random.choice, self.buy_unit_choices)
        self.toolbox.register("leverage", random.randint, 1, 5)
        self.toolbox.register("target_profit", random.uniform, 1.01, 1.10)
        self.toolbox.register("target_loss", random.uniform, 0.0, 0.99)
        
        # 개체 생성
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                            (self.toolbox.buy_unit, self.toolbox.leverage,
                             self.toolbox.target_profit, self.toolbox.target_loss), n=1)
        
        # 인구 생성
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 유전 연산자 설정
        self.toolbox.register("evaluate", self.evaluate_strategy)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def mutate_individual(self, individual):
        """개별 매개변수에 대한 돌연변이 함수"""
        for i in range(len(individual)):
            if random.random() < 0.2:  # 20% 확률로 돌연변이 발생
                if i == 0:  # buy_unit
                    individual[i] = random.choice(self.buy_unit_choices)
                elif i == 1:  # leverage
                    individual[i] = random.randint(1, 5)
                elif i == 2:  # target_profit
                    individual[i] = random.uniform(1.01, 1.10)
                elif i == 3:  # target_loss
                    individual[i] = random.uniform(0.0, 0.99)
        return individual,

    def evaluate_strategy(self, individual):
        """전략 평가 함수"""
        buy_unit, leverage, target_profit, target_loss = individual
        
        try:
            # AccountManager 인스턴스 생성
            account_manager = AccountManager(
                symbol=self.symbol,
                timeframe=self.timeframe,
                total_asset=self.total_asset,
                initial_asset=self.initial_asset,
                buy_unit=buy_unit,
                leverage=leverage,
                target_profit=target_profit,
                target_loss=target_loss
            )
            
            # 데이터 가져오기 및 시뮬레이션
            start_date = datetime(2019, 1, 1)  # 최적화 기간 설정
            end_date = datetime.now() - timedelta(1)
            
            account_manager.fetch_cryto_data(start_date, end_date)
            account_manager.simulate_trading()
            account_manager.evaluate_performance(False)  # bshow=False로 설정

            # 결과 출력
            print(f"\n현재 파라미터 평가:")
            print(f"매수단위: {buy_unit:.2f}, 레버리지: {leverage:.2f}")
            print(f"익절: {target_profit:.3f}, 손절: {target_loss:.3f}")
            print(f"샤프비율: {account_manager.sharpe:.3f}")
            print(f"년평균수익율: {account_manager.cagr*100:.2f}%")
            print(f"MDD: {account_manager.mdd*100:.2f}%")
            print("-" * 50)
            
            # 샤프 지수가 유효한 값인지 확인
            if np.isnan(account_manager.sharpe):
                return (-np.inf,)  # 유효하지 않은 경우 최소값 반환
                
            # 수익률이 음수인 경우 페널티 부여
            if account_manager.cumulative_return < 0:
                return (account_manager.sharpe * 0.5,)
                
            return (account_manager.sharpe,)
            
        except Exception as e:
            print(f"Error in evaluate_strategy: {str(e)}")
            return (-np.inf,)  # 에러 발생 시 최소값 반환
    

    def optimize(self, population_size=50, generations=30, cpu_count=None):
        """최적화 실행"""
        if cpu_count is None:
            cpu_count = multiprocessing.cpu_count() - 1
        
        # 멀티프로세싱 풀 제거
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        
        # 통계 객체 설정
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # 유전 알고리즘 실행
        try :
            pop, logbook = algorithms.eaSimple(pop, self.toolbox,
                                        cxpb=0.7,  # 교차 확률
                                        mutpb=0.3,  # 돌연변이 확률
                                        ngen=generations,
                                        stats=stats,
                                        halloffame=hof,
                                        verbose=True)
            
            if len(hof) > 0:
                return hof[0], logbook
            else:
                print("최적화 실패: 적합한 해를 찾지 못했습니다.")
                return None, None
                
        except Exception as e :
            print(f"최적화 중 오류 발생: {str(e)}")
            return None, None
        
        
def main_optimize():
    # 최적화 실행
    optimizer = StrategyOptimizer(
        symbol='BTC/USDT',
        timeframe='1h',
        total_asset=10000,
        initial_asset=2000
    )
    
    print("최적화 시작...")
    try:
        result = optimizer.optimize(population_size=50, generations=30)
        if result is None or result[0] is None:
            print("최적화 실패: 적절한 파라미터를 찾지 못했습니다.")
            return
            
        best_params, stats = result

        print("\n최적화된 매개변수:")
        print(f"매수 단위: {best_params[0]:.2f}")
        print(f"레버리지: {best_params[1]:.2f}")
        print(f"익절 목표: {best_params[2]:.3f}")
        print(f"손절 목표: {best_params[3]:.3f}")
        
        # 최적화된 매개변수로 백테스트 실행
        account_manager = AccountManager(
            symbol='BTC/USDT',
            timeframe='1h',
            total_asset=10000,
            initial_asset=2000,
            buy_unit=best_params[0],
            leverage=best_params[1],
            target_profit=best_params[2],
            target_loss=best_params[3]
        )
        
        # 최종 결과 저장
        start_date = datetime(2019, 1, 1)
        end_date = datetime.now() - timedelta(1)
        
        account_manager.fetch_cryto_data(start_date, end_date)
        account_manager.simulate_trading()
        account_manager.evaluate_performance()
        
        # 결과 시각화 및 저장
        fig = account_manager.plot_results()
        fig.savefig('optimized_strategy_results.png')
        account_manager.save_results()
        
    except Exception as e:
        print(f"최적화 과정에서 오류 발생: {str(e)}")

def main_single():
    # 파라미터 설정 : cursor 사용시 변수 변경시 추가 코드 발생하여 수정
    symbol = 'BTC/USDT'
    timeframe = '1h'
    total_asset = 10000
    initial_asset = 2000
    buy_unit = 100
    leverage = 5
    target_profit = 1.02
    target_loss = 0.

    # account manager 생성
    account_manager = AccountManager (symbol=symbol, timeframe=timeframe, total_asset=total_asset, initial_asset=initial_asset, buy_unit=buy_unit, leverage=leverage, target_profit=target_profit, target_loss=target_loss)

    # 데이터 가져오기
    year = 2019
    start_date = datetime(year, 1, 1)
    end_date = datetime.now() - timedelta(1)

    account_manager.fetch_cryto_data(start_date, end_date)

    account_manager.simulate_trading()
    
    # 성과 평가
    account_manager.evaluate_performance()

    # 결과시각화
    fig = account_manager.plot_results()

    name, ext = os.path.splitext(account_manager.img_filename)

    #fig.savefig(f"{name}_{year}{ext}")

        
    account_manager.plot_results_interactive(account_manager.df, account_manager.symbol, account_manager.timeframe, bshow=False)

    account_manager.save_results()

def main_evaluate():
   # 파라미터 설정 : cursor 사용시 변수 변경시 추가 코드 발생하여 수정
    symbol = 'BTC/USDT'
    timeframe = '1h'
    total_asset = 10000
    initial_asset = 2000
    buy_unit = 100
    leverage = 5
    target_profit = 1.02
    target_loss = 0.

    # account manager 생성
    account_manager = AccountManager (symbol=symbol, timeframe=timeframe, total_asset=total_asset, initial_asset=initial_asset, buy_unit=buy_unit, leverage=leverage, target_profit=target_profit, target_loss=target_loss)

    # 데이터 가져오기
    """
    # 데이터 가져오기
    year = 2025
    start_date = datetime(year, 1, 1)
    end_date = datetime.now() - timedelta(1)

    account_manager.fetch_cryto_data(start_date, end_date)

    account_manager.simulate_trading()
    
    # 성과 평가
    account_manager.evaluate_performance()

    # 결과시각화
    fig = account_manager.plot_results()

    name, ext = os.path.splitext(account_manager.img_filename)

    #fig.savefig(f"{name}_{year}{ext}")
    
    print(f"--- 완료 ---")
    """
    name, ext = os.path.splitext(account_manager.img_filename)
    
    for year in range(2025, 2018, -1):
        start_date = datetime(year, 1, 1)
        end_date = datetime.now() - timedelta(1)

        account_manager.fetch_cryto_data(start_date, end_date)

        account_manager.simulate_trading()
    
        # 성과 평가
        account_manager.evaluate_performance()

        # 결과시각화
        fig = account_manager.plot_results()

        fig.savefig(f"{name}_{year}{ext}")

        print(f"--- {year}년 완료 ---")


    # account_manager.plot_results_interactive(account_manager.df, account_manager.symbol, account_manager.timeframe, bshow=False)

    # account_manager.save_results()

    print(f"### 완료 ###")

def main_ccx():
    # CCXT OKX 인스턴스 생성 (스왑 시장을 기본으로 설정)
        
    exchange = ccxt.okx({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'password': API_PASSPHRASE,
        'options': {
            'defaultType': 'SWAP',  # 중요: 'swap' (무기한 계약) 또는 'future' (만기 선물)
            # 'defaultSubType': 'linear', # USDT 기반 스왑이 일반적이므로 명시 안해도 될 수 있음
        },
        # 'verbose': True, # API 요청/응답 로깅 (디버깅 시 유용)
    })

    exchange.set_sandbox_mode(True) 

    # 샌드박스/테스트넷 사용 (실제 거래 전 필수!)
    # exchange.set_sandbox_mode(True) # OKX는 CCXT에서 set_sandbox_mode를 직접 지원하지 않을 수 있음.
    # OKX API 문서에서 테스트넷 엔드포인트를 확인하고,
    # exchange.urls['api'] = 'OKX_TESTNET_API_ENDPOINT' 와 같이 직접 설정해야 할 수 있습니다.
    # OKX는 데모 트레이딩(모의투자) 환경을 제공하며, API 키도 별도로 발급받아야 합니다.
    # https://www.okx.com/account/subaccount-api (데모 트레이딩 API 키 생성 가능)
    # exchange.options['brokerId'] = 'YOUR_DEMO_BROKER_ID' # 데모 트레이딩시 필요할 수 있음 (OKX 문서 확인)

    try:
        # 1. 시장 정보 로드 (선물/스왑 시장 포함)        
        markets = exchange.load_markets()
        print("OKX 시장 정보 로드 완료.")

        # USDT 기반 BTC 무기한 스왑 심볼 찾기 (예시)
        # 실제 심볼은 markets 객체를 탐색하여 확인해야 합니다.
        # OKX의 경우 BTC-USDT-SWAP 또는 CCXT 표준 형식인 BTC/USDT:USDT 일 수 있습니다.
        target_symbol = None
        for symbol, market_info in markets.items():
            if market_info['base'] == 'BTC' and market_info['quote'] == 'USDT' and market_info['type'] == 'swap' and market_info['linear']:
                target_symbol = symbol
                print(f"타겟 심볼 발견: {target_symbol} (ID: {market_info['id']})")
                break

        if not target_symbol:
            print("BTC/USDT 스왑 시장을 찾을 수 없습니다.")
            exit()

        # 2. 선물/스왑 계정 잔고 조회
        # params로 type을 지정해주거나, exchange 생성 시 defaultType을 지정해야 함
        balance = exchange.fetch_balance() # defaultType이 'swap'으로 설정되어 있으므로 스왑 잔고 조회
        # balance = exchange.fetch_balance({'type': 'swap'}) # 명시적으로 지정도 가능
        usdt_balance = balance['USDT']['free'] if 'USDT' in balance and 'free' in balance['USDT'] else 0
        print(f"스왑 계정 USDT 잔고: {usdt_balance}")

        # 3. 레버리지 설정 (예: BTC/USDT 스왑, 10배)
        # OKX는 심볼별, 포지션별(격리/교차), 매수/매도별 레버리지 설정이 가능할 수 있음.
        # CCXT의 set_leverage가 모든 경우를 커버하지 못할 수 있으므로 params를 활용하거나 API문서 확인.
        try:
            leverage_response = exchange.set_leverage(10, target_symbol, params={'marginMode': 'isolated'}) # 격리 마진 예시
            # leverage_response = exchange.set_leverage(10, target_symbol, params={'mgnMode': 'isolated', 'posSide': 'long'}) # 롱 포지션에 대한 격리 마진 레버리지
            print(f"레버리지 설정 응답: {leverage_response}")
        except Exception as e:
            print(f"레버리지 설정 오류: {e}")
            print("참고: OKX는 UI에서 먼저 '선물 계좌 모드' (단일 통화 마진, 다중 통화 마진 등) 및 '포지션 모드' (단방향/양방향)를 설정해야 할 수 있습니다.")
            print("API로 레버리지 설정 전 계정 설정 및 권한을 확인하세요.")


        # 4. 포지션 조회
        positions = exchange.fetch_positions([target_symbol])
        print(f"\n현재 {target_symbol} 포지션:")
        if positions:
            for position in positions:
                # CCXT 버전 및 거래소에 따라 포지션 객체 구조가 다를 수 있음
                print(f"  심볼: {position.get('symbol', 'N/A')}, 수량: {position.get('contracts', position.get('contractSize', 'N/A'))}, 진입가격: {position.get('entryPrice', 'N/A')}, 미실현손익: {position.get('unrealizedPnl', 'N/A')}, 레버리지: {position.get('leverage', 'N/A')}")
        else:
            print("  현재 보유 포지션 없음")


        # 5. 주문 예시 (실제 주문이므로 주의! 테스트넷에서 실행 권장)
        # 지정가 롱(매수) 주문
        order_type = 'limit'
        side = 'buy' # 'buy' (long), 'sell' (short)
        amount = 0.001  # BTC 단위 (계약 수량)
        # 현재가보다 낮은 가격으로 지정가 매수 시도 (예시)
        ticker = exchange.fetch_ticker(target_symbol)
        price = ticker['bid'] * 0.98 # 현재 매수 호가보다 2% 낮은 가격 (체결 안될 수 있음)

        print(f"\n테스트 주문 시도: {target_symbol}, {side}, 수량: {amount}, 가격: {price}")

        # ### 실제 주문 실행 부분 (매우 주의!) ###
        # # try:
        # #     # 포지션 모드 (헤지 모드/양방향 포지션) 사용 시 'posSide' 파라미터 필요
        # #     # params = {'posSide': 'long'} # 롱 포지션 진입
        # #     # params = {'posSide': 'short'} # 숏 포지션 진입
        # #     # 단방향 포지션 모드에서는 'posSide' 불필요
        # #     params = {}
        # #     if exchange.options.get('defaultPositionMode') == 'hedged' or exchange.options.get('positionMode') == 'hedged': # 가정
        # #          params['posSide'] = 'long' if side == 'buy' else 'short'

        # #     order = exchange.create_order(target_symbol, order_type, side, amount, price, params)
        # #     print("주문 성공:")
        # #     print(order)
        # # except Exception as e:
        # #     print(f"주문 실패: {e}")
        # print("실제 주문은 주석 처리되어 있습니다. 테스트 후 주의하여 사용하세요.")


    except ccxt.NetworkError as e:
        print(f"네트워크 오류: {e}")
    except ccxt.ExchangeError as e:
        print(f"거래소 오류: {e}")
    except Exception as e:
        print(f"기타 오류: {e}")
        


# 메인 실행 흐름
if __name__ == "__main__":


    main_single()
    # main_evaluate()
    # main_ccx()

