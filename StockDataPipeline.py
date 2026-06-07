import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from StockDatabase import StockDatabase
from datetime import datetime, timedelta

class StockDataPipeline:
    """DBの学習フラグと連動し、日足の特徴量生成と未学習データの正確な切り分けを行うクラス"""
    def __init__(self, features_list, default_period="20y"):
        self.features_list = features_list
        self.default_period = default_period
        self.db = StockDatabase()

    def fetch_and_transform(self, ticker_symbol):
        """
        初回一括ダウンロードデータを100%信頼する設計。
        DBにデータがあればWeb通信の判定を完全にスキップして、ローカル処理のみをミリ秒単位で実行します。
        """
        try:
            # 1. ローカルDBにあるキャッシュデータを取得
            df_cached = self.db.get_all_cached_data(ticker_symbol)
            
            # 2. 判定ロジックの爆速化 (100行以上あればWeb通信フラグを強制的にFalseにする)
            if not df_cached.empty and len(df_cached) >= 100:
                need_web_fetch = False
            else:
                need_web_fetch = True

            # 3. DBが完全に空の場合のみセーフティとしてyfinanceを叩く
            df_new_delta = pd.DataFrame()
            if need_web_fetch:
                stock = yf.Ticker(ticker_symbol)
                df_web = stock.history(period=self.default_period, interval="1d", actions=False)
                
                if not df_web.empty:
                    if df_web.index.tz is not None:
                        df_web.index = df_web.index.tz_localize(None)
                    df_web.columns = [c.lower() for c in df_web.columns]
                    df_web = df_web[['open', 'high', 'low', 'close', 'volume']]
                    df_raw = df_web
                    df_new_delta = df_web
                else:
                    df_raw = df_cached
            else:
                # 🚀 ネットワークを介さず、100%ローカルDBの高速展開データを利用
                df_raw = df_cached

            if df_raw.empty or len(df_raw) < 60:
                return None

            if not df_new_delta.empty:
                self.db.save_new_data(ticker_symbol, df_new_delta)
            
            df_cached_updated = self.db.get_all_cached_data(ticker_symbol)

            # 4. テクニカル指標の計算
            macd_df = ta.macd(close=df_raw['close'], fast=12, slow=26, signal=9)
            rsi_series = ta.rsi(close=df_raw['close'], length=14)
            adx_df = ta.adx(high=df_raw['high'], low=df_raw['low'], close=df_raw['close'], length=14)
            
            df_calc = pd.concat([df_raw, macd_df, rsi_series, adx_df], axis=1)
            df_calc['SMA_20'] = ta.sma(close=df_raw['close'], length=20)
            df_calc['SMA_50'] = ta.sma(close=df_raw['close'], length=50)
            df_calc['SMA_Gap'] = (df_calc['SMA_20'] - df_calc['SMA_50']) / df_calc['SMA_50'] * 100
            
            df_calc['Next_Day_Return'] = ((df_calc['close'].shift(-1) - df_calc['close']) / df_calc['close']) * 100
            
            df_calc = df_calc.rename(columns={
                'MACD_12_26_9': 'MACD',
                'MACDh_12_26_9': 'MACD_Hist',
                'RSI_14': 'RSI',
                'ADX_14': 'ADX'
            })
            
            final_df = df_calc[self.features_list + ['Next_Day_Return']].dropna()
            if final_df.empty:
                return None

            latest_available_date = df_calc.index[-1]
            X_latest_today = df_calc[self.features_list].loc[latest_available_date].values
            
            # 5. 未学習データの正確なフィルタリング
            final_df['is_trained'] = df_cached_updated['is_trained'].reindex(final_df.index, fill_value=0)
            df_untrained = final_df[final_df['is_trained'] == 0]
            
            if not df_untrained.empty:
                X_past = df_untrained[self.features_list].values
                y_past = df_untrained['Next_Day_Return'].values
                trained_dates = df_untrained.index.tolist()
            else:
                X_past = np.empty((0, len(self.features_list)))
                y_past = np.empty((0,))
                trained_dates = []

            current_rsi = df_calc['RSI'].iloc[-1]
            current_adx = df_calc['ADX'].iloc[-1]

            return X_past, y_past, X_latest_today, current_rsi, current_adx, trained_dates, need_web_fetch

        except Exception as e:
            print(f"💥 [{ticker_symbol}] 処理例外: {e}")
            return None