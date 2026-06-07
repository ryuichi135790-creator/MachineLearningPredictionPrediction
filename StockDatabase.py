import sqlite3
import pandas as pd
from tqdm import tqdm # 冒頭でインポート

class StockDatabase:
    """株価の時系列データと学習フラグをローカルのSQLiteデータベースで一元管理するクラス"""
    def __init__(self, db_path="nikkei_stock_data.db"):
        self.db_path = db_path
        self._create_table()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    is_trained INTEGER DEFAULT 0,
                    PRIMARY KEY (ticker, date)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_date 
                ON daily_prices (ticker, date)
            """)
            conn.commit()

    def get_all_cached_data(self, ticker):
        with self._get_connection() as conn:
            query = "SELECT date, open, high, low, close, volume, is_trained FROM daily_prices WHERE ticker = ? ORDER BY date ASC"
            df = pd.read_sql_query(query, conn, params=(ticker,), index_col='date')
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
            return df

    def save_new_data(self, ticker, df_new):
        if df_new.empty:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for date_idx, row in df_new.iterrows():
                date_str = pd.to_datetime(date_idx).tz_localize(None).strftime('%Y-%m-%d')
                cursor.execute("""
                    INSERT OR IGNORE INTO daily_prices 
                    (ticker, date, open, high, low, close, volume, is_trained)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """, (ticker, date_str, float(row['open']), float(row['high']), float(row['low']), float(row['close']), int(row['volume'])))
            conn.commit()

    def save_batch_data(self, df_batch):
        if df_batch.empty:
            return

        # 1. 既存データの取得
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT date, ticker FROM daily_prices")
            existing_records = set(cursor.fetchall()) 

        tickers = df_batch.columns.levels[0].unique()
        payload = []

        # 2. 銘柄ごとの進捗を表示
        print(f"--- データの保存処理を開始: {len(tickers)} 銘柄 ---")
        for ticker in tqdm(tickers, desc="銘柄処理中"):
            ticker_data = df_batch[ticker]

            for date, row in ticker_data.iterrows():
                try:
                    # NaNチェック
                    if row.isnull().any():
                        continue 
                    
                    date_str = str(date.date())
                    
                    # 重複チェック
                    if (date_str, ticker) in existing_records:
                        continue
                    
                    payload.append((
                        ticker, 
                        date_str,
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume']),
                        0
                    ))
                except Exception as e:
                    continue

        # 3. 登録実行
        if payload:
            print(f"--- {len(payload)} 件の新規データをDBへ一括登録中... ---")
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO daily_prices 
                    (ticker, date, open, high, low, close, volume, is_trained)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, payload)
                conn.commit()
            print("登録が完了しました。")
        else:
            print("登録すべき新規データはありませんでした。")

    def update_trained_status(self, ticker, dates):
        if not dates:
            return
        date_strs = [pd.to_datetime(d).tz_localize(None).strftime('%Y-%m-%d') for d in dates]
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(date_strs))
            query = f"UPDATE daily_prices SET is_trained = 1 WHERE ticker = ? AND date IN ({placeholders})"
            cursor.execute(query, [ticker] + date_strs)
            conn.commit()
            