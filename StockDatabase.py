import sqlite3
import pandas as pd

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
        
        df_stacked = df_batch.stack(level=1)
        df_stacked.index.names = ['date', 'ticker']
        df_stacked = df_stacked.reset_index()
        
        df_stacked.columns = [c.lower() for c in df_stacked.columns]
        df_stacked = df_stacked[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
        df_stacked['date'] = pd.to_datetime(df_stacked['date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        df_stacked = df_stacked.dropna(subset=['open', 'close'])
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            payload = [
                (row['ticker'], row['date'], float(row['open']), float(row['high']), float(row['low']), float(row['close']), int(row['volume']))
                for _, row in df_stacked.iterrows()
            ]
            cursor.executemany("""
                INSERT OR IGNORE INTO daily_prices 
                (ticker, date, open, high, low, close, volume, is_trained)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """, payload)
            conn.commit()

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