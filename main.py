import os
import pandas as pd
import numpy as np
import time
import yfinance as yf
import tkinter as tk
import logging
import matplotlib

# Matplotlibのグローバルフォントを強制適用してグラフの文字化けを防ぐ
matplotlib.rcParams['font.family'] = 'MS Gothic'

from NikkeiTickerManager import NikkeiTickerManager
from StockDataPipeline import StockDataPipeline
from IncrementalLearningModel import IncrementalLearningModel
from StockDatabase import StockDatabase
from StockPredictionApp import StockPredictionApp

def main():
    total_start_time = time.time()
    
    FETCH_PERIOD = "20y"
    FEATURES = ['MACD', 'MACD_Hist', 'RSI', 'ADX', 'SMA_Gap']
    
    db = StockDatabase()
    ticker_list = NikkeiTickerManager.get_tickers()
    
    print(f"\n=== 🚀 全 {len(ticker_list)} 銘柄の日足データ一括並列ダウンロードを開始 ===")
    print(f"対象期間: {FETCH_PERIOD} (Web接続を最初の1回に集約して高速化します)")
    print("-" * 75)
    
    try:
        tickers_str = " ".join(ticker_list)
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        
        df_batch = yf.download(
            tickers_str, 
            period=FETCH_PERIOD, 
            interval="1d", 
            group_by='ticker', 
            threads=True, 
            actions=False,
            progress=False
        )
        print("\n=== ダウンロードデータのデバッグ表示 ===")
        print(f"データフレームの形状: {df_batch.shape}")
        print(f"列名(Columns): {df_batch.columns.tolist()}")
        print("\n--- データの先頭5行 ---")
        print(df_batch.head())
        print("===================================\n")
        if not df_batch.empty:
            print("📥 ダウンロード完了。データをクレンジングしてローカルDBへ一括保存しています...")
            db.save_batch_data(df_batch)
            print("✅ ローカルDBへの一括同期が完了しました。")
        else:
            print("⚠️ 一括ダウンロードデータが空でした。個別処理側での補填に切り替えます。")
    except Exception as e:
        print(f"⚠️ 一括ダウンロード中にエラーが発生したため、個別取得へフォールバックします: {e}")

    pipeline = StockDataPipeline(features_list=FEATURES, default_period=FETCH_PERIOD)
    ai_brain = IncrementalLearningModel()
    ai_brain.load()
    
    predictions_pool = []
    total_training_time = 0.0
    trained_tickers_count = 0
    
    print(f"\n=== 全銘柄のデータ精査 ＆ 追加学習パイプラインを開始 ===")
    print("※ データベース同期済みのため、Web通信はスキップされ超高速に処理されます。")
    print("-" * 75)
    
    for idx, ticker in enumerate(ticker_list, 1):
        ticker_start = time.time()
        result = pipeline.fetch_and_transform(ticker)
        
        if result is None:
            continue
            
        X_past, y_past, X_latest_today, current_rsi, current_adx, trained_dates, need_web_fetch = result
        
        if need_web_fetch:
            time.sleep(0.5)
        
        if X_past.shape[0] > 0:
            train_start = time.time()
            ai_brain.learn_incremental(X_past, y_past)
            pipeline.db.update_trained_status(ticker, trained_dates)
            
            train_end = time.time()
            elapsed_train = train_end - train_start
            total_training_time += elapsed_train
            trained_tickers_count += 1
            
            ticker_end = time.time()
            elapsed_ticker = ticker_end - ticker_start
            fetch_status = "Web取得" if need_web_fetch else "DBローカル"
            print(f" 📈 [{ticker}] {len(trained_dates)}件のデータを追加学習完了 ({fetch_status} / 総時間: {elapsed_ticker:.4f}秒 / 機械学習: {elapsed_train:.4f}秒)")
        else:
            ticker_end = time.time()
            elapsed_ticker = ticker_end - ticker_start
            fetch_status = "Web接続あり" if need_web_fetch else "DBキャッシュ"
            print(f" 😴 [{ticker}] 未学習データなし ({fetch_status} / 処理時間: {elapsed_ticker:.4f}秒)")
            
        pred_return = ai_brain.predict_next_day(X_latest_today)
        predictions_pool.append({
            'Ticker': ticker,
            'Predicted_Return': pred_return,
            'RSI': current_rsi,
            'ADX': current_adx
        })
        
        if idx % 30 == 0 or idx == len(ticker_list):
            print(f"\n--- 📊 進捗レポート: {idx}/{len(ticker_list)} 銘柄完了 ---")

    if not predictions_pool:
        print("\n❌ エラー: 予測データを作成できませんでした。")
        return

    ai_brain.save()
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    df_results = pd.DataFrame(predictions_pool)
    df_results = df_results.sort_values(by='Predicted_Return', ascending=False)
    
    print("\n" + "="*60)
    print("🔮 【AI予測：明日伸びる確率・リターンが高い日経225銘柄 TOP 10】")
    print("="*60)
    print(df_results.head(10).to_string(index=False))
    
    print("\n" + "-"*60)
    print("⏱️  【処理時間計測レポート】")
    print(f"・全行程の総実行時間: {total_elapsed_time:.2f}秒 (約{total_elapsed_time/60:.1f}分)")
    if trained_tickers_count > 0:
        print(f"...追加機械学習の総時間: {total_training_time:.4f}秒 (対象: {trained_tickers_count}銘柄)")
        print(f"...1銘柄あたりの平均機械学習時間: {total_training_time / trained_tickers_count:.4f}秒")
    else:
        print("・追加機械学習の総時間: 0.0000秒 (すべての過去データが学習済みです)")
    print("-"*60 + "\n")

    if not df_results.empty:
        print("="*60)
        print(f"🎬 【リアルタイムUI】コンソール画面を起動します。リストから銘柄を選択してください...")
        print("="*60)
        
        root = tk.Tk()
        # pipeline と ai_brain を引数に渡して、GUI側での期間再計算を有効化
        app = StockPredictionApp(root, df_results, pipeline, ai_brain)
        root.mainloop()

if __name__ == "__main__":
    main()