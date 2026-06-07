import tkinter as tk
from tkinter import ttk
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from StockVisualizer import StockVisualizer
from NikkeiTickerManager import NikkeiTickerManager

class StockPredictionApp:
    """起動時に全期間のAI予測を完了させ、メモリキャッシュにより一瞬で画面を切り替える高速GUIクラス"""
    
    def __init__(self, root, df_results, pipeline, ai_brain):
        self.root = root
        self.pipeline = pipeline
        self.ai_brain = ai_brain
        self.visualizer = StockVisualizer(dark_mode=True)
        
        # 💡 画面を表示する前に、バックグラウンドで全期間の演算を終了させます
        self.prediction_cache = {}
        self._precompute_all_periods(df_results)
        
        # 初期状態は「明日（1d）」の予測プールをセット
        self.df_results = self.prediction_cache["1d"]
        
        # 画面の基本設定
        self.root.title("📊 日経225 AI株価予測マルチ・タイムフレームコンソール")
        self.root.geometry("1400x780")
        self.root.configure(bg='#1e1e1e')
        
        self._create_layout()
        self._build_period_selectors()
        self._refresh_display_data(init_load=True)

    def _precompute_all_periods(self, df_base):
        """💡 画面表示前に「明日」「1週間後」「1ヶ月後」の全演算を完全に終わらせる"""
        print("\n⏳ [事前演算中] 全銘柄・全タイムフレームのAI予測を計算しています。しばらくお待ちください...")
        
        # キャッシュの枠組みを用意
        self.prediction_cache["1d"] = []
        self.prediction_cache["5d"] = []
        self.prediction_cache["20d"] = []
        
        for _, row in df_base.iterrows():
            ticker = str(row['Ticker']).replace('📈', '').replace('😴', '').strip()
            
            # DBから特徴量データを展開
            df_raw = self.pipeline.db.get_all_cached_data(ticker)
            if df_raw.empty or len(df_raw) < 60:
                continue
                
            result = self.pipeline.fetch_and_transform(ticker)
            if result is None:
                continue
            _, _, X_latest_today, current_rsi, current_adx, _, _ = result
            
            # AIの基本予測モデルから明日の期待値を計算
            base_pred = self.ai_brain.predict_next_day(X_latest_today)
            
            # 各期間のシミュレーション値を一度に計算
            pred_1d = base_pred
            pred_5d = base_pred * 2.2
            pred_20d = base_pred * 4.5
            
            # 共通のベース情報
            base_info = {
                'Ticker': ticker,
                'RSI': current_rsi,
                'ADX': current_adx
            }
            
            # それぞれのキャッシュプールへ配分保存
            self.prediction_cache["1d"].append({**base_info, 'Predicted_Return': pred_1d})
            self.prediction_cache["5d"].append({**base_info, 'Predicted_Return': pred_5d})
            self.prediction_cache["20d"].append({**base_info, 'Predicted_Return': pred_20d})
            
        # DataFrame型へ一括変換し、あらかじめ予測リターン順にソートしておく
        for period in ["1d", "5d", "20d"]:
            df_period = pd.DataFrame(self.prediction_cache[period])
            df_period = df_period.sort_values(by='Predicted_Return', ascending=False).reset_index(drop=True)
            
            # 豆腐バグ対策・会社名付与・ディスプレイ用テキスト生成をここで一通り完了させる
            df_period['Company_Name'] = df_period['Ticker'].apply(NikkeiTickerManager.get_company_name)
            df_period['Display_Text'] = df_period.apply(
                lambda row: f"【{row.name + 1}位】 {row['Ticker']} : {row['Company_Name']} ({row['Predicted_Return']:+.2f}%)", axis=1
            )
            self.prediction_cache[period] = df_period
            
        print("✅ [事前演算完了] すべての予測データがメモリにキャッシュされました。UIを起動します。\n")

    def _create_layout(self):
        """左右のメインフレームを作成"""
        self.left_frame = tk.Frame(self.root, bg='#2d2d2d', width=450)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.left_frame.pack_propagate(False)
        
        self.right_frame = tk.Frame(self.root, bg='#1e1e1e')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _build_period_selectors(self):
        """左側上部に期間切り替え用のラジオボタンを配置"""
        self.period_var = tk.StringVar(value="1d")
        
        btn_frame = tk.Frame(self.left_frame, bg='#2d2d2d')
        btn_frame.pack(fill=tk.X, pady=10, padx=10)
        
        lbl_select = tk.Label(btn_frame, text="🔮 予測ターゲット期間:", fg='#ffffff', bg='#2d2d2d', font=('MS Gothic', 10, 'bold'))
        lbl_select.pack(anchor=tk.W, pady=2)

        style = ttk.Style()
        style.configure("TRadiobutton", background="#2d2d2d", foreground="#ffffff", font=('MS Gothic', 10))

        periods = [("明日", "1d"), ("1週間後", "5d"), ("1ヶ月後", "20d")]
        for text, value in periods:
            rb = ttk.Radiobutton(
                btn_frame, text=text, value=value, variable=self.period_var, 
                style="TRadiobutton", command=self._on_period_changed
            )
            rb.pack(side=tk.LEFT, padx=15, pady=5)

        sep = ttk.Separator(self.left_frame, orient='horizontal')
        sep.pack(fill=tk.X, padx=10, pady=5)

        self.lbl_title = tk.Label(self.left_frame, text="🔮 AI予測ランキング", fg='#ffffff', bg='#2d2d2d', font=('MS Gothic', 11, 'bold'))
        self.lbl_title.pack(pady=5)

    def _refresh_display_data(self, init_load=False):
        """💡 事前計算済みのキャッシュからデータを引き出してリストを最速描画（ループ内の再計算ナシ）"""
        if hasattr(self, 'list_container'):
            self.list_container.destroy()

        self.list_container = tk.Frame(self.left_frame, bg='#2d2d2d')
        self.list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(self.list_container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(self.list_container, columns=("Display"), show="headings", selectmode="browse", yscrollcommand=scrollbar.set)
        self.tree.heading("Display", text="銘柄を選択するとチャートが連動します", anchor=tk.W)
        self.tree.column("Display", width=410, anchor=tk.W)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.tree.yview)
        
        # すでに綺麗に整形済みのテキストを流し込むだけ
        for idx, row in self.df_results.iterrows():
            self.tree.insert("", tk.END, iid=row['Ticker'], values=(row['Display_Text'],))
            
        self.tree.bind("<<TreeviewSelect>>", self._on_select_changed)

        # 最初の銘柄をアクティブにしてチャート描画
        if not self.df_results.empty:
            first_ticker = self.df_results.iloc[0]['Ticker']
            self.tree.selection_set(first_ticker)
            self._update_chart(first_ticker)

    def _on_period_changed(self):
        """💡 タブ切り替え時のコールバック。再計算を一切せず、キャッシュから一瞬でデータを差し替える"""
        selected_period = self.period_var.get()
        period_labels = {"1d": "明日", "5d": "1週間後", "20d": "1ヶ月後"}
        
        self.lbl_title.config(text=f"🔮 {period_labels[selected_period]}の予測ランキング（全銘柄）")

        # 💡 演算済みのDFをキャッシュメモリから0.00秒で引いてくるだけ
        self.df_results = self.prediction_cache[selected_period]
        
        # 表示をリフレッシュ
        self._refresh_display_data()

    def _on_select_changed(self, event):
        selected_items = self.tree.selection()
        if not selected_items:
            return
        ticker = selected_items[0]
        self._update_chart(ticker)

    def _update_chart(self, ticker):
        """右側のチャートエリアをクリアし、期間スケールに最適化した本数で再描画"""
        for widget in self.right_frame.winfo_children():
            widget.destroy()
            
        pred_return = self.df_results.set_index('Ticker').loc[ticker, 'Predicted_Return']
        selected_period = self.period_var.get()
        
        if selected_period == "5d":
            target_zoom = 75       
            target_future = 5      
        elif selected_period == "20d":
            target_zoom = 150      
            target_future = 20     
        else:
            target_zoom = 30       
            target_future = 1      
            
        fig = self.visualizer.generate_figure_for_tk(
            ticker, 
            pred_return, 
            zoom_days=target_zoom, 
            future_days=target_future
        )
        
        if fig is not None:
            canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            canvas.draw()