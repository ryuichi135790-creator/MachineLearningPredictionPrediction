import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from NikkeiTickerManager import NikkeiTickerManager

class StockVisualizer:
    """株価データおよびAIの予測結果を可視化（チャート描画）するクラス"""
    
    def __init__(self, db_path="nikkei_stock_data.db", dark_mode=True):
        self.db_path = db_path
        
        if dark_mode:
            self.bg_color = '#1e1e1e'       
            self.text_color = '#ffffff'     
            self.grid_color = '#444444'     
            self.up_color = '#ff4d4d'       
            self.down_color = '#00cc66'     
            self.pred_up_color = '#ff9900'   # 予測足（陽線）
            self.pred_down_color = '#b36b00' # 予測足（陰線）
        else:
            self.bg_color = '#ffffff'
            self.text_color = '#000000'
            self.grid_color = '#cccccc'
            self.up_color = '#ee0000'
            self.down_color = '#00aa00'
            self.pred_up_color = '#ff6600'
            self.pred_down_color = '#993300'

    def generate_figure_for_tk(self, ticker_symbol, predicted_return, zoom_days=30, future_days=1):
        """
        【GUI連動用】指定された表示本数（zoom_days）と予測未来日数（future_days）に基づいて
        未来の連続するローソク足を含めてシミュレーション描画する
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT date, open, high, low, close, volume FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(ticker_symbol, zoom_days + 10))
            conn.close()
            
            if df.empty or len(df) < 5:
                return None

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df = df.tail(zoom_days).reset_index(drop=True)

            last_row = df.iloc[-1]
            today_close = last_row['close']
            today_date = last_row['date']
            
            last_idx = len(df) - 1

            # 過去のデータから1日あたりの平均的な値動き（ボラティリティ）を計算
            df['daily_pct'] = df['close'].pct_change()
            volatility = df['daily_pct'].std()
            if pd.isna(volatility) or volatility == 0:
                volatility = 0.015 # フォールバック値（1.5%）

            avg_high_offset = (df['high'] - df[['open', 'close']].max(axis=1)).mean()
            avg_low_offset = (df[['open', 'close']].min(axis=1) - df['low']).mean()

            from matplotlib.figure import Figure
            fig = Figure(figsize=(10, 5), dpi=100)
            fig.patch.set_facecolor(self.bg_color)
            ax = fig.add_subplot(111)
            ax.set_facecolor(self.bg_color)
            
            # 1. 実績ローソク足（過去データ）の描画
            for i in range(len(df)):
                row = df.iloc[i]
                color = self.up_color if row['close'] >= row['open'] else self.down_color
                lower_body, upper_body = min(row['open'], row['close']), max(row['open'], row['close'])
                ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1.5)
                ax.bar(i, upper_body - lower_body, bottom=lower_body, color=color, width=0.6)

            # 2. 💡 未来のローソク足を1本ずつ連続シミュレーション生成して描画
            current_sim_close = today_close
            # 最終目的地のリターンから、1日あたりの平均必要リターン（トレンドの傾き）を逆算
            daily_trend_return = predicted_return / future_days 
            
            # 乱数のシードを銘柄コードで固定し、選択し直しても同じ形状の波形が再現されるようにする
            np.random.seed(abs(hash(ticker_symbol)) % (10**8))

            for step in range(1, future_days + 1):
                pred_idx = last_idx + step
                
                # 始値は前日の終値
                sim_open = current_sim_close
                
                # トレンドの傾きに、過去のボラティリティに基づいたランダムなノイズ（ランダムウォーク）をブレンド
                random_noise = np.random.normal(0, volatility)
                sim_return_pct = (daily_trend_return / 100) + random_noise
                
                # 今日の終値
                sim_close = sim_open * (1 + sim_return_pct)
                
                # 最終ステップ（目標期日）は、AIの予測したターゲットリターンにジャストフィットさせる
                if step == future_days:
                    sim_close = today_close * (1 + (predicted_return / 100))

                # ヒゲの計算
                sim_lower_body, sim_upper_body = min(sim_open, sim_close), max(sim_open, sim_close)
                sim_high = sim_upper_body + (avg_high_offset * np.random.uniform(0.5, 1.5))
                sim_low = sim_lower_body - (avg_low_offset * np.random.uniform(0.5, 1.5))
                
                # 陽線・陰線の色分け
                p_color = self.pred_up_color if sim_close >= sim_open else self.pred_down_color
                
                # 未来のローソク足を描画
                ax.plot([pred_idx, pred_idx], [sim_low, sim_high], color=p_color, linewidth=1.5, linestyle='-')
                ax.bar(pred_idx, sim_upper_body - sim_lower_body, bottom=sim_lower_body, color=p_color, width=0.6, alpha=0.85)
                
                # 次の日のために終値を保存
                current_sim_close = sim_close

            # 3. 今日から最終目的地への軌跡点線を描画
            final_pred_idx = last_idx + future_days
            final_pred_close = today_close * (1 + (predicted_return / 100))
            ax.plot([last_idx, final_pred_idx], [today_close, final_pred_close], color=self.pred_up_color, linewidth=1.2, linestyle=':')

            # 4. X軸の目盛りと日付ラベルの動的生成
            tick_indices = list(range(0, len(df), max(1, zoom_days // 5)))
            if last_idx not in tick_indices:
                tick_indices.append(last_idx)
            
            # 未来の各チェックポイントをインデックスに追加
            if final_pred_idx not in tick_indices:
                tick_indices.append(final_pred_idx)
            
            tick_labels = [df.iloc[i]['date'].strftime('%m/%d') for i in tick_indices[:-1]]
            
            # 未来の予測期日のラベル設定
            if future_days == 1:
                future_date = today_date + timedelta(days=3 if today_date.weekday() == 4 else 1)
                tick_labels.append(future_date.strftime('%m/%d') + "\n(明日予測)")
            elif future_days == 5:
                tick_labels.append("1週間後\n(予測足5本)")
            else:
                tick_labels.append("1ヶ月後\n(予測足20本)")
            
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels, color=self.text_color)
            ax.tick_params(axis='y', colors=self.text_color)
            
            # タイトルとグリッド
            company_name = NikkeiTickerManager.get_company_name(ticker_symbol)
            ax.set_title(f"📈 {ticker_symbol} : {company_name}", color=self.text_color, fontsize=12, fontweight='bold', fontname='MS Gothic')
            ax.grid(True, color=self.grid_color, linestyle=':', alpha=0.6)
            
            # 現在値の水平基準線
            ax.axhline(today_close, color=self.text_color, linestyle=':', alpha=0.5)
            ax.text(last_idx + 0.5, today_close, f"Today:{today_close:.1f}", color=self.text_color, va='center', fontsize=8)
            
            # 予測結果テキストの配置
            text_align = 'right' if future_days > 1 else 'left'
            text_x_offset = -0.5 if future_days > 1 else 0.5
            ax.text(final_pred_idx + text_x_offset, final_pred_close, f"Target:{predicted_return:+.2f}%", color=self.pred_up_color, va='center', ha=text_align, fontsize=9, fontweight='bold')
            
            ax.set_xlim(-1, final_pred_idx + (max(1, future_days // 4)))
            
            return fig
        except Exception as e:
            print(f"💥 [{ticker_symbol}] グラフ生成例外: {e}")
            return None