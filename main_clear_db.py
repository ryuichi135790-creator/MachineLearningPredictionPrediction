from StockDatabase import StockDatabase

def clear_all_data():
    """DB内の全テーブル（daily_prices）からデータを削除する"""
    db = StockDatabase()
    
    print(">> ⚠️ 警告: データベースの全データを削除します。")
    confirm = input("本当に実行しますか？ (y/N): ")
    
    if confirm.lower() == 'y':
        try:
            with db._get_connection() as conn:
                cursor = conn.cursor()
                # DELETE FROM で全行削除
                cursor.execute("UPDATE daily_prices SET is_trained = 0")
                conn.commit()
                print(">> 完了: daily_prices の全データを削除しました。")
        except Exception as e:
            print(f">> エラー: 削除中に問題が発生しました: {e}")
    else:
        print(">> 中止しました。")

if __name__ == "__main__":
    clear_all_data()