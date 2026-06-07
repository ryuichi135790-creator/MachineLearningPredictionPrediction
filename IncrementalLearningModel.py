import os
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

class IncrementalLearningModel:
    """モデルとスケーラーの保存・読み込み、および追加学習・予測を担当するクラス"""
    def __init__(self, model_path="nikkei_clean_model.pkl"):
        self.model_path = model_path
        self.model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def load(self):
        if os.path.exists(self.model_path):
            artifacts = joblib.load(self.model_path)
            self.model = artifacts['model']
            self.scaler = artifacts['scaler']
            self.is_trained = artifacts['is_trained']
            print(">> 【モデル復元】既存の学習済みモデルをロードしました。")
            return True
        print(">> 【新規作成】既存モデルがないため、新しいモデルとして開始します。")
        return False

    def save(self):
        artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        joblib.dump(artifacts, self.model_path)
        print(">> 今日の学習内容をモデルファイルに永続化しました。")

    def learn_incremental(self, X_past, y_past):
        # 1. StandardScaler の更新（必要であれば）
        # 完全に固定するのではなく、オンライン学習向きのスケーラーに変えるのが理想ですが、
        # 手始めに現在のデータをスケーラーに追加学習させる（partial_fit 相当）
        X_scaled = self.scaler.partial_fit(X_past).transform(X_past)
        
        # 2. モデルの学習
        if not self.is_trained:
            self.model.fit(X_scaled, y_past)
            self.is_trained = True
        else:
            self.model.partial_fit(X_scaled, y_past)

    def predict_next_day(self, X_latest):
        if not self.is_trained:
            raise ValueError("モデルが一度も学習されていません。")
        X_scaled = self.scaler.transform(X_latest.reshape(1, -1))
        return self.model.predict(X_scaled)[0]