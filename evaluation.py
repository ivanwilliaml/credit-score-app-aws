import os
import json
import joblib
import pandas as pd
from typing import Tuple
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report


# Tahap 4 pipeline: load model & test set, hitung metrik, simpan evaluation.json (format SageMaker)
class ModelEvaluator:
    def __init__(self):
        if os.environ.get("SM_CHANNEL_TEST") or os.path.exists("/opt/ml/processing"):
            self.model_path = "/opt/ml/processing/model/model_credit.joblib"
            self.test_path = "/opt/ml/processing/test/test.csv"
            self.output_dir = "/opt/ml/processing/evaluation"
        else:
            self.model_path = "model/model_credit.joblib"
            self.test_path = "test/test.csv"
            self.output_dir = "eval"

    def run(self) -> Tuple[float, float, float]:
        print("--- Step 4: Evaluation ---")
        os.makedirs(self.output_dir, exist_ok=True)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Missing model at: {self.model_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"❌ Missing test data at: {self.test_path}")

        model = joblib.load(self.model_path)
        test_df = pd.read_csv(self.test_path)
        X_test = test_df.drop("Credit_Score", axis=1)
        y_test = test_df["Credit_Score"]

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        recall_poor = recall_score(y_test, preds, labels=["Poor"], average="macro")

        print(classification_report(y_test, preds, digits=3))
        print(f"Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f} | Recall(Poor)={recall_poor:.4f}")

        report = {
            "multiclass_classification_metrics": {
                "accuracy": {"value": acc, "standard_deviation": "NaN"},
                "macro_f1": {"value": macro_f1, "standard_deviation": "NaN"},
                "recall_poor": {"value": recall_poor, "standard_deviation": "NaN"},
            }
        }
        output_path = os.path.join(self.output_dir, "evaluation.json")
        with open(output_path, "w") as f:
            json.dump(report, f)
        print(f"✅ Evaluation complete. Report saved to: {output_path}")
        return acc, macro_f1, recall_poor


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()
