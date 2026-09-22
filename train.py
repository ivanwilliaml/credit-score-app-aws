import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline as SkPipeline
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

from transformer import CreditScorePreprocessor, smt_strategy

SEED = 42


# Tahap 3 pipeline SageMaker: latih LightGBM (hyperparameter sama dgn local_deployment) & simpan model
class CreditModelTrainer:
    """Fits ImbPipeline (preprocessing → SMOTE-Tomek → LightGBM) on training data,
    then saves the deployable sklearn Pipeline (preprocessing → classifier) as model_credit.joblib."""

    def __init__(self, random_state: int = SEED):
        self.train_dir = os.environ.get("SM_CHANNEL_TRAIN", "train")
        self.model_dir = os.environ.get("SM_MODEL_DIR", "model")
        self.random_state = random_state

    def run(self) -> str:
        print("--- Step 3: Training ---")
        os.makedirs(self.model_dir, exist_ok=True)
        train_file = os.path.join(self.train_dir, "train.csv")

        if not os.path.exists(train_file):
            raise FileNotFoundError(f"❌ Error: {train_file} not found!")

        df = pd.read_csv(train_file)
        X_train = df.drop("Credit_Score", axis=1)
        y_train = df["Credit_Score"]

        clf = LGBMClassifier(n_estimators=600, num_leaves=127, learning_rate=0.05,
                             random_state=self.random_state, n_jobs=-1, verbose=-1)
        train_pipe = ImbPipeline([
            ("preprocessing", CreditScorePreprocessor()),
            ("smote_tomek", SMOTETomek(
                smote=SMOTE(sampling_strategy=smt_strategy, random_state=self.random_state),
                random_state=self.random_state)),
            ("classifier", clf),
        ])
        train_pipe.fit(X_train, y_train)

        # Buang step SMOTE-Tomek untuk pipeline deployment (cuma dibutuhkan saat training, bukan saat prediksi)
        model = SkPipeline([
            ("preprocessing", train_pipe.named_steps["preprocessing"]),
            ("classifier", train_pipe.named_steps["classifier"]),
        ])

        output_path = os.path.join(self.model_dir, "model_credit.joblib")
        joblib.dump(model, output_path)
        print(f"✅ Training complete. Model saved to {output_path}")
        return output_path


if __name__ == "__main__":
    trainer = CreditModelTrainer()
    trainer.run()
