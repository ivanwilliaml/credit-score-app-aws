import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Tahap 2 pipeline SageMaker: stratified train/test split (bukan feature preprocessing - itu ada di transformer.py)
class DataSplitter:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        if os.environ.get("SM_CHANNEL_INPUT") or os.path.exists("/opt/ml/processing"):
            self.input_file = "/opt/ml/processing/ingested/data_C.csv"
            self.out_train = "/opt/ml/processing/train"
            self.out_test = "/opt/ml/processing/test"
        else:
            self.input_file = "ingested/data_C.csv"
            self.out_train = "train"
            self.out_test = "test"
        self.test_size = test_size
        self.random_state = random_state

    def run(self):
        print("--- Step 2: Preprocessing & Train/Test Split ---")
        os.makedirs(self.out_train, exist_ok=True)
        os.makedirs(self.out_test, exist_ok=True)

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"❌ Error: {self.input_file} not found!")

        df = pd.read_csv(self.input_file)
        train_df, test_df = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state,
            stratify=df["Credit_Score"])
        train_df.to_csv(os.path.join(self.out_train, "train.csv"), index=False)
        test_df.to_csv(os.path.join(self.out_test, "test.csv"), index=False)
        print(f"✅ Split complete: train={len(train_df)}, test={len(test_df)}")


if __name__ == "__main__":
    splitter = DataSplitter()
    splitter.run()
