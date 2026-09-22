import sys

from data_ingestion import DataIngestion
from preprocessing import DataSplitter
from train import CreditModelTrainer
from evaluation import ModelEvaluator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# Orkestrator: menjalankan 4 tahap (ingestion -> split -> train -> evaluate) versi cloud/SageMaker
class CreditScorePipeline:
    def __init__(self, macro_f1_threshold: float = 0.65):
        self.macro_f1_threshold = macro_f1_threshold

        # Core component instantiation
        self.ingestor = DataIngestion()
        self.splitter = DataSplitter()
        self.trainer = CreditModelTrainer()
        self.evaluator = ModelEvaluator()

    def execute(self):
        print("🚀 Executing AWS Credit Score Pipeline...\n")

        # 1. Handle raw data ingestion safely
        self.ingestor.run()

        # 2. Preprocessing & stratified train/test split
        self.splitter.run()

        # 3. Fit model and save deployable pipeline artifact
        self.trainer.run()

        # 4. Evaluate and write evaluation.json
        acc, macro_f1, recall_poor = self.evaluator.run()

        # 5. Final conditional release assessment
        print("\n--- Deployment Approval Decision ---")
        if macro_f1 >= self.macro_f1_threshold:
            print(f"🎉 Success: Macro-F1 ({macro_f1:.3f}) lolos QA. Disetujui untuk deployment!")
        else:
            print(f"❌ Rejected: Macro-F1 ({macro_f1:.3f}) di bawah threshold ({self.macro_f1_threshold})")


if __name__ == "__main__":
    pipeline = CreditScorePipeline(macro_f1_threshold=0.65)
    pipeline.execute()
