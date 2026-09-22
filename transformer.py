import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

# ----------------- Konstanta preprocessing (mengikuti notebook.ipynb) -----------------
ORDINAL_MAPS = {
    "Month": {m: i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July", "August"])},
    "Credit_Mix": {"Bad": 0, "Standard": 1, "Good": 2, "Unknown": 1},
    "Spending_Level": {"Low": 0, "High": 1, "Unknown": -1},
    "Payment_Size": {"Small": 0, "Medium": 1, "Large": 2, "Unknown": -1},
    "Payment_of_Min_Amount": {"No": 0, "Yes": 1, "Unknown": 0},
}
IQR_COLS = ["Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan", "Num_of_Delayed_Payment",
            "Num_Credit_Inquiries", "Delay_from_due_date", "Interest_Rate"]
P99_COLS = ["Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt",
            "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]
ENGINEERED = ["Debt_to_Income", "EMI_to_Salary", "Investment_Rate", "Balance_to_Salary",
              "Credit_History_Years", "Loan_per_Account", "Delay_per_Loan", "Debt_per_Card",
              "Total_Credit_Products", "Delay_to_History"]
HIGH_CARD = ["Occupation", "Primary_Loan"]

# Skema kolom mentah yang diterima pipeline (dipakai inference & app Streamlit)
RAW_FEATURES = ["Month", "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
                "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
                "Type_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
                "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
                "Credit_Utilization_Ratio", "Credit_History_Age", "Payment_of_Min_Amount",
                "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance"]


# Dipakai train.py: strategi oversampling, hanya kelas minoritas dinaikkan (bukan full balance semua kelas)
def smt_strategy(y):
    """SMOTE-Tomek terbatas: naikkan HANYA kelas minoritas sampai = jumlah kelas terbesar-kedua."""
    c = Counter(y)
    ordered = sorted(c, key=lambda k: c[k], reverse=True)   # [majority, middle, minority]
    minority = ordered[-1]
    return {minority: max(c[ordered[1]], c[minority])}


# Dipakai train.py (fit) & inference.py (transform) - satu class yang sama dipakai training & serving
class CreditScorePreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessing notebook (clean -> impute -> cap -> decompose -> FE -> ordinal + one-hot)."""

    # --- transformasi per-baris deterministik (tanpa state) ---
    # Bersihkan tipe data & noise mentah (angka nyasar simbol, umur invalid, kategori placeholder, dll)
    def _core_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data = data.drop(columns=["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"], errors="ignore")
        for col in ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
                    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
                    "Monthly_Balance"]:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True).replace("", np.nan),
                errors="coerce")
        data["Age"] = data["Age"].where(data["Age"].between(18, 100), np.nan)
        data["Monthly_Balance"] = data["Monthly_Balance"].where(data["Monthly_Balance"] >= 0, np.nan)
        for col in ["Num_of_Loan", "Num_Bank_Accounts", "Num_of_Delayed_Payment",
                    "Delay_from_due_date", "Num_Credit_Card", "Num_Credit_Inquiries"]:
            data[col] = pd.to_numeric(data[col], errors="coerce").clip(lower=0)

        def to_months(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)):
                return float(x)
            m = re.match(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(x))
            return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan
        data["Credit_History_Age"] = data["Credit_History_Age"].apply(to_months)

        data["Occupation"] = data["Occupation"].replace("_______", "Unknown")
        data["Credit_Mix"] = data["Credit_Mix"].replace("_", "Unknown")
        data["Payment_of_Min_Amount"] = data["Payment_of_Min_Amount"].replace("NM", "Unknown")
        data["Payment_Behaviour"] = data["Payment_Behaviour"].replace("!@9#%8", np.nan)
        data["Type_of_Loan"] = data["Type_of_Loan"].fillna("Not Specified")
        return data

    # Pecah kolom majemuk (Payment_Behaviour, Type_of_Loan) jadi beberapa fitur turunan
    def _decompose(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["Spending_Level"] = data["Payment_Behaviour"].str.extract(r"(Low|High)_spent")
        data["Payment_Size"] = data["Payment_Behaviour"].str.extract(r"(Small|Medium|Large)_value")
        data[["Spending_Level", "Payment_Size"]] = data[["Spending_Level", "Payment_Size"]].fillna("Unknown")
        data = data.drop(columns=["Payment_Behaviour"])
        loans = data["Type_of_Loan"].str.replace(" and ", ",", regex=False)
        data["Num_Loan_Types"] = loans.apply(
            lambda s: 0 if str(s).strip() == "Not Specified" else len([t for t in str(s).split(",") if t.strip()]))
        data["Primary_Loan"] = loans.str.split(",").str[0].str.strip()
        data = data.drop(columns=["Type_of_Loan"])
        return data

    # Feature engineering berbasis domain (rasio utang, cicilan, investasi, dll terhadap income/saldo)
    def _add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        monthly_income = data["Annual_Income"] / 12 + 1
        data["Debt_to_Income"] = data["Outstanding_Debt"] / monthly_income
        data["EMI_to_Salary"] = data["Total_EMI_per_month"] / (data["Monthly_Inhand_Salary"] + 1)
        data["Investment_Rate"] = data["Amount_invested_monthly"] / (data["Monthly_Inhand_Salary"] + 1)
        data["Balance_to_Salary"] = data["Monthly_Balance"] / (data["Monthly_Inhand_Salary"] + 1)
        data["Credit_History_Years"] = data["Credit_History_Age"] / 12
        data["Loan_per_Account"] = data["Num_of_Loan"] / (data["Num_Bank_Accounts"] + 1)
        data["Delay_per_Loan"] = data["Num_of_Delayed_Payment"] / (data["Num_of_Loan"] + 1)
        data["Debt_per_Card"] = data["Outstanding_Debt"] / (data["Num_Credit_Card"] + 1)
        data["Total_Credit_Products"] = data["Num_Bank_Accounts"] + data["Num_Credit_Card"] + data["Num_of_Loan"]
        data["Delay_to_History"] = data["Num_of_Delayed_Payment"] / (data["Credit_History_Age"] / 12 + 1)
        return data

    # Encode kolom ordinal (Month, Credit_Mix, dst) jadi angka sesuai urutan tingkatannya
    def _apply_ordinal(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for col, mp in ORDINAL_MAPS.items():
            data[col] = data[col].map(mp).fillna(-1).astype(float)
        return data

    # --- API sklearn ---
    # fit: pelajari statistik (median, cap outlier, one-hot) HANYA dari data train, simpan sebagai state
    def fit(self, X: pd.DataFrame, y=None):
        df = self._core_clean(X)
        self.num_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = df.select_dtypes(include=["object"]).columns.tolist()
        self.medians_ = {c: df[c].median() for c in self.num_cols_}
        df[self.num_cols_] = df[self.num_cols_].fillna(self.medians_)
        df[self.cat_cols_] = df[self.cat_cols_].fillna("Unknown")
        self.caps_ = {}
        for col in IQR_COLS:
            q1, q3 = df[col].quantile([0.25, 0.75]); iqr = q3 - q1
            self.caps_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        for col in P99_COLS:
            self.caps_[col] = (df[col].min(), df[col].quantile(0.99))
        for col, (lo, hi) in self.caps_.items():
            df[col] = df[col].clip(lo, hi)
        df = self._add_features(self._decompose(df))
        for col in ENGINEERED:
            hi = df[col].quantile(0.99)
            self.caps_[col] = (df[col].min(), hi)
            df[col] = df[col].clip(upper=hi)
        df = self._apply_ordinal(df)
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[HIGH_CARD])
        self.ohe_names_ = self.ohe_.get_feature_names_out(HIGH_CARD).tolist()
        self.feature_names_ = df.drop(columns=HIGH_CARD).columns.tolist() + self.ohe_names_
        return self

    # transform: terapkan statistik hasil fit ke data baru (test set / request inference), tanpa hitung ulang
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._core_clean(X)
        df[self.num_cols_] = df[self.num_cols_].fillna(self.medians_)
        df[self.cat_cols_] = df[self.cat_cols_].fillna("Unknown")
        for col in IQR_COLS + P99_COLS:                       # cap fitur dasar SEBELUM FE (identik fit)
            lo, hi = self.caps_[col]
            df[col] = df[col].clip(lo, hi)
        df = self._add_features(self._decompose(df))
        for col in ENGINEERED:
            df[col] = df[col].clip(upper=self.caps_[col][1])
        df = self._apply_ordinal(df)
        ohe_arr = self.ohe_.transform(df[HIGH_CARD])
        out = pd.concat([df.drop(columns=HIGH_CARD),
                         pd.DataFrame(ohe_arr, columns=self.ohe_names_, index=df.index)], axis=1)
        return out.reindex(columns=self.feature_names_, fill_value=0)
