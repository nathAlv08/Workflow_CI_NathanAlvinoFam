import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

def main():
    print("Memuat dataset untuk Workflow CI...")
    # Path relatif terhadap file modelling.py
    data_path = "dataset_preprocessing/credit_risk_clean.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Data tidak ditemukan di {data_path}")
        return

    df = pd.read_csv(data_path)

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()

    # mlflow.start_run() akan otomatis nyambung ke "Bos" (mlflow run)
    with mlflow.start_run() as run:
        print("Melatih model di environment MLflow Project...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        print(f"Training Selesai! Run ID: {run.info.run_id}")

        # Simpan Run ID ke file teks agar bisa dibaca oleh GitHub Actions
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    main()