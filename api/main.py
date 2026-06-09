from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ── Transformer : nettoyage + feature engineering ───────────────────────────
class FraudPreprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Nettoyage colonnes
        df.columns = (df.columns.str.strip()
                                 .str.replace(' ', '_')
                                 .str.replace('?', '')
                                 .str.replace(',', ''))

        # Feature engineering
        df['is_online'] = (df['Use_Chip'] == 'Online Transaction').astype(int)
        df['has_error'] = (df['Errors'].notna() & (df['Errors'] != '')).astype(int)
        df['is_night'] = df['Hour'].apply(lambda h: 1 if h >= 22 or h <= 5 else 0)

        return df


# ── Fonction d'entraînement ──────────────────────────────────────────────────
def train_pipeline(csv_path='fraud_sample_raw.csv'):

    print("Chargement des données brutes...")
    df = pd.read_csv(csv_path)

    # Nettoyage colonne cible
    df.columns = (df.columns.str.strip()
                             .str.replace(' ', '_')
                             .str.replace('?', '')
                             .str.replace(',', ''))

    if 'Is_Fraud' not in df.columns:
        df.rename(columns={'Is_Fraud_': 'Is_Fraud'}, inplace=True)

    df['Is_Fraud'] = df['Is_Fraud'].apply(lambda x: 1 if str(x).strip() == 'Yes' else 0)

    # Appliquer le preprocessing
    preprocessor = FraudPreprocessor()
    df = preprocessor.transform(df)

    # Features
    cat_features = ['Use_Chip', 'Errors']
    num_features = ['Amount', 'Hour', 'Year', 'Month', 'Day',
                    'is_online', 'has_error', 'is_night']
    features = num_features + cat_features

    X = df[features]
    y = df['Is_Fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline complet
    encoder = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ], remainder='passthrough')

    pipeline = Pipeline(steps=[
        ('encoder', encoder),
        ('model', lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ))
    ])

    print("Entraînement...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))

    joblib.dump(pipeline, 'fraud_pipeline.pkl')
    print("Modèle sauvegardé : fraud_pipeline.pkl")

    return pipeline


# ── API FastAPI ──────────────────────────────────────────────────────────────
app = FastAPI(title="Fraud Risk Scoring API")

# Charger ou entraîner le modèle au démarrage
try:
    pipeline = joblib.load('fraud_pipeline.pkl')
    print("Modèle chargé depuis fraud_pipeline.pkl")
except FileNotFoundError:
    print("Modèle introuvable — entraînement en cours...")
    pipeline = train_pipeline()


# Format d'une transaction brute
class Transaction(BaseModel):
    Amount: float
    Hour: int
    Year: int
    Month: int
    Day: int
    Use_Chip: str      # "Online Transaction", "Chip Transaction", "Swipe Transaction"
    Errors: str = ""   # "Bad CVV,", "Bad PIN,", "" si aucune erreur


@app.get("/")
def home():
    return {"message": "Fraud Risk Scoring API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    df = pd.DataFrame([transaction.model_dump()])

    # Feature engineering
    df['is_online'] = (df['Use_Chip'] == 'Online Transaction').astype(int)
    df['has_error'] = (df['Errors'].notna() & (df['Errors'] != '')).astype(int)
    df['is_night'] = df['Hour'].apply(lambda h: 1 if h >= 22 or h <= 5 else 0)

    features = ['Amount', 'Hour', 'Year', 'Month', 'Day',
                'is_online', 'has_error', 'is_night', 'Use_Chip', 'Errors']

    score = pipeline.predict_proba(df[features])[0][1]
    decision = "FRAUDE" if score > 0.5 else "NORMAL"

    return {
        "score_risque": round(float(score), 4),
        "decision": decision,
        "seuil": 0.5
    }