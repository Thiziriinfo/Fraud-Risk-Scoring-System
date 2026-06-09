# Fraud Risk Scoring System
Détection de fraude bancaire par scoring ML — IBM TabFormer Dataset

---

## Présentation du projet

Ce projet construit un système de scoring de risque de fraude sur des transactions bancaires réelles synthétiques. L'objectif : identifier automatiquement les transactions frauduleuses parmi des millions d'opérations, en s'appuyant sur des modèles de Machine Learning utilisés en production dans le secteur financier.

**Stack** : Python · pandas · LightGBM · XGBoost · scikit-learn · FastAPI · Matplotlib · Seaborn

---

## Dataset

**Source** : [IBM TabFormer](https://github.com/IBM/TabFormer) — Credit Card Transaction Dataset  
**Publication** : *"Tabular Transformers for Modeling Multivariate Time Series"* — IBM Research  
**Volume** : 24M+ transactions synthétiques réalistes, labelisées fraude / non-fraude  
**Variables** : montant, type de transaction (Swipe/Chip/Online), marchand, heure, erreurs, localisation

Ce dataset a été choisi pour sa richesse en variables métier interprétables, contrairement aux datasets Kaggle classiques dont les features sont anonymisées (PCA). Chaque variable a un sens concret pour une équipe risque bancaire.

---

## Choix de la plage temporelle

Le dataset couvre 1991 à 2020. Plusieurs plages ont été testées et comparées avant de retenir **2015-2019** :

| Plage | Transactions | Fraudes | Taux |
|-------|-------------|---------|------|
| 2015-2019 | 8 579 208 | 11 693 | 0.136% |
| 2012-2019 | 13 513 297 | 16 096 | 0.119% |
| 2010-2019 | 16 575 073 | 19 986 | 0.121% |

**Choix retenu : 2015-2019**
- Taux de fraude le plus élevé — meilleur signal pour le modèle
- Patterns de fraude récents et homogènes
- Volume gérable pour l'entraînement
- 2020 exclue : données incomplètes, aucune fraude recensée
- Les pics 2008-2010 liés à la crise financière introduiraient un biais

---

## Démarche

### 1. EDA
- Analyse de la distribution temporelle — justification de la plage choisie
- Comparaison des montants fraude vs normal (KDE + Boxplot)
- Taux de fraude par type de transaction : Online (0.560%) vs Chip (0.080%) vs Swipe (0.060%)
- Analyse des erreurs : Bad CVV + Insufficient Balance = 8% de fraude
- Patterns temporels : pic de fraude entre 10h et 16h

### 2. Préprocessing
- Nettoyage des colonnes (Amount, Errors?, Merchant State)
- Création de la variable cible numérique
- Échantillon stratifié : toutes les fraudes (11 693) + 200k transactions normales
  - Ratio 1/17 — choisi après comparaison avec ratio 1/43 (500k normales)
  - Meilleur compromis Precision/Recall pour un système de scoring bancaire

### 3. Feature Engineering
- Extraction de l'heure depuis la colonne Time
- Variable `is_online` — transaction en ligne ou non
- Variable `has_error` — présence d'une erreur sur la transaction
- Encodage One-Hot des variables catégorielles (Use Chip, Errors?)

### 4. Modélisation
Deux modèles comparés, standards industrie en détection de fraude :

| Modèle | AUC-ROC | Recall (fraude) | Precision (fraude) |
|--------|---------|-----------------|-------------------|
| LightGBM | **0.9904** | **0.94** | **0.59** |
| XGBoost | 0.9885 | 0.92 | 0.57 |

**Modèle retenu : LightGBM** — meilleur sur tous les indicateurs, plus rapide, standard industrie sur les gros volumes de données financières.

Validation croisée 5 folds — écart-type AUC : 0.0004 → pas d'overfitting.

### 5. Feature Importance

Les variables les plus prédictives selon LightGBM :
1. **MCC** (Merchant Category Code) — le type de commerce est le signal le plus fort
2. **Merchant Name** — certains marchands sont des cibles récurrentes
3. **Amount** — les fraudes portent sur des montants 2.5x plus élevés
4. **Year / Hour** — dimension temporelle importante

Insight métier : le **profil du marchand** est plus prédictif que le comportement de l'utilisateur.

### 6. Pipeline ML & API

Le modèle entraîné est industrialisé via un pipeline sklearn et exposé en API REST avec FastAPI.

**Pipeline** : le preprocessing, le feature engineering et l'encodage sont enchaînés dans un objet sklearn `Pipeline` — un CSV brut en entrée suffit pour reproduire l'intégralité du processus sans intervention manuelle.

**API FastAPI** : le modèle est accessible via une route `/predict` qui accepte une transaction en JSON et retourne un score de risque et une décision en temps réel.

```bash
# Lancer l'API localement
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

Exemple de requête :

```json
{
  "Amount": 250.0,
  "Hour": 11,
  "Year": 2019,
  "Month": 6,
  "Day": 15,
  "Use_Chip": "Online Transaction",
  "Errors": ""
}
```

Réponse :

```json
{
  "score_risque": 0.0121,
  "decision": "NORMAL",
  "seuil": 0.5
}
```

La documentation interactive est disponible sur `http://127.0.0.1:8000/docs`.

---

## Résultats et limites

**AUC-ROC : 0.9904** — le modèle distingue très bien fraudes et transactions normales.

**Recall : 0.94** — 94% des fraudes sont détectées. En banque, on privilégie le Recall : mieux vaut bloquer une transaction légitime que laisser passer une fraude.

**Limites** : le dataset est synthétique — les patterns sont plus réguliers que sur des données bancaires réelles. En production, un AUC de 0.85-0.92 est un bon score sur données réelles.

---

## Lancer le projet

```bash
git clone https://github.com/Thiziriinfo/Fraud-Risk-Scoring-System.git
```

Ouvrir le notebook dans Google Colab — le dataset se télécharge automatiquement depuis IBM TabFormer.

---

## Auteur

**Thiziri Abchiche** — Data Analyst  
[Portfolio](https://thiziriinfo.github.io) · [LinkedIn](https://linkedin.com/in/thiziri-abchiche)
