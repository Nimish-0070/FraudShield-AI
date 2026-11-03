# 🛡️ FraudShield AI
### Intelligent Credit Card Fraud Detection & Risk Analysis

An end-to-end Machine Learning project that detects potentially fraudulent credit card transactions using classification algorithms, class-imbalance handling, model optimization and Explainable AI.

## 📌 Project Overview

Credit card fraud is a major challenge in digital financial transactions. Fraudulent transactions are rare compared with legitimate transactions, making fraud detection a highly imbalanced binary classification problem.

**FraudShield AI** aims to identify potentially fraudulent transactions by learning patterns from historical transaction data.

The project covers the complete Machine Learning lifecycle, from data understanding and preprocessing to model evaluation, explainability and Streamlit deployment.

## 🎯 Project Objectives

- Analyze historical credit card transaction data.
- Identify patterns associated with fraudulent transactions.
- Handle severe class imbalance.
- Compare multiple Machine Learning algorithms.
- Optimize model performance using hyperparameter tuning.
- Explain model predictions using SHAP.
- Deploy the trained model through a Streamlit application.

## 📊 Dataset

**Dataset:** European Credit Card Fraud Detection Dataset

**Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Dataset characteristic | Value |
|---|---:|
| Total transactions | 284,807 |
| Input features | 30 |
| Target variable | Class |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |

### Feature Description

- `Time`: Seconds elapsed between a transaction and the first transaction in the dataset.
- `Amount`: Transaction amount.
- `V1–V28`: Anonymized features obtained through Principal Component Analysis (PCA).
- `Class`: Target variable.

Target encoding:

- `0` – Legitimate transaction
- `1` – Fraudulent transaction

**Note:** The dataset is highly imbalanced, with fraudulent transactions representing approximately 0.173% of the original records.

The dataset is not included in this repository. Download it from Kaggle and place `creditcard.csv` inside the `data/` directory.

## 🧠 Machine Learning Methodology

### 1. Data Understanding and Exploratory Data Analysis

- Dataset structure and statistical analysis.
- Missing-value and duplicate inspection.
- Fraud versus legitimate transaction distribution.
- Transaction amount and time analysis.
- Correlation analysis.

### 2. Data Preprocessing

- Duplicate record inspection and handling.
- Feature and target separation.
- Stratified train-test splitting.
- StandardScaler for `Time` and `Amount`.
- Consistent preprocessing between training and prediction.

### 3. Baseline Model Development

The following classification algorithms are compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### 4. Class Imbalance Handling

Techniques investigated:

- SMOTE
- Random Undersampling
- Class-weighted Logistic Regression
- Class-weighted Random Forest
- XGBoost scale_pos_weight

Resampling is applied only to training data to prevent data leakage.

### 5. Model Optimization

- RandomizedSearchCV
- Cross-validation
- Hyperparameter optimization
- Classification threshold analysis

### 6. Model Evaluation

Performance is evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Accuracy alone is not sufficient for this highly imbalanced classification problem.

### 7. Explainable AI

SHAP (SHapley Additive exPlanations) is used to investigate:

- Global feature importance.
- Feature contributions to predictions.
- Individual transaction explanations.

**Limitation:** PCA-transformed features are anonymized, so their original business meanings are not available.

## 🖥️ Streamlit Application

FraudShield AI includes an interactive application with:

- Fraud detection dashboard.
- Single transaction analysis using sample dataset records.
- Bulk CSV transaction prediction.
- Fraud probability scoring.
- Legitimate versus potentially fraudulent classification.
- Model information.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Imbalance Handling | Imbalanced-learn |
| Explainable AI | SHAP |
| Application | Streamlit |
| Model Persistence | Joblib |
| Development | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

## 📁 Project Structure

```text
FraudShield-AI/
│
├── app/
│   └── app.py
│
├── data/
│   └── README.md
│
├── models/
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── screenshots/
│
├── requirements.txt
├── .gitignore
└── README.md
```

Additional notebooks and Python modules will be added as the project develops.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nimish-0070/FraudShield-AI.git
cd FraudShield-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the Kaggle dataset and place the CSV file here:

```text
data/creditcard.csv
```

### 5. Run the Streamlit application

```bash
streamlit run app/app.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

## 📈 Model Performance

Model performance will be documented here after reproducible evaluation on the held-out test dataset.

Metrics to be reported:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

No performance figures are claimed until the final evaluation is verified.

## ⚠️ Limitations

- PCA features V1–V28 are anonymized and lack direct business interpretation.
- The project uses a historical benchmark dataset rather than live banking transactions.
- Predictions are model estimates, not confirmed fraud determinations.
- Production deployment would require real-time feature engineering, monitoring, security and additional fraud investigation processes.

## 🔮 Future Improvements

- Improve model calibration and threshold selection.
- Add more comprehensive model monitoring.
- Develop a more detailed risk analytics dashboard.
- Add transaction prediction history.
- Explore real-time transaction scoring architecture.
- Improve automated model evaluation and reproducibility.

## 👨‍💻 Author

**Nimish Sawant**

B.Tech – Computer Science and Engineering

Interested in Machine Learning, Data Science and Data Analytics.

- GitHub: [Nimish-0070](https://github.com/Nimish-0070)

---

⭐ If you find this project interesting, feel free to explore the repository and share feedback.