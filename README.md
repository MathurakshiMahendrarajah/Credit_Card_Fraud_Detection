# 💳 Credit Card Fraud Detection (Machine Learning + Streamlit)

**Python | Scikit-Learn | XGBoost | Streamlit**  

This project implements a **machine learning pipeline** to detect fraudulent credit card transactions. The focus is on **handling highly imbalanced data**, creating robust features, and providing **real-time predictions** via a **Streamlit web app**.

---

## 🔹 Problem Statement

Fraud detection is a **binary classification problem**:

| Label | Meaning                  |
|-------|--------------------------|
| 0     | Legitimate transaction   |
| 1     | Fraudulent transaction   |

**Challenge:** Fraudulent transactions are extremely rare, making standard accuracy metrics unreliable.  

---

## 📂 Dataset

- Anonymized credit card transaction dataset  
- Numerical and categorical features derived from transaction behavior  
- Fraudulent transactions represent a very small fraction of total samples  
- **Dataset Source:** [Kaggle - Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)  

> Note: Raw data is not included in the repo due to size constraints.

---

## 🛠 Approach

The project follows a **complete machine learning workflow**:

1. Data loading and inspection  
2. Exploratory Data Analysis (EDA)  
3. Handling **class imbalance**  
4. Feature engineering and preprocessing:  
   - Derived features like `is_night`, `is_high_amount`, `log_amt`  
   - Scaling numeric features and one-hot encoding categorical variables  
5. Training models:  
   - **Logistic Regression** (baseline linear model)  
   - **Random Forest** (ensemble tree-based model)  
   - **XGBoost** (final deployed model)  
6. Evaluation using **imbalance-aware metrics**  
7. Deployment as a **Streamlit web app**

---

## ⚙️ Models Implemented

| Model                | Type             | Notes                                         |
|---------------------|-----------------|-----------------------------------------------|
| Logistic Regression  | Linear           | Baseline model                                |
| Random Forest        | Tree-based       | Handles non-linear relationships             |
| XGBoost              | Gradient Boosted | Selected for final deployment due to performance |

---

## 📊 Evaluation Metrics

Due to **class imbalance**, evaluation focuses on **precision, recall, and F1-score**:

- **Precision**: How many predicted frauds are actually fraud  
- **Recall**: How many actual frauds were correctly predicted  
- **F1-score**: Balance between precision and recall  
- **Confusion Matrix**: Visual representation of predictions  
- **Precision–Recall Curve**: Evaluate classifier threshold performance  

> Accuracy is **not used** as the primary metric.

---

## 🖥 Streamlit Deployment

The trained **XGBoost model** is deployed as an **interactive web app**:

🔗 [Click here to access the app](https://creditcardfrauddetection-83scm92ncfwceyuzvl2dk6.streamlit.app)  

**Features of the app:**

- Left-side **input panel**: Enter transaction amount, time, customer info, state, category, and gender  
- Right-side **output panel**: Shows predicted label, fraud probability, and progress bar  
- **Threshold-based prediction** (default 50%)  
- User-friendly, wide layout for clear visualization  

---

## 🧩 How it Works

1. **Input preprocessing**:  
   - Encodes categorical variables (`state`, `category`, `gender`)  
   - Creates derived features like `is_night`, `is_high_amount`, `log_amt`, `amt_to_mean`  

2. **ML Pipeline**:  
   - Uses a **ColumnTransformer** for numeric scaling and one-hot encoding  
   - Passes processed data into **XGBoost classifier**  

3. **Threshold-based prediction**:  
   - Probabilities above **0.5** → Fraud  
   - Probabilities below **0.5** → Legitimate  

---

## 🔧 Installation (Optional - Local Run)

1.  ```bash
    git clone https://github.com/MathurakshiMahendrarajah/Credit_Card_Fraud_Detection.git
    cd Credit_Card_Fraud_Detection
    
    pip install -r requirements.txt
    
    streamlit run app.py

## Screenshots
<img width="1643" height="849" alt="Screenshot 2025-12-30 at 8 25 29 PM" src="https://github.com/user-attachments/assets/6bba7fe2-d982-4620-8b69-17bfde178b1d" />

