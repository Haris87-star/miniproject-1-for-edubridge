# 🏠 House Price Prediction (Mini Project 1 for EduBridge)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![EduBridge](https://img.shields.io/badge/EduBridge-Mini%20Project%201-success.svg)](https://www.edubridgeindia.com/)

An end-to-end Machine Learning web application designed to predict residential property prices based on key housing features (living area square footage, number of bedrooms, bathrooms, and property age). Built as part of **EduBridge Mini Project 1**.

---

## 📁 Repository Structure

```
House-Price-Prediction/
│
├── app.py                  # Interactive Streamlit web application
├── train_model.py          # Model training, comparison, and evaluation script
├── house_price_model.pkl   # Serialized trained machine learning model
├── requirements.txt        # Python project dependencies
├── README.md               # Project documentation and guide
└── .gitignore              # Git ignored files (cache, venv, IDE configs)
```

---

## 🚀 Key Features

- **Multi-Model Evaluation**: Automatically trains and benchmarks multiple regression algorithms (**Linear Regression**, **Decision Tree**, and **Random Forest**).
- **Model Serialization**: Automatically saves the top-performing model and feature metadata into `house_price_model.pkl`.
- **Interactive Web UI**: A clean, responsive user interface built using **Streamlit** allowing users to adjust sliders and numerical inputs to receive instantaneous property valuations.
- **Key Metrics Display**: Provides real-time estimations alongside estimated price per square foot.

---

## 🧠 Machine Learning Workflow

### 1. Features
The model learns from property attributes:
- `area_sqft`: Total living area in square feet (600 – 4,500 sq ft)
- `bedrooms`: Total number of bedrooms (1 – 6)
- `bathrooms`: Total number of bathrooms (1 – 4)
- `age_years`: Age of the property in years (0 – 50)
- `price`: Target property price ($)

### 2. Model Benchmarking
Models are evaluated on an 80/20 train-test split using standard regression metrics:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Coefficient of Determination ($R^2$ Score)**

| Model | MAE ($) | RMSE ($) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | **$12,535.86** | **$16,275.20** | **0.9948** |
| Random Forest | $19,507.79 | $24,124.51 | 0.9886 |
| Decision Tree | $33,581.79 | $40,808.54 | 0.9672 |

> The best model (**Linear Regression**, $R^2 = 0.9948$) is exported to `house_price_model.pkl`.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ installed
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/Haris87-star/miniproject-1-for-edubridge.git
cd miniproject-1-for-edubridge
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Step 1: (Optional) Train the Model
The pre-trained model `house_price_model.pkl` is already included. If you want to retrain the model:
```bash
python train_model.py
```

### Step 2: Launch the Streamlit Web App
```bash
streamlit run app.py
```

Your default web browser will automatically open:
```
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```

---

## 📦 Dependencies

- `streamlit` - Web application interface
- `scikit-learn` - Machine learning models & evaluation metrics
- `pandas` - Data manipulation & tabular structures
- `numpy` - Numerical computing & reproducible dataset generation
- `joblib` - Model serialization and loading

---

## 👨‍💻 Author

- **Harish** ([@Haris87-star](https://github.com/Haris87-star))
- **Program**: EduBridge Data Science / Machine Learning Course
- **Project**: Mini Project 1
