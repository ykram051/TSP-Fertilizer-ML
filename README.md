# TSP Fertilizer Machine Learning Dashboard

## Overview

This project aims to analyze industrial production data from the Triple Super Phosphate (TSP) manufacturing process and develop a machine learning model capable of predicting critical production parameters. The trained model will be deployed within a real-time dashboard to assist operators in monitoring the production process and supporting data-driven decision making.

> **Note:** This project is part of an engineering internship. All proprietary production data remains confidential and is not included in this repository.

---

## Objectives

- Analyze historical production data
- Perform data preprocessing and feature engineering
- Explore relationships between process variables
- Develop and compare machine learning models
- Evaluate model performance
- Deploy the best-performing model
- Build a real-time dashboard for monitoring and prediction

---
## Project Structure

```text
tsp-fertilizer-ml/
│
├── .github/                     # GitHub workflows (CI/CD)
│   └── workflows/
│
├── data/
│   ├── raw/                     # Original production data (not tracked)
│   ├── processed/               # Cleaned and transformed datasets
│   ├── external/                # External/reference datasets
│   └── sample/                  # Sample data for testing/demo
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── validation.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── evaluate.py
│   │   └── tuning.py
│   │
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── pages/
│   │   └── components/
│   │
│   ├── deployment/
│   │   ├── api.py
│   │   └── inference.py
│   │
│   ├── visualization/
│   │   └── plots.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── helpers.py
│
├── models/                      # Saved trained models
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── final_report.pdf
│
├── tests/                       # Unit tests
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── docs/                        # Documentation
│
├── config/
│   └── config.yaml
│
├── requirements.txt
├── .gitignore
├── README.md
├── LICENSE
└── Dockerfile
```

## Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Model Deployment
8. Real-Time Dashboard

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- FastAPI *(optional)*
- Docker *(optional)*
- PostgreSQL / SQL
- Git & GitHub

---

## Machine Learning Pipeline
          Production Data
                │
                ▼
          Data Cleaning
                │
                ▼
          Feature Engineering
                │
                ▼
          Model Training
                │
                ▼
          Model Evaluation
                │
                ▼
          Deployment
                │
                ▼
          Real-Time Dashboard


---

## Current Status

- [ ] Understand production process
- [ ] Explore dataset
- [ ] Data preprocessing
- [ ] Exploratory Data Analysis
- [ ] Feature engineering
- [ ] Baseline model
- [ ] Model comparison
- [ ] Hyperparameter tuning
- [ ] Model deployment
- [ ] Dashboard development
- [ ] Documentation

---

## Notes

This repository contains only development code and documentation. No confidential industrial datasets or sensitive production information are included.

---

## License

This project is intended for educational and internship purposes.
