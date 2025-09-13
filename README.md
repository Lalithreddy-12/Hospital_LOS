
# Hospital LOS Prediction + XAI (Streamlit)

This project predicts **hospital length of stay (LOS)** using a trained machine learning model.  
It includes **interactive SHAP explainability, feature importance, fairness dashboards, and model evaluation**, making it presentation-ready.

## Features

- **Predict Patient LOS** with custom alerts for high-risk patients.
- **Interactive SHAP explainability** to understand feature contributions.
- **Global Feature Importance Dashboard**.
- **Fairness Report** by Gender and Age groups.
- **Model Evaluation Tab** showing overall performance and group-wise MAE.

## Setup (Local + GitHub)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Lalithreddy-12/Hospital_LOS.git
   cd hospital_los_project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate   # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place your dataset in `data/hospital_los.csv`. Ensure it has a numeric column named `LOS`.**
- dataset already provided in repo you can access directly.

## Run Training

```bash
python hospital_los_predict.py --data data/hospital_los.csv --target LOS --output out/
```

- Model is saved in `out/model.joblib`
- Evaluation metrics are in `out/evaluation.txt`

## Run the Streamlit App
```bash
streamlit run app.py
```

- Open the URL displayed in the terminal (usually http://localhost:8501).
