
# Hospital LOS Prediction + XAI

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate   # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset in `data/hospital_los.csv`. Ensure it has a numeric column named `LOS`.

## Run Training

```bash
python hospital_los_predict.py --data data/hospital_los.csv --target LOS --output out/
```

- Model is saved in `out/model.joblib`
- Evaluation metrics are in `out/evaluation.txt`

## Predict New Data

You can later load the model in Python:
```python
import joblib
model = joblib.load("out/model.joblib")
preds = model.predict(new_data)
```
