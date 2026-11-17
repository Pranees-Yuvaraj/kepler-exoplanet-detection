# KEPLER — Clean ML Dashboard

A simple, professional sidebar-driven Streamlit application. No FastAPI, React or 3D dependencies.

## Windows
Double-click `start_windows.bat` (Python 3.10+ required).

## Manual
```
pip install -r requirements.txt
streamlit run app.py
```

## Data
Download the Kepler-derived labeled time series dataset from https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data . Upload `exoTrain.csv` and `exoTest.csv` in Data Library. No bundled synthetic dataset.

## Limitations
This is an educational Random Forest baseline using summary statistics, not a confirmed planet discovery pipeline. Data labels are 1 = non-host, 2 = host. The dataset has significant class imbalance. The labeled Kaggle CSV is not raw NASA FITS data.
