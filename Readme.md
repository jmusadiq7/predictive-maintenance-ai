# Jet Engine Predictive Maintenance (NASA C-MAPSS)

An end-to-end Machine Learning project to predict engine failure within a 30-cycle window using sensor data. 

Project Overview
This project uses the NASA C-MAPSS dataset to build a classification model. By analyzing sensor readings like static pressure (Ps30) and outlet temperatures (T50), the model flags engines that are within 30 cycles of a potential failure.

Key Results
- **Precision:** 94% (High confidence in failure alerts)
- **Accuracy:** 91% 
- **Model:** Random Forest Classifier with Rolling Mean features.

Features
- **Temporal Engineering:** Used 10-cycle rolling windows to smooth sensor noise.
- **RUL Calculation:** Manually derived Remaining Useful Life (RUL) labels from raw time-series data.
- **Feature Importance:** Identified Ps30 and T50 as the most critical indicators of engine degradation.

Visualizations
![Sensor Trend](path/to/your/trend_plot.png)
*Example of Ps30 sensor trend moving into the failure warning zone.*

Setup
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/predictive-maintenance-ai.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook in the `notebooks/` folder.

Additional API setup:
```bash
pip install fastapi uvicorn
```

API Usage
Start the FastAPI server:
```bash
python run.py
```

Example `/predict` request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": 20,
    "sensor_readings": {
      "setting1": 0.001,
      "setting2": 0.0001,
      "T24": 642.0,
      "T30": 1585.0,
      "T50": 1400.0,
      "P15": 21.6,
      "P30": 554.0,
      "Nf": 2388.0,
      "Nc": 9050.0,
      "Ps30": 47.5,
      "Phi": 522.0,
      "NRf": 2388.0,
      "NRc": 8130.0,
      "BPR": 8.4,
      "htBleed": 392,
      "W31": 39.0,
      "W32": 23.3
    }
  }'
```

The API keeps the latest 10 readings per `engine_id` to calculate rolling mean features. Send 10 readings for the same engine before expecting a prediction.

Alert System
When `/predict` returns `"prediction": "DANGER"`, the API writes an alert entry to:

```text
logs/alerts.log
```

Each alert line includes a UTC timestamp, `engine_id`, and prediction confidence. Safe predictions return `"alert": { "triggered": false }` and do not write to the alert log.
