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