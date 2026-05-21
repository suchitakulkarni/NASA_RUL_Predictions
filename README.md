# NASA Turbofan Engine Remaining Useful Life (RUL) Prediction

Note:  Note: s6 and s10 from DEFAULT_SENSORS_TO_DROP don't appear as red because they normalise to std=1 by construction (they vary within clusters, just not in a degradation-correlated way). Those two are
  domain-knowledge exclusions, not variance-based ones — the plot can't capture that distinction, which is worth noting in the portfolio narrative.

# Run comamnds

  # 1. Train the joint model (if not already done)
  python train.py --mode joint --trials 100 --cap 125 --window 5 10

  # 2. Phase 5 — conformal calibration
  python calibrate.py --alpha 0.1 --cal-frac 0.2 --cap 125 --window 5 10

  # 3. Phase 6 — evaluation, plots, metrics
  python run_evaluation.py --cap 125 --window 5 10 --n-units 5 --lead-time 30 \
      --cost-unplanned 100000 --cost-planned 20000

  # 4 you should also train separate models with
     python train.py --mode separate --trials 100 --cap 125 --window 5 10 
## Project Summary
This project predicts the Remaining Useful Life (RUL) of turbofan engines using NASA's C-MAPSS dataset. Accurate RUL predictions enable predictive maintenance to reduce downtime and costs in aviation and manufacturing.

## Dataset
- NASA C-MAPSS dataset with multi-sensor time series data of engine degradation.
- Includes training and test sets with operational settings and sensor readings.

## Methodology
- Data cleaning and feature engineering to extract degradation patterns.
- Applied regression models: Random Forest, Gradient Boosting.
- Evaluated using RMSE and MAE metrics.

## Results
- Visualization of predicted vs actual RUL shown below.

![Predicted vs Actual RUL](images/final_predictions.png)

## How to Run
1. Clone repo  
2. Create environment: `pip install -r requirements.txt`  
3. Run `notebooks/RUL_timeseries_XGBoost.ipynb` for detailed analysis and model training. You can run this via google colab.
4. There is also a modular framework, if you have enough computing power. Run python main.py to run locally without notebook interface
5. Finally there is a streamlit interface to display results `run streamlit run app.py`

## Future Work
- Incorporate deep learning models (LSTM/GRU) for sequence modeling.
- Implement uncertainty quantification.

## Technologies
Python, pandas, scikit-learn, matplotlib, Jupyter Notebook

---

Feel free to reach out or check my portfolio: [suchitakulkarni.github.io](https://suchitakulkarni.github.io)

