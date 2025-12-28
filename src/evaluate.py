from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
from src.utils import RESULTS_DIR

def write_results_to_csv(uid, y_pred):
    df_preds = pd.DataFrame({'unit_no': uid_test,
                             'RUL_pred05': ypreds[0.05],
                             'RUL_pred5': ypreds[0.5],
                             'RUL_pred95': ypreds[0.95],
                             'RUL_true': y_test
                             })
    df_preds.to_csv('final_results.txt', index=False)
    return df_preds

def plot_results(df_preds, title="pred_vs_true"):
    import matplotlib.pyplot as plt

    RUL_per_unit = df_preds.groupby('unit_no').last().reset_index()
    RUL_per_unit = RUL_per_unit.merge(RUL_labels, on='unit_no').reset_index()
    RUL_per_unit = RUL_per_unit.sort_values(by=['RUL'])
    RUL_per_unit.to_csv('RUL_per_unit.txt', index=False)

    plt.figure(figsize=(10, 8))

    plt.plot(RUL_per_unit['RUL'], RUL_per_unit['RUL'], label='Perfect prediction', color='black', linestyle='--')
    plt.plot(RUL_per_unit['RUL'], RUL_per_unit['RUL_pred5'], label='Median (0.5)', color='blue')
    plt.fill_between(RUL_per_unit['RUL'], RUL_per_unit['RUL_pred05'], RUL_per_unit['RUL_pred95'], color='gray',
                     alpha=0.3, label='90% Interval')

    plt.legend()
    plt.xlabel('RUL true')
    plt.ylabel('RUL predicted')
    plt.title('Quantile Prediction vs True RUL')
    return fig