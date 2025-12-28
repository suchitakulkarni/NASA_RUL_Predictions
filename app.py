import streamlit as st
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import math

#1. Coverage (PICP – Prediction Interval Coverage Probability)
#Check how often the true value falls within your predicted interval (e.g. between the 5th and 95th percentile).

def picp(y_true, q_lower, q_upper):
    within_interval = np.logical_and(y_true >= q_lower, y_true <= q_upper)
    return np.mean(within_interval)  # Should ideally be close to 0.90 for [0.05, 0.95]

#2. Interval Width (MPIW – Mean Prediction Interval Width)
#How wide are your prediction intervals? Narrower intervals are better if coverage is still acceptable.

def mpiw(q_lower, q_upper):
    return np.mean(q_upper - q_lower)

#3. Quantile Loss (Pinball Loss)
#This directly evaluates how well your model estimates each quantile.

def quantile_loss(y_true, y_pred, q):
    return np.mean(np.maximum(q * (y_true - y_pred), (q - 1) * (y_true - y_pred)))

def plot_rul_predictions_per_unit(unit_preds, unit_truths, quantiles=[0.05, 0.5, 0.95],
                                  max_units_per_figure=6, height=800):
    """
    Plot RUL predictions per unit with confidence intervals using Plotly for Streamlit

    Args:
        unit_preds: Dictionary with unit_id as key and predictions array as value
        unit_truths: Dictionary with unit_id as key and true RUL values as value
        quantiles: List of quantiles [lower, median, upper]
        max_units_per_figure: Maximum number of units to plot per figure
        height: Figure height in pixels
    """
    units = list(unit_preds.keys())
    n_units = len(units)
    n_figures = math.ceil(n_units / max_units_per_figure)

    for fig_idx in range(n_figures):
        start_idx = fig_idx * max_units_per_figure
        end_idx = min(start_idx + max_units_per_figure, n_units)
        units_subset = units[start_idx:end_idx]
        n_units_subplot = len(units_subset)

        # Calculate subplot grid
        n_cols = min(3, n_units_subplot)
        n_rows = math.ceil(n_units_subplot / n_cols)

        # Create subplot titles
        subplot_titles = [f'Unit {unit_id}' for unit_id in units_subset]

        # Create subplots with more vertical spacing to prevent title overlap
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.12,  # Increased from 0.08
            horizontal_spacing=0.08
        )

        for i, unit_id in enumerate(units_subset):
            # Calculate row and column for this subplot
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1

            # Get predictions and truths for this unit
            preds = np.array(unit_preds[unit_id])  # Shape: [n_sequences, n_quantiles]
            truths = np.array(unit_truths[unit_id])  # Shape: [n_sequences]

            # Create time axis (sequence index for this unit)
            time_steps = np.arange(len(truths))

            # Extract quantiles
            lower_pred = preds[:, 0]  # 5th percentile
            median_pred = preds[:, 1]  # 50th percentile (median)
            upper_pred = preds[:, 2]  # 95th percentile

            # Add confidence interval (fill between)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([time_steps, time_steps[::-1]]),
                    y=np.concatenate([upper_pred, lower_pred[::-1]]),
                    fill='toself',
                    fillcolor='rgba(173, 216, 230, 0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True if i == 0 else False,
                    name=f'{int(quantiles[0] * 100)}-{int(quantiles[2] * 100)}% CI',
                    legendgroup='ci'
                ),
                row=row, col=col
            )

            # Add median predictions
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=median_pred,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Predicted RUL (Median)',
                    showlegend=True if i == 0 else False,
                    legendgroup='pred',
                    hovertemplate='<b>Predicted RUL</b><br>' +
                                  'Time Step: %{x}<br>' +
                                  'RUL: %{y:.1f}<br>' +
                                  '<extra></extra>'
                ),
                row=row, col=col
            )

            # Add true RUL
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=truths,
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name='True RUL',
                    showlegend=True if i == 0 else False,
                    legendgroup='true',
                    hovertemplate='<b>True RUL</b><br>' +
                                  'Time Step: %{x}<br>' +
                                  'RUL: %{y:.1f}<br>' +
                                  '<extra></extra>'
                ),
                row=row, col=col
            )

            # Calculate final error for annotation
            final_true = truths[-1]
            final_pred_median = median_pred[-1]
            final_error = abs(final_true - final_pred_median)

            # Calculate the subplot position for annotation reference
            subplot_num = (row - 1) * n_cols + col
            xref = 'x domain' if subplot_num == 1 else f'x{subplot_num} domain'
            yref = 'y domain' if subplot_num == 1 else f'y{subplot_num} domain'

            # Add annotation with final error
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref=xref,
                yref=yref,
                text=f'Final Error: {final_error:.1f}',
                showarrow=False,
                bgcolor='rgba(245, 222, 179, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                font=dict(size=10),
                xanchor='left',
                yanchor='top'
            )

        # Update layout with improved spacing
        fig.update_layout(
            #title=dict(
            #    text=f'RUL Predictions per Unit (Figure {fig_idx + 1}/{n_figures})',
            #    x=0.5,
            #    font=dict(size=16, color='black')
            #),
            height=height,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(t=100, b=80, l=60, r=150)  # Increased top and bottom margins
        )

        # Update subplot title styling to prevent overlap
        fig.update_annotations(font=dict(size=12))  # Slightly smaller subplot titles

        # Update axes labels
        for i in range(1, n_rows * n_cols + 1):
            fig.update_xaxes(title_text='Time Step (Sequence Index)', row=(i - 1) // n_cols + 1,
                             col=(i - 1) % n_cols + 1)
            fig.update_yaxes(title_text='RUL', row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)


# Alternative function that returns the figure instead of displaying it
def create_rul_predictions_figure(unit_preds, unit_truths, quantiles=[0.05, 0.5, 0.95],
                                  max_units_per_figure=6, height=800, figure_index=0):
    """
    Create a single RUL predictions figure (useful for more control over display)

    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    units = list(unit_preds.keys())
    n_units = len(units)
    n_figures = math.ceil(n_units / max_units_per_figure)

    start_idx = figure_index * max_units_per_figure
    end_idx = min(start_idx + max_units_per_figure, n_units)
    units_subset = units[start_idx:end_idx]
    n_units_subplot = len(units_subset)

    # Calculate subplot grid
    n_cols = min(3, n_units_subplot)
    n_rows = math.ceil(n_units_subplot / n_cols)

    # Create subplot titles
    subplot_titles = [f'Unit {unit_id}' for unit_id in units_subset]

    # Create subplots with more vertical spacing to prevent title overlap
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,  # Increased from 0.08
        horizontal_spacing=0.08
    )

    for i, unit_id in enumerate(units_subset):
        # Calculate row and column for this subplot
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        # Get predictions and truths for this unit
        preds = np.array(unit_preds[unit_id])  # Shape: [n_sequences, n_quantiles]
        truths = np.array(unit_truths[unit_id])  # Shape: [n_sequences]

        # Create time axis (sequence index for this unit)
        time_steps = np.arange(len(truths))

        # Extract quantiles
        lower_pred = preds[:, 0]  # 5th percentile
        median_pred = preds[:, 1]  # 50th percentile (median)
        upper_pred = preds[:, 2]  # 95th percentile

        # Add confidence interval (fill between)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time_steps, time_steps[::-1]]),
                y=np.concatenate([upper_pred, lower_pred[::-1]]),
                fill='toself',
                fillcolor='rgba(173, 216, 230, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True if i == 0 else False,
                name=f'{int(quantiles[0] * 100)}-{int(quantiles[2] * 100)}% CI',
                legendgroup='ci'
            ),
            row=row, col=col
        )

        # Add median predictions
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=median_pred,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Predicted RUL (Median)',
                showlegend=True if i == 0 else False,
                legendgroup='pred',
                hovertemplate='<b>Predicted RUL</b><br>' +
                              'Time Step: %{x}<br>' +
                              'RUL: %{y:.1f}<br>' +
                              '<extra></extra>'
            ),
            row=row, col=col
        )

        # Add true RUL
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=truths,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='True RUL',
                showlegend=True if i == 0 else False,
                legendgroup='true',
                hovertemplate='<b>True RUL</b><br>' +
                              'Time Step: %{x}<br>' +
                              'RUL: %{y:.1f}<br>' +
                              '<extra></extra>'
            ),
            row=row, col=col
        )

        # Calculate final error for annotation
        final_true = truths[-1]
        final_pred_median = median_pred[-1]
        final_error = abs(final_true - final_pred_median)

        # Calculate the subplot position for annotation reference
        subplot_num = (row - 1) * n_cols + col
        xref = 'x domain' if subplot_num == 1 else f'x{subplot_num} domain'
        yref = 'y domain' if subplot_num == 1 else f'y{subplot_num} domain'

        # Add annotation with final error
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref=xref,
            yref=yref,
            text=f'Final Error: {final_error:.1f}',
            showarrow=False,
            bgcolor='rgba(245, 222, 179, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            font=dict(size=10),
            xanchor='left',
            yanchor='top'
        )

    # Update layout with improved spacing
    fig.update_layout(
        title=dict(
            text=f'RUL Predictions per Unit (Figure {figure_index + 1}/{n_figures})',
            x=0.5,
            font=dict(size=16, color='black')
        ),
        height=height,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        margin=dict(t=100, b=80, l=60, r=150)  # Increased top and bottom margins
    )

    # Update subplot title styling to prevent overlap
    fig.update_annotations(font=dict(size=12))  # Slightly smaller subplot titles

    # Update axes labels
    for i in range(1, n_rows * n_cols + 1):
        fig.update_xaxes(title_text='Time Step (Sequence Index)', row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)
        fig.update_yaxes(title_text='RUL', row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)

    return fig



true_color = 'black'
pred05_col = 'rgba(150, 150, 150, 0.3)'
pred95_col = 'rgba(150, 150, 150, 0.3)'
st.set_page_config(page_title="NASA engines RUL predictions", layout="wide")
st.header("NASA engines RUL predictions")
with st.expander("Dataset"):
    url = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
    notebook_repo = "https://github.com/suchitakulkarni/DataScience/blob/main/NASA_RUL_predictions/notebooks/RUL_timeseries_XGBoost.ipynb"
    code_repo = "https://github.com/suchitakulkarni/DataScience/tree/main/NASA_RUL_predictions"
    st.write("The data has been provided by NASA and is available [here](%s)" % url)
    st.write("This project uses only one of the four available datasets")
    st.write("Software stack: pandas, numpy, plotly, sklearn, XGBoost, Optuna")
    st.write("Final notebook available here: %s"%(notebook_repo))
    st.write("Final code available here: %s"%(code_repo))
st.subheader("Final RUL predictions and comparisons with true values")
df_predictions_over_time = pd.read_csv('./final_results.txt')
df_per_unit = pd.read_csv('./RUL_per_unit.txt')

unit_no = df_predictions_over_time['unit_no'].unique()
RUL05 = df_predictions_over_time['RUL_pred05'].values
RUL5 = df_predictions_over_time['RUL_pred5'].values
RUL95 = df_predictions_over_time['RUL_pred95'].values
RUL_true = df_predictions_over_time['RUL_true'].values

rul_dict = {}
rul_truth_dict = {}
for unit in unit_no:
    if unit == 1: continue
    RUL05 = df_predictions_over_time[df_predictions_over_time['unit_no'] == unit]['RUL_pred05'].values
    RUL5 = df_predictions_over_time[df_predictions_over_time['unit_no'] == unit]['RUL_pred5'].values
    RUL95 = df_predictions_over_time[df_predictions_over_time['unit_no'] == unit]['RUL_pred95'].values
    RUL_true = df_predictions_over_time[df_predictions_over_time['unit_no'] == unit]['RUL_true'].values
    vals = list(zip(RUL05, RUL5, RUL95))
    rul_dict[unit] = vals
    rul_truth_dict[unit] = RUL_true

selected_quantiles = st.multiselect("Select quantiles to display:",
                    options = ["5%", "50%", "95%",  "Interval (5%-95%)"],
                    default=["50%",  "Interval (5%-95%)"]
                    )
fig = go.Figure()
if "50%" in selected_quantiles:

    x = np.arange(len(df_per_unit['RUL']))
    y = df_per_unit['RUL_pred5']
    y_true = df_per_unit['RUL']
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='50% quantile', line=dict(color=true_color, width=2)))
    fig.add_trace(
        go.Scatter(x=x, y=y_true, mode='lines', name='True RUL', line=dict(dash='dash', color=true_color, width=2)))

if "95%" in selected_quantiles:
    x = np.arange(len(df_per_unit['RUL']))
    y = df_per_unit['RUL_pred95']
    y_true = df_per_unit['RUL']
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='95% quantile', line=dict(color=pred95_col, width=2)))
    fig.add_trace(
        go.Scatter(x=x, y=y_true, mode='lines', name='True RUL', line=dict(dash='dash', color=true_color, width=2)))

if "5%" in selected_quantiles:
    x = np.arange(len(df_per_unit['RUL']))
    y = df_per_unit['RUL_pred05']
    y_true = df_per_unit['RUL']
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='5% quantile', line=dict(color=pred05_col, width=2)))
    fig.add_trace(
        go.Scatter(x=x, y=y_true, mode='lines', name='True RUL', line=dict(dash='dash', color=true_color, width=2)))

if "Interval (5%-95%)" in selected_quantiles:
    x = np.arange(len(df_per_unit['RUL']))
    y05 = df_per_unit['RUL_pred05']
    y95 = df_per_unit['RUL_pred95']
    y5 = df_per_unit['RUL_pred5']
    y_true = df_per_unit['RUL']

    fig.add_trace(go.Scatter(x=x, y=y5, mode='lines', name='5% quantile'))
    fig.add_trace(go.Scatter(x=x, y=y05, mode='lines', name='5% quantile', fill='tonexty',
                             fillcolor='rgba(150, 150, 150, 0.3)', line=dict(color=pred05_col, width=2)))
    fig.add_trace(go.Scatter(x=x, y=y95, mode='lines', name='5% quantile', fill='tonexty',
                             fillcolor='rgba(150, 150, 150, 0.3)', line=dict(color=pred95_col, width=2)))
    fig.add_trace(
        go.Scatter(x=x, y=y_true, mode='lines', name='True RUL', line=dict(dash='dash', color=true_color, width=2)))
# Add a rectangle as a frame
fig.add_shape(
    type="rect",
    x0=0, x1=1, y0=0, y1=1,  # full area of the plot
    xref="paper", yref="paper",  # use the paper coordinate system
    line=dict(color="black", width=1)
)

fig.update_layout(
    title='Quantile Regression Predictions',
    xaxis_title='Unit no',
    yaxis_title='RUL',
    legend=dict(orientation='h'),
    width=800,
    height=800,
    margin=dict(l=40, r=40, t=40, b=40)
)

st.plotly_chart(fig, use_container_width=True)
st.write(f'**Evaluation matrices**')
st.write(f'Prediction Interval Coverage Probability = {picp(df_per_unit['RUL'], df_per_unit['RUL_pred05'], df_per_unit['RUL_pred95'])}')
st.write(f'Mean Prediction Interval Width = {mpiw(df_per_unit['RUL_pred05'], df_per_unit['RUL_pred95'])}')
st.write(f'Quantile Loss (Pinball Loss) for q = 0.5 = {quantile_loss(df_per_unit['RUL'], df_per_unit['RUL_pred5'], 0.5)}')
st.write(f'Quantile Loss (Pinball Loss) for q = 0.05 = {quantile_loss(df_per_unit['RUL'], df_per_unit['RUL_pred05'], 0.05)}')
st.write(f'Quantile Loss (Pinball Loss) for q = 0.95 = {quantile_loss(df_per_unit['RUL'], df_per_unit['RUL_pred95'], 0.95)}')

st.subheader("RUL predictions per unit")
plot_rul_predictions_per_unit(rul_dict, rul_truth_dict)
