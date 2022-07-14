import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor


def calculate_error(predictions):
    """
    Calculate the absolute error associated with prediction intervals

    :param predictions: dataframe of predictions
    :return: None, modifies the prediction dataframe

    """
    predictions['absolute_error_lower'] = (predictions['lower'] - predictions["value"]).abs()
    predictions['absolute_error_upper'] = (predictions['upper'] - predictions["value"]).abs()

    predictions['absolute_error_interval'] = (predictions['absolute_error_lower'] + predictions['absolute_error_upper']) / 2
    predictions['absolute_error_mid'] = (predictions['mid'] - predictions["value"]).abs()

    predictions['in_bounds'] = predictions["value"].between(left=predictions['lower'], right=predictions['upper'])

    return predictions

def show_metrics(metrics):
    """
    Make a boxplot of the metrics associated with prediction intervals

    :param metrics: dataframe of metrics produced from calculate error 
    :return fig: plotly figure
    """
    percent_in_bounds = metrics['in_bounds'].mean() * 100
    metrics_to_plot = metrics[[c for c in metrics if 'absolute_error' in c]]

    # Rename the columns
    metrics_to_plot.columns = [column.split('_')[-1].title() for column in metrics_to_plot]

    # Create a boxplot of the metrics
    fig = px.box(
        metrics_to_plot.melt(var_name="metric", value_name='Absolute Error'),
        x="metric",
        y="Absolute Error",
        color='metric',
        title=f"Error Metrics Boxplots    In Bounds = {percent_in_bounds:.2f}%",
        height=800,
        width=1000,
        points=False,
    )

    # Create new data with no legends
    d = []

    for trace in fig.data:
        # Remove legend for each trace
        trace['showlegend'] = False
        d.append(trace)

    # Make the plot look a little better
    fig.data = d
    fig['layout']['font'] = dict(size=20)
    return fig

def confidence_interval_error(CI_DF):
    CI_DF["upper_error"] =  CI_DF.upper - CI_DF.value 
    CI_DF["lower_error"] =  CI_DF.value - CI_DF.lower
    lower_count = (CI_DF["upper_error"] < 0).sum().sum()
    upper_count = (CI_DF["lower_error"] < 0).sum().sum()

    count = lower_count + upper_count
    total_collum = CI_DF.shape[0]

    print ("Error percentage: {:0.2f} %".format(count/total_collum * 100))
    
def make_intervals_box_plot(CI_DF):
    predictions = calculate_error(data)
    metrics = predictions[['absolute_error_lower', 'absolute_error_upper', 'absolute_error_interval', 'absolute_error_mid', 'in_bounds']].copy()
    metrics.describe()
    error_plots = show_metrics(metrics)
    return error_plots