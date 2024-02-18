# Deep Forecasting

## Description
This project demontrates the power of deep learning applied to forecasting use-cases by engineering a model based on Google Research's 2023 Time Series Mixer architecture.

A demonstration of the model's capacity to simultaneously forecast a large number of parallel time series' is included using data derived from an open ![dataset](https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy/about_data) that records Iowa liquor sales.

![Model](https://img.shields.io/badge/Neural_Network_Architecture-Complete-green)
![Utilities](https://img.shields.io/badge/Utilities-Complete-green)
![Demo Data](https://img.shields.io/badge/Demo_Data-Complete-green)
![Demo](https://img.shields.io/badge/Demo-Complete-green)

## Installation
While this is a small project not published to PyPi, it is still fully functional to learn about the TS Mixer architecture and use it for forecasting.

1. Clone this repo and cd to it

`git clone https://github.com/A-J-V/deep_forecasting.git`

`cd deep_forecasting`

2. Create and activate a virtual environment, then install the requirements using the provided requirements.txt file. Note that you may need a different version of Pytorch depending on your GPU configuration!

`python3 -m venv deep-forecasting`

On Unix: `source deep-forecasting/bin/activate`

On Windows: `deep-forecasting\Scripts\activate.bat`

`pip install -r requirements.txt`

## Usage and Demo
This repo can be used in parallel time series forecasting. The model and utilities are included in their respective py files, and __main__.py and the dataset in the assets folder provide a working example.

The example dataset is a 100 parallel time series' representing the liquor revenues in the counties of Iowa over the last few years. Each observation is one month. The last three features are sin/cos encoded periodicity and the time step.

In __main__.py, we're setting our lookback to 12 and forecast to 6. This indicates that we want to forecast the next 6 time steps by taking the previous into consideration (by using them as features).

The utils.py file includes a few helpful utilities. A good practice in forecasting in deep learning, just as it often is in classical statistics, is to render the data stationary and normalize it. We can instantiate a TSManager object and pass it the dataset as well as how many features are auxiliary. It can then be used to automatically take the first difference of all time series in the dataframe as well as store the relevant information to also invert the differencing so that we can see our forecasting back on the natural scale.

As a quick example, here is a plot of 5 counties in the dataset over time prior to being processed.

[placeholder for example of before]

And here are the same 5 time series of the counties after being processed by TSManager. They're now stationary and on the same scale.

[placeholder for example of after]

With the data ready for training, we use a couple helper functions, get_tsds() and get_dataloaders(), and can instantiate the model. The TSM object contains both a Pytorch model and some added functionality for convenience.

We can train it just by calling it's train() method and passing in the relevant arguments. It also has a loss_curve() method and score() method to use as a training diagnostic and performance check.

In this example, despite using only the time series data and their periodicities, and not allowing for tuning of hidden units, the model achieves a ~50% boost in performance over a naive model. With more careful feature engineering and including additional functionality, such as the ability for the model to accept static covariates specific to certain time series', the abilities of this model can likely expand.

Forecasting is hard, and this model is imperfect. But considering that it forecasts all 100 counties in the state simultaneously with minimal headache, it seems like quite the success. Check it's forecasts on the first few counties in the state.
[placeholder for forecast images]

