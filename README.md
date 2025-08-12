# Deep Forecasting

## Description
This package contains a powerful forecasting model able to handle many parallel time-series at once with the ability to handle clusters/hierarchies natively.

A demonstration of the model's capacity to simultaneously forecast a large number of parallel time series is included using data derived from an open [dataset](https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy/about_data) that records Iowa liquor sales.

![Model](https://img.shields.io/badge/Neural_Network_Architecture-Complete-green)
![Utilities](https://img.shields.io/badge/Utilities-Complete-green)
![Demo Data](https://img.shields.io/badge/Demo_Data-Complete-green)
![Demo](https://img.shields.io/badge/Demo-Complete-green)

## Installation
While this is not yet published to PyPi, it is fully functional and can be used for forecasting.

Install directly from GitHub using pip

`pip install git+https://github.com/A-J-V/deep_forecasting.git@master`

## Usage and Demo
This repo can be used in parallel time series forecasting. The model and utilities are included in their respective py files, and __main__.py and the dataset in the assets folder provide a working example.

The example dataset is a 100 parallel time series representing the liquor revenues in the counties of Iowa over the last few years. Each observation is one month. The last three features are sin/cos encoded periodicity and the time step.

In __main__.py, we're setting our lookback to 12 and forecast to 6. This indicates that we want to forecast the next 6 time steps by taking the previous into consideration (by using them as features).

The utils.py file includes a few helpful utilities. A good practice in forecasting in deep learning, just as it often is in classical statistics, is to render the data stationary and normalize it. We can instantiate a TSManager object and pass it the dataset as well as how many features are auxiliary. It can then be used to automatically take the first difference of all time series in the dataframe as well as store the relevant information to also invert the differencing so that we can see our forecasting back on the natural scale.

As a quick example, here is a plot of 5 counties in the dataset over time prior to being processed.

<p align="center">
  <img src="https://github.com/A-J-V/deep_forecasting/assets/72227828/e444d46c-9b79-43f5-9404-da4faf4baf77" alt="Five county time series before processing" width="500" height="300">
  <br>
  <em>Prior to any processing, county revenues are on very different scales and they are not stationary.</em>
</p>

And here are the same 5 time series of the counties after being processed by TSManager. They're now stationary and on the same scale.

<p align="center">
  <img src="https://github.com/A-J-V/deep_forecasting/assets/72227828/160d4a66-5908-451b-b44e-13935958cc07" alt="Five county time series after processing" width="500" height="300">
  <br>
  <em>After processing, county revenues are stationary and on the same scale.</em>
</p>

With the data ready for training, we use a couple helper functions, get_tsds() and get_dataloaders(), and can instantiate the model. The TSM object contains both a Pytorch model and some added functionality for convenience.

We can train it just by calling it's train() method and passing in the relevant arguments. It also has a loss_curve() method and score() method to use as a training diagnostic and performance check.

In this example, despite using only the time series data and their periodicities, and not allowing for tuning of hidden units, the model achieves a ~50% boost in performance over a naive model. With more careful feature engineering and including additional functionality, such as the ability for the model to accept static covariates specific to certain time series, the abilities of this model can likely expand.

This model is imperfect, but considering that it's a quick and dirty forecast of all 100 counties with minimal effort, it isn't bad at all.

<table>
  <tr>
    <td><img src="https://github.com/A-J-V/deep_forecasting/assets/72227828/a71b2482-9c66-4ec0-b351-cb193c6054b8" /></td>
    <td><img src="https://github.com/A-J-V/deep_forecasting/assets/72227828/c913b7ec-a1eb-49c3-ad4c-17662a3b90c0" /></td>
    <td><img src="https://github.com/A-J-V/deep_forecasting/assets/72227828/e198461a-31c5-4a56-802f-988d2bc2a258" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/A-J-V/deep_forecasting/assets/72227828/7e67faed-f12c-4c5c-afc6-6aa60f7d371e" /></td>
    <td><img src="https://github.com/A-J-V/deep_forecasting/assets/72227828/391dd5bd-f182-4679-b4db-faf25479c137" /></td>
    <td><img src="https://github.com/A-J-V/deep_forecasting/assets/72227828/42ad9a5b-81df-410a-a69d-b285e30cce2a" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/A-J-V/deep_forecasting/assets/72227828/aaf335ef-8e61-4b83-9aec-22a9085b26d6" /></td>
    <td><img src="https://github.com/A-J-V/deep_forecasting/assets/72227828/4357f8a8-4973-445b-ade1-cbecffd335bd" /></td>
    <td><img src="https://github.com/A-J-V/deep_forecasting/assets/72227828/49016f04-f9be-41d7-aa26-301f110da847" /></td>
  </tr>
</table>

