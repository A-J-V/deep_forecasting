import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.optim import Adam

from deep_forecasting import HTSM, TSManager, get_tsds, get_dataloaders
from deep_forecasting.utils import TSDS

# This is just a sample script of how the library works.

# Lookback sets how many time steps back in the past we want to consider as features in our forecast
lookback = 12

# Forecast is how many time steps into the future we want to forecast
forecast = 6

# The number of auxiliary features is the number of non-timeseries features we're including that apply to every
# time series in the data set. In this example, we have 3 which are sin(month), cosine(month), and timestep.
num_aux = 3

# Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the demo dataset. The prepared iowa liquor demo dataset is available on GitHub.
data = pd.read_csv("../demo_folder/iowa_liquor_demo_dataset.csv")
print(data.head())

# We're using a helper class to automatically detrend and normalize all of our time series data and store the info
# to be able to invert that process to get the forecasts back to the natural scale.
# Note that the three aux features are timestep (to learn linear trends), and sin/cos (to learn seasonality)
manager = TSManager(data)
processed_data = manager.transform_all(data)

# Load the processed data into Pytorch datasets using a function
train_ds, test_ds = get_tsds(processed_data,
                             lookback=lookback,
                             forecast=forecast,
                             train_prop=0.7,
                             )

# Put the datasets into Pytorch dataloaders using a function
train_loader, test_loader = get_dataloaders(train_ds, test_ds, train_batch_size=4, test_batch_size=4)

# Instantiate the model class. This contains both the Pytorch model and some helpful additional functionality.
model = HTSM(lookback=lookback,
             dataset=train_ds,
             hidden_features=24,
             forecast=forecast,
             blocks=1,
             dropout=0.7,
             num_aux=num_aux,
             device=device,
             final_global_mixer=False,
             )

# Choose the optimizer we'll be using.
optimizer = Adam(params=model.model.parameters(),
                 lr=0.001,
                 weight_decay=0.001,
                 )

# Train the model.
model.train(train_dataloader=train_loader,
            test_dataloader=test_loader,
            epochs=250,
            loss_fn=torch.nn.L1Loss(),
            optimizer=optimizer,
            verbose=True,
            )

# We can plot a loss curve as a simple diagnostic to see how training went
model.loss_curve()
print(f"Shape of an observation is: {test_ds[0][0].shape}")
print(f"The shape of all processed data is: {processed_data.shape}")

# Given a timeseries up until time t and the targets up to t + forecast, the model can score itself and
# report how much better it is than a naive model using Mean Absolute Scaled Error (MASE)
# NOTE that the .score function works best when passing it predictions from an alternative model to which we want
# our model's predictions compared.
#print(f"Model predictions are {100 * (1 - model.score(test_ds[0][0], test_ds[0][1])):.2f}% better than Naive Prediction error.")

# With no static data and only the periodicity and timesteps as auxiliary data, we can simultaneously forecast
# liquor revenues of all 100 counties in Iowa 6 months into the future with a ~50% improvement over naive methods.

#preds = model.predict(observations=processed_data.values[-lookback:, :])


# To get the forecasts back on the original scale, we can use predict_scale and pass the dataset and TSManager.
scaled_preds = model.predict_scale(dataset=TSDS(processed_data, lookback, forecast),
                                   manager=manager,
                                   predictions_only=False,
                                   headers=True,
                                   )

# View a few forecasts to see if they look reasonable.
for i in range(min(10, scaled_preds.shape[1])):
    # Extract a single complete time series
    single_series = scaled_preds.iloc[:, i]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 4))

    # Print debugging info
    print(f"Plotting series {i + 1}, data shape: {single_series.shape}")

    # Plot the time series with index as x-axis
    ax.plot(single_series.index, single_series.values)

    # Add the vertical line at the forecast boundary
    ax.axvline(x=scaled_preds.shape[0] - forecast, color='red', linestyle='--')

    # Add a title to identify which series we're looking at
    ax.set_title(f"Time Series {i + 1}")

    # Show the plot
    plt.show()
