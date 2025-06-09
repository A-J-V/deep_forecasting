import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch

import deep_forecasting.utils
from .model import HierarchicalTimeSeriesMixer
from .utils import TSManager


class HTSM:
    """ A model class used to build, train, and predict with a group-aware time series network. """

    def __init__(self,
                 lookback: int,
                 dataset: deep_forecasting.utils.TSDS,
                 hidden_features: int,
                 forecast: int,
                 blocks: int,
                 dropout: float = 0.0,
                 num_aux: int = 0,
                 device: str = 'cpu',
                 final_global_mixer: bool = False,
                 ) -> None:
        """
        Data parameters must match the DataLoaders that will be passed in for training.

        :param lookback: The number of prior timesteps to use as features for predictions.
        :type lookback: int
        :param dataset: The dataset to use as a schema for the model.
        :type dataset: TSDS
        :param hidden_features: The hidden unit size inside the mixer operations.
        :type hidden_features: int
        :param forecast: The number of future timesteps to forecast.
        :type forecast: int
        :param blocks: The number of grouped mixer blocks to use in the model. Recommended to try 1 or 2.
        :type blocks: int
        :param dropout: The proportion of weights to drop in the parameters during training (for regularization).
                        Note that empirically, setting unusually high dropout often works well with this model.
        :type dropout: float
        :param num_aux: The number of auxiliary features in the data.
        :type num_aux: int
        :param device: Which device to use with Pytorch. Tested using 'cpu' and 'cuda'.
        :type device: str
        """
        assert num_aux >= 0, "num_aux must be >= 0! Try using normalized timestep or sin/cos for periodicity."
        assert 0.0 <= dropout < 1.0, "dropout must fall within [0.0, 1.0)!"


        self.lookback = lookback
        self.group_registry = dataset.group_registry
        self.covariate_registry = dataset.covariate_registry
        self.aux_indices = dataset.aux_indices
        self.hidden_features = hidden_features
        self.forecast = forecast
        self.blocks = blocks
        self.dropout = dropout
        self.device = device

        self.model = HierarchicalTimeSeriesMixer(
            lookback=self.lookback,
            group_registry=self.group_registry,
            covariate_registry=self.covariate_registry,
            aux_indices=self.aux_indices,
            hidden_features=self.hidden_features,
            forecast=self.forecast,
            blocks=self.blocks,
            dropout=self.dropout,
            final_global_mixer=final_global_mixer,
        )

        self.model.to(device)
        self.logs = {'epoch': [],
                     'train_loss': [],
                     'test_loss': [],
                     }
        self.num_aux = num_aux
        self.device = device

    def train(self,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              epochs: int,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              verbose=False,
              ):

        """ Train the network! """

        for epoch in range(1, epochs + 1):

            train_loss = self.train_step_(model=self.model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          )

            test_loss = self.test_step_(model=self.model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        )

            self.logs['epoch'].append(epoch)
            self.logs['train_loss'].append(train_loss)
            self.logs['test_loss'].append(test_loss)

            if verbose:
                print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {test_loss:.4f} |')

    def predict(self,
                observations: np.ndarray,
                ) -> np.ndarray:
        """
        Feed the observations through the model and returns prediction.
        Expects dimension of observations to be (batch, time_steps, features).
        By default, squeezes and converts the prediction to numpy before returning it.
        NOTE: If this model was trained on pre-processed data using TSManager, consider using predict_scale() instead!

        :param observations: A batch of observations.
        :type observations: np.ndarray
        :return: A batch of predictions.
        :rtype: np.ndarray
        """

        # 1) Convert the numpy array to a torch tensor [features, time], then add batch dim -> [1, features, time]
        data_tensor = torch.from_numpy(observations).float().permute(1, 0).to(self.device)
        if data_tensor.dim() == 2:
            data_tensor = data_tensor.unsqueeze(0)  # [1, features, time]
        elif data_tensor.dim() != 3:
            raise ValueError(f"Expected tsds.df to yield 2D or 3D tensor; got {data_tensor.dim()}D.")

        if data_tensor.shape[-1] != self.lookback:
            raise ValueError(f"Expected `observations` to have a lookback of {self.lookback}; got {data_tensor[-1]}")

        # 2) Run the model on that lookback to get raw (transformed) forecasts of core series
        self.model.eval()
        with torch.no_grad():
            # model expects [B, S_total, lookback_len] and returns [B, num_core, forecast]
            raw_preds = self.model(data_tensor)  # [1, forecast, series]

        return raw_preds

    def predict_scale(self,
                      dataset: deep_forecasting.utils.TSDS,
                      manager: TSManager,
                      predictions_only: bool = True,
                      headers: bool = False
                      ) -> np.ndarray:
        """
        Feed the dataset through the model and returns forecasts on the original scale.

        This function requires a TSManager so that it can automatically convert the predictions to the original
        scale. The workflow of this would be:
        1) use TSManager to preprocess a dataframe.
        2) use that to build TSDS objects and train/validate the model.
        3) build a TSDS object of the entire dataset and pass it to this method for forecasting.

        By default, squeezes and converts the prediction to numpy before returning it.

        :param dataset: The entire dataset of time series' in TSDS format.
        :param manager: The TSManager that will be used to rescale the predictions.
        :param predictions_only: `True` returns predictions only, `False` returns all data with predictions at the end.
        :param headers: `True` returns a Pandas dataframe with column names, `False` returns a Numpy ndarray.

        :return: A batch of predictions converted to the original dataset's scale.
        :rtype: np.ndarray
        """

        # 1) Grab the full preprocessed DataFrame directly from tsds
        transformed_df = dataset.df

        # 2) Convert that DataFrame to a torch tensor [features, time], then add batch dim -> [1, features, time]
        data_tensor = torch.from_numpy(transformed_df.values).float().permute(1, 0).to(self.device)
        if data_tensor.dim() == 2:
            data_tensor = data_tensor.unsqueeze(0)  # [1, features, time]
        elif data_tensor.dim() != 3:
            raise ValueError(f"Expected tsds.df to yield 2D or 3D tensor; got {data_tensor.dim()}D.")

        # 3) Extract the "lookback" window from the rightmost time steps
        lookback_len = dataset.lookback
        lookback_input = data_tensor[:, :, -lookback_len:]  # [1, features, lookback_len]

        # 4) Run the model on that lookback to get raw (transformed) forecasts of core series
        self.model.eval()
        with torch.no_grad():
            # model expects [B, S_total, lookback_len] and returns [B, num_core, forecast]
            raw_preds = self.model(lookback_input)  # [1, forecast, series]
            raw_preds = raw_preds.permute(0, 2, 1)  # [1, series, forecast]

        # 5) Extract historical core‐series from data_tensor (drop covariates+aux)
        series_idx = dataset.group_indices         # e.g. [1,2,3,...]
        core_series = data_tensor[:, series_idx, :]  # [1, series, time]

        print(f"shape of core_series: {core_series.shape}")
        print(f"shape of raw_preds: {raw_preds.shape}")

        # 6) Concatenate historical core time series' with their forecasts along the time axis
        full_core = torch.cat([core_series, raw_preds], dim=2)  # [1, series, time+forecast]

        # 7) Move to NumPy 2D array and recover column names
        #    After squeeze(0), shape = [series, time+forecast] -> permute -> [time+forecast, series]
        data_with_preds = full_core.squeeze(0).permute(1, 0).cpu().numpy()
        # The reason we need the col names despite the method returning a Numpy array is that the TSManager needs the
        # names to be able to know how to invert each series correctly.
        prefix_pattern = r'^(g\d+[_.]|c\d+[_.]|aux_)'
        col_names = [re.sub(prefix_pattern, '', dataset.df.columns[i]) for i in series_idx]
        print(col_names)

        # 8) Convert to a Pandas DataFrame and invert the preprocessing to get everything back on the original scale.
        scaled_preds = pd.DataFrame(data_with_preds, columns=col_names)
        scaled_preds.columns = col_names
        scaled_preds = manager.invert_all(scaled_preds)

        # The predictions are now back in the original scale of the data. Return them according to predictions_only.
        if predictions_only:
            scaled_preds = scaled_preds.iloc[-self.forecast:, :]
        if headers:
            return scaled_preds
        return scaled_preds.to_numpy()

    def loss_curve(self):
        """ Print the train and test loss curve of this model over training """
        plt.plot(self.logs['epoch'], self.logs['train_loss'], color='blue')
        plt.plot(self.logs['epoch'], self.logs['test_loss'], color='red')
        plt.legend({'Training Loss': 'blue',
                    'Test Loss': 'red'
                    })
        plt.show()

    def to(self, device):
        """ Switch the device of the underlying Pytorch model to 'cpu' or 'cuda' """
        if device == 'cpu':
            self.model.to('cpu')
            self.device = 'cpu'
        elif device == 'cuda':
            self.model.to('cuda')
            self.device = 'cuda'
        else:
            raise Exception(f"Device {device} not supported! Try 'cpu' or 'cuda'.")

    def train_step_(self,
                    model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_fn: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    ) -> float:

        # Set the model to train mode
        model.train()

        # Set up the training loss and validation accuracy
        train_loss = 0

        for X, y in dataloader:
            # Send the data to the target device
            X, y = X.to(self.device), y.to(self.device)

            # 1. Forward pass
            y_predictions = model(X)

            # 2. Calculate the loss
            loss = loss_fn(y_predictions, y)
            train_loss += loss.item()

            # 3. Zero out the gradients
            optimizer.zero_grad()

            # 4. Backpropagation
            loss.backward()

            # 5. Update the parameters
            optimizer.step()

        # Adjust the metrics to be the per-batch averages
        train_loss /= len(dataloader)

        return train_loss

    def test_step_(self,
                   model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   ) -> float:

        # Set the model to evaluation mode
        model.eval()

        # Set up the training loss and validation accuracy
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                # Send the data to the target device
                X, y = X.to(self.device), y.to(self.device)

                # 1. Forward pass
                y_predictions = model(X)

                # 2. Calculate the loss
                loss = loss_fn(y_predictions, y)
                test_loss += loss.item()

        # Adjust the metrics to be the per-batch averages
        test_loss /= len(dataloader)

        return test_loss
