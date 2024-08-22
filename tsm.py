import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Callable

from model import TSMixer


class TSM:
    """ A model class used to build, train, and predict with a TSMixer architecture. """

    def __init__(self,
                 lookback: int,
                 features: int,
                 forecast: int,
                 blocks: int,
                 dropout: float = 0.0,
                 num_aux: int = 0,
                 device: str = 'cpu',
                 ) -> None:
        """
        Initialize the model.
        Data parameters must match the DataLoaders that will be passed in for training.

        :param lookback: The number of prior timesteps to use as features for predictions.
        :type lookback: int
        :param features: The number of columns in the data that represent type series.
        :type features: int
        :param forecast: The number of future timesteps to forecast when predicting.
        :type forecast: int
        :param blocks: The number of Time Series Mixer blocks to use in the model. More blocks means a deeper model.
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
        self.features = features
        self.forecast = forecast
        self.blocks = blocks
        self.dropout = dropout
        self.device = device
        self.model = TSMixer(lookback, features, forecast, blocks, dropout, num_aux)
        self.model.to(device)
        self.logs = {'epoch': [],
                     'train_loss': [],
                     'test_loss': [],
                     }
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
                obs: torch.Tensor,
                ) -> np.ndarray:
        """
        Feed the observation through the model and returns prediction.

        Expects dimension of obs to be (batch, features, timesteps). Three
        shall be the number of dimensions. Five is right out.

        By default, squeezes and converts the prediction to numpy before returning it.
        """

        # If the passed obs is only 2 dimension, it probably isn't a batch, so unsqueeze it.
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        elif obs.dim() != 3:
            raise ValueError(f"Expected input to be 3 dimensions, received {obs.dim()}.")

        self.model.eval()
        with torch.no_grad():
            pred = self.model(obs.to(self.device))
            pred = pred.squeeze().cpu().numpy()

        return pred

    def score(self,
              obs,
              true,
              ):
        """ Score the model by taking the predictors and true values and returning the model's MASE """
        pred = self.predict(obs)

        if not isinstance(true, np.ndarray):
            true = true.cpu().numpy()
        MASE = self.MASE_(pred, true)
        return MASE

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
            raise Exception(f"Device {device} not recognized!")

    def MASE_(self,
              pred: np.ndarray,
              true: np.ndarray,
              naive_error: Optional[float] = None,
              ) -> float:
        """ Calculate the Mean Absolute Scaled Error.

        Given the model's predictions, the true values, and an optional error calculated from an alternative method,
        return the model's MAE scaled to the alternative method's MAE. This can be interpreted the same way as "lift"
        and gives a strong intuitive indication of how much better this model is to the alternative.

        :param pred: The model's predictions.
        :type pred: np.ndarray
        :param true: The true values.
        :type true: np.ndarray
        :param naive_error: The MAE of the predictions made by an alternative method.
        :type naive_error: Optional[float]
        :return: The Mean Absolute Scaled Error.
        :rtype: float
        """

        # Calculate the MAE for each prediction, then get the average MAE over the batch of predictions.
        error = pred - true
        abs_error = np.abs(error)
        MAE = abs_error.sum() / (true.shape[0] * true.shape[1])

        # If the naive_error from some other method is provided, we will use that. Should be MSE or
        if naive_error is not None:
            abs_naive_error = naive_error
        else:
            shifted = true[1:, :]
            clipped_original = true[:-1, :]
            naive_error = shifted - clipped_original
            abs_naive_error = np.abs(naive_error)

        MANE = abs_naive_error.sum() / ((true.shape[0] - 1) * true.shape[1])

        MASE = MAE / MANE

        return MASE

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
        test_acc = 0

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
