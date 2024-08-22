import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class TSTransformer:
    """ Holds information for transforming and inverting a single time series. """

    def __init__(self) -> None:
        """ Given a numpy array of a time series, store info to normalize and denormalize."""
        self.mini = None
        self.maxi = None
        self.init_x = None
        self.normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        self.denormalize = lambda x: x * (self.maxi - self.mini) + self.mini

    def transform(self,
                  series: np.ndarray,
                  log=False,
                  ) -> np.ndarray:
        """
        De-trend and normalize the given series; store values to invert the transformation.

        :param series: A Numpy array representing a time series.
        :type series: np.ndarray
        :param log: A boolean value to choose whether to log the series prior to transformation.
        :type log: bool
        :return: Returns the transformed series that has been de-trended and normalized.
        :rtype: np.ndarray
        """

        # If needed, start by taking the logarithm of the series. This can help reduce variance in some cases.
        if log:
            series = np.log(series)

        # Since this transformation must be invertible, store the initial value at the start of the series.
        self.init_x = series[0]

        # Take the first difference of the series. This (usually) eliminates any trend, aiding learning.
        series = np.diff(series, 1)

        # Normalization also must be invertible, so we need to store the original min and max of the de-trended series.
        self.mini = series.min()
        self.maxi = series.max()

        # Finally, we normalize the series to reduce its values to the range [0, 1].
        series = self.normalize(series)

        # We need to add in an extra step at the end to keep the number of steps the same as the original
        series = np.insert(series, -1, series[-1])

        return series

    def invert(self,
               series: np.ndarray,
               exp=False,
               ) -> np.ndarray:
        """
        Invert the given series using the stored information from transform().
        Inverts the normalization and adds back in trend information.

        :param series: A Numpy array representing a time series.
        :type series: np.ndarray
        :param exp: A boolean value to choose whether to exponentiate the series after inversion.
        :type exp: bool
        :return: Returns the transformed series that has been de-trended and normalized.
        :rtype: np.ndarray
        """

        # The last step put in with the transformation is a placeholder step, so remove it
        series = series[:-1]

        # Denormalize the data using the stored min and max of the original de-trended series
        series = self.denormalize(series)

        # Add back in the true original timestep then recover the trend using a cumulative sum over the series.
        series = np.insert(series, 0, self.init_x)
        series = np.cumsum(series)

        # If the transformation logged the series first, then we need to exponentiate it now to undo the log.
        if exp:
            series = np.exp(series)

        return series


class TSManager:
    "Helper class for a Pandas DataFrame of Time Series variable transformations."""

    def __init__(self,
                 parallel_series_df: pd.DataFrame,
                 num_aux: int = 0,
                 ) -> None:
        """
        Create and record transforms for every non-auxiliary features in
        the DataFrame.

        Auxiliary features are expected to be the right-most columns!

        :param parallel_series_df: A Pandas Dataframe object in which rows are timesteps and columns are series or
                                   auxiliary features.
        :type parallel_series_df: pd.core.Frame.DataFrame
        :param num_aux: The number of auxiliary features in the DataFrame.
        :type num_aux: int
        """
        # This won't work if there is no timeseries in the DataFrame!
        assert len(parallel_series_df.columns) - num_aux > 0, "Must have at least 1 time series, not just auxiliaries!"

        # TSManager holds a dictionary of a TSTransformer for every series in order to manage the transformations.
        self.transforms_ = dict()

        # Loop through every non-auxiliary column of the parallel_series_df and instantiate a TSTransformer for it.
        for col in list(parallel_series_df.iloc[:, :(len(parallel_series_df.columns) - num_aux)].columns):
            self.transforms_[col] = TSTransformer()

    def transform_all(self,
                      parallel_series_df: pd.DataFrame,
                      ) -> pd.DataFrame:
        """ Apply transforms to all non-auxiliary columns of the given DataFrame.

        :param parallel_series_df: A Pandas Dataframe object in which rows are timesteps and columns are series or
                                   auxiliary features.
        :type parallel_series_df: pd.core.Frame.DataFrame
        :return: The transformed DataFrame in which all time series have been de-trended and normalized.
        :rtype: pd.core.Frame.DataFrame
        """

        # For every column that had a TSTransformer made for it during initialization, perform that transformation.
        for key in self.transforms_.keys():
            parallel_series_df.loc[:, key] = self.transforms_[key].transform(parallel_series_df.loc[:, key].values)
        return parallel_series_df

    def invert_all(self,
                   parallel_series_df: pd.DataFrame,
                   ) -> pd.DataFrame:
        """ Invert transforms of all non-auxiliary columns of the given DataFrame.

        :param parallel_series_df: A Pandas Dataframe object in which rows are timesteps and columns are series or
                                   auxiliary features.
        :type parallel_series_df: pd.core.Frame.DataFrame
        :return: The inverted DataFrame in which all time series have been re-trended and reverted to original scale.
        :rtype: pd.core.Frame.DataFrame
        """
        for key in self.transforms_.keys():
            parallel_series_df.loc[:, key] = self.transforms_[key].invert(parallel_series_df.loc[:, key].values)
        return parallel_series_df


class TSDS(Dataset):
    """
    Inherits from torch.utils.data.Dataset.

    A time series dataset from a data_frame consisting of multiple parallel
    time series and optionally auxiliary features.

    Rows are expected to be time steps. Columns are expected to be time series
    or auxiliary features.

    Auxiliary features are temporal features that apply to all series, such as
    time step or feature engineered periodicity data. They are NOT static covariates
    of any specific series.

    Auxiliary features are assumed to be the right-most columns. They are included in
    the predictor portion of __getitem__, but not in the target portion.
    """

    def __init__(self,
                 parallel_time_series: pd.DataFrame,
                 lookback: int,
                 forecast: int = 1,
                 num_aux: int = 0,
                 ):
        self.data = torch.from_numpy(parallel_time_series.values).to(torch.float32)
        self.num_aux = num_aux
        self.lookback = lookback
        self.forecast = forecast

        assert self.__len__() > 0, "There are no remaining time steps in the data after lookback and forecast!"
        assert self.num_aux >= 0, "The current code expects at least 1 auxiliary feature (timestep or periodicity, etc)"

    def __len__(self) -> int:
        """
        The length of the dataset is the number of time steps remaining
        after subtracting the lookback and forecast lengths.
        """
        return self.data.shape[0] - (self.lookback + self.forecast)

    def __getitem__(self,
                    idx: int,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Return an X, y tuple. Auxiliary features are in X, but not y. """

        if idx >= self.__len__():
            raise IndexError("Index out range")

        return (self.data[idx: idx + self.lookback].permute(1, 0),
                self.data[idx + self.lookback: idx + self.lookback + self.forecast, :-self.num_aux])


def get_tsds(parallel_series_df: pd.DataFrame,
             lookback: int,
             forecast: int,
             train_prop: float,
             num_aux: int = 0) -> Tuple[TSDS, TSDS]:
    """ Return two TSDS classes for training and testing.
    :param parallel_series_df: A Pandas Dataframe object in which rows are timesteps and columns are series or
                               auxiliary features.
    :type parallel_series_df: pd.core.Frame.DataFrame
    :param lookback: The number of prior timesteps to consider when making forecasting.
    :type lookback: int
    :param forecast: The number of future timesteps to predict when making a forecast.
    :type forecast: int
    :train_prop: The proportion of the timesteps in the data to reserve for training. The rest will be for testing.
    :type train_prop: float
    :param num_aux: The number of auxiliary features in the DataFrame.
    :type num_aux: int
    :return: A tuple containing the train dataset and the test dataset.
    :rtype: Tuple[TSDS, TSDS]
    """

    assert 0.0 < train_prop < 1.0, "train_prop must be between 0.0 and 1.0, exclusive!"
    assert num_aux >= 0, "num_aux must be at least one! Try using the normalized timestep or the sin/cos periodicity."

    # Get the number of samples to use for training by rounding the train_prop proportion to the nearest sample.
    train_len = round(parallel_series_df.shape[0] * train_prop)

    # Instantiate the train_ds and test_ds with the correct number of samples.
    train_ds = TSDS(parallel_series_df.loc[: train_len, :], lookback, forecast, num_aux=num_aux)
    test_ds = TSDS(parallel_series_df.loc[train_len:, :], lookback, forecast, num_aux=num_aux)

    return train_ds, test_ds


def get_dataloaders(train_ds: TSDS,
                    test_ds: TSDS,
                    train_batch_size: int,
                    test_batch_size: int,
                    shuffle_test: bool = False,
                    drop_last: bool = False,
                    ) -> Tuple[DataLoader, DataLoader]:
    """ Return two Pytorch DataLoaders for training and testing.

    :param train_ds: The TSDS object containing the training data.
    :type train_ds: TSDS
    :param test_ds: The TSDS object containing the testing data.
    :type test_ds: TSDS
    :param train_batch_size: The batch size that the dataloaders should use for training.
    :type train_batch_size: int
    :param test_batch_size: The batch size that the dataloaders should use for testing (can be different from training
                            since its possible that there are very few samples in our test set).
    :type test_batch_size: int
    :param shuffle_test: Whether the dataloaders should shuffle the data to randomize the samples in each batch.
    :type shuffle_test: bool
    :param drop_last: Should the dataloaders drop the final batch?
    :type drop_last: bool
    :return: A tuple containing the train dataloader and the test dataloader.
    :rtype: Tuple[DataLoader, DataLoader]
    """

    train_loader = DataLoader(train_ds,
                              batch_size=train_batch_size,
                              shuffle=True,
                              drop_last=drop_last,
                              )
    test_loader = DataLoader(test_ds,
                             batch_size=test_batch_size,
                             shuffle=shuffle_test,
                             drop_last=drop_last,
                             )

    return train_loader, test_loader

