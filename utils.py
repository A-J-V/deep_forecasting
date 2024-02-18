import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TSTransformer:
    """ Hold information for transforming and inverting a single time series. """

    def __init__(self, series: np.ndarray):
        """ Given a numpy array of a time series, store info to normalize and denormalize."""
        self.mini = None
        self.maxi = None
        self.init_x = None
        self.normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        self.denormalize = lambda x: x * (self.maxi - self.mini) + self.mini

    def transform(self, series, log=False):
        """
        Detrend and normalize the given series using the values recorded in init.
        Store values to invert the transformation.
        """

        if log:
            series = np.log(series)
        self.init_x = series[0]
        series = np.diff(series, 1)
        self.mini = series.min()
        self.maxi = series.max()
        series = self.normalize(series)
        series = np.insert(series, -1, series[-1])
        return series

    def invert(self, series, exp=False):
        """
        Invert the given series using the stored information from init and transform().
        Inverts the normalization and adds back in trend information.
        """

        series = series[:-1]
        series = self.denormalize(series)
        series = np.insert(series, 0, self.init_x)
        series = np.cumsum(series)
        if exp:
            series = np.exp(series)
        return series


class TSManager:
    "Helper class for a Pandas DataFrame of Time Series variable transformations."""

    def __init__(self,
                 df: pd.DataFrame,
                 num_aux: int = 0,
                 ):
        """
        Create and record transforms for every non-auxiliary features in
        the DataFrame.

        Auxiliary features are expected to be the right-most columns!
        """

        self.transforms_ = dict()

        for col in list(df.iloc[:, :(len(df.columns) - num_aux)].columns):
            self.transforms_[col] = TSTransformer(df[col].values)

    def transform_all(self, df):
        """ Apply transforms to all non-auxiliary columns of the given DataFrame."""

        for key in self.transforms_.keys():
            df.loc[:, key] = self.transforms_[key].transform(df.loc[:, key].values)
        return df

    def invert_all(self, df):
        """ Invert transforms of all non-auxiliary columns of the given DataFrame."""
        for key in self.transforms_.keys():
            df.loc[:, key] = self.transforms_[key].invert(df.loc[:, key].values)
        return df


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

    def __init__(self, df, lookback, forecast=1, num_aux=0):
        self.data = torch.from_numpy(df.values).to(torch.float32)
        self.num_aux = num_aux
        self.lookback = lookback
        self.forecast = forecast

    def __len__(self):
        """
        The length of the dataset is the number of time steps remaining
        after subtracting the lookback and forecast lengths.
        """
        return self.data.shape[0] - (self.lookback + self.forecast)

    def __getitem__(self, index):
        """ Return an X, y tuple. Auxiliary features are in X, but not y. """

        if index >= self.__len__():
            raise IndexError("Index out range")

        return (self.data[index: index + self.lookback].permute(1, 0),
                self.data[index + self.lookback: index + self.lookback + self.forecast, :-self.num_aux])


def get_tsds(df, lookback, forecast, train_prop, num_aux=0):
    """ Return two TSDS classes for training and testing. """

    train_len = round(df.shape[0] * train_prop)
    train_ds = TSDS(df.loc[: train_len, :], lookback, forecast, num_aux=num_aux)
    test_ds = TSDS(df.loc[train_len:, :], lookback, forecast, num_aux=num_aux)

    return train_ds, test_ds


def get_dataloaders(train_ds,
                    test_ds,
                    batch_size,
                    shuffle_test=False,
                    drop_last=False,
                    ):
    """ Return two Pytorch DataLoaders for training and testing. """

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=drop_last,
                              )
    test_loader = DataLoader(test_ds,
                             batch_size=4,
                             shuffle=shuffle_test,
                             drop_last=drop_last,
                             )

    return train_loader, test_loader

