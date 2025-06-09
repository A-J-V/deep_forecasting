from collections import defaultdict
import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List


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

    def __init__(self, parallel_series_df: pd.DataFrame) -> None:
        """
        Create and record transforms for every non-auxiliary feature in
        the DataFrame.

        Global auxiliary columns should be prefixed with aux_ to avoid any transformation.
        """
        # This won't work if there is no timeseries in the DataFrame!
        assert len(parallel_series_df.columns) > 0, "Must have at least 1 time series!"

        # TSManager holds a dictionary of a TSTransformer for every series in order to manage the transformations.
        self.transforms_ = dict()

        # Loop through every non-auxiliary column of the parallel_series_df and instantiate a TSTransformer for it.
        for col in list(parallel_series_df.columns):
            if col.split('_')[0] == 'aux':
                # If the column is prefixed with aux_, skip it so that it isn't transformed.
                continue
            self.transforms_[col] = TSTransformer()

    def transform_all(self, parallel_series_df: pd.DataFrame) -> pd.DataFrame:
        """ Apply transforms to all registered columns of the given DataFrame. """

        # For every column that had a TSTransformer made for it during initialization, perform that transformation.
        for key in self.transforms_.keys():
            parallel_series_df.loc[:, key] = self.transforms_[key].transform(parallel_series_df.loc[:, key].values)
        return parallel_series_df

    def invert_all(self, parallel_series_df: pd.DataFrame) -> pd.DataFrame:
        """ Invert transforms of all non-auxiliary columns of the given DataFrame.

        :param parallel_series_df: A Pandas Dataframe object in which rows are timesteps and columns are features.
        :type parallel_series_df: pd.core.Frame.DataFrame
        :return: The inverted DataFrame in which all time series have been re-trended and reverted to original scale.
        :rtype: pd.core.Frame.DataFrame
        """
        for key in self.transforms_.keys():
            parallel_series_df.loc[:, key] = self.transforms_[key].invert(parallel_series_df.loc[:, key].values)
        return parallel_series_df


class TSDS(Dataset):
    def __init__(self,
                 parallel_time_series: pd.DataFrame,
                 lookback: int,
                 forecast: int = 1,
                 ):
        self.df = parallel_time_series.copy()
        self.lookback = lookback
        self.forecast = forecast

        # Patterns
        group_pattern = re.compile(r'^g\d+_')
        cov_pattern = re.compile(r'^c\d+_')
        aux_prefix = 'aux_'
        default_group = 'g0_'

        # Step 1: Normalize unnamed columns (assume group 0)
        new_cols = []
        for col in self.df.columns:
            if col.startswith(aux_prefix) or group_pattern.match(col) or cov_pattern.match(col):
                new_cols.append(col)
            else:
                new_cols.append(f"{default_group}{col}")
        self.df.columns = new_cols

        # Step 2: Build column sets
        self.group_cols = [c for c in self.df.columns if group_pattern.match(c)]
        self.cov_cols = [c for c in self.df.columns if cov_pattern.match(c)]
        self.aux_cols = [c for c in self.df.columns if c.startswith(aux_prefix)]

        # Order the columns into blocks, [g1, g2, g3, ..., c1, c2, c3, ..., aux_1, aux_2, aux_3, ...]
        # This is important to keep group-wise operations correct inside the network
        ordered_cols = (
                [c for c in self.df.columns if c in self.group_cols] +
                [c for c in self.df.columns if c in self.cov_cols] +
                [c for c in self.df.columns if c in self.aux_cols]
        )
        self.df = self.df[ordered_cols]

        # Step 3: Build index mappings
        self.aux_indices = [self.df.columns.get_loc(c) for c in self.aux_cols]
        self.group_indices = [self.df.columns.get_loc(c) for c in self.group_cols]
        self.cov_indices = [self.df.columns.get_loc(c) for c in self.cov_cols]

        # Step 4: Build registries
        self.group_registry: Dict[str, List[int]] = defaultdict(list)
        self.covariate_registry: Dict[str, List[int]] = defaultdict(list)

        for col in self.group_cols:
            group_id = col.split('_')[0]
            idx = self.df.columns.get_loc(col)
            self.group_registry[group_id].append(idx)

        for col in self.cov_cols:
            cov_id = col.split('_')[0].replace('c', 'g')  # Convert c1 → g1
            idx = self.df.columns.get_loc(col)
            self.covariate_registry[cov_id].append(idx)

        self.group_registry = dict(self.group_registry)
        self.covariate_registry = dict(self.covariate_registry)
        self.group_ids: List[str] = list(self.group_registry.keys())

        # Convert data to tensor [T, S]
        self.data = torch.from_numpy(self.df.values).float()

        assert self.__len__() > 0, "Not enough data for lookback + forecast"

    def __len__(self) -> int:
        return self.data.shape[0] - (self.lookback + self.forecast)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= self.__len__():
            raise IndexError("Index out of range")

        X_window = self.data[idx: idx + self.lookback]  # [L, S]
        y_window = self.data[idx + self.lookback: idx + self.lookback + self.forecast, :]

        X = X_window.permute(1, 0)  # [T, L]
        y = y_window[:, self.group_indices]  # [F, T_core]

        return X, y


def get_tsds(parallel_series_df: pd.DataFrame,
             lookback: int,
             forecast: int,
             train_prop: float,
             ) -> Tuple[TSDS, TSDS]:
    assert 0.0 < train_prop < 1.0, "train_prop must be between 0.0 and 1.0, exclusive!"

    # Get the number of samples to use for training by rounding the train_prop proportion to the nearest sample.
    train_len = round(parallel_series_df.shape[0] * train_prop)

    # Instantiate the train_ds and test_ds with the correct number of samples.
    train_ds = TSDS(parallel_series_df.loc[: train_len, :], lookback, forecast)
    test_ds = TSDS(parallel_series_df.loc[train_len:, :], lookback, forecast)

    return train_ds, test_ds


def get_dataloaders(train_ds: TSDS,
                    test_ds: TSDS,
                    train_batch_size: int,
                    test_batch_size: int,
                    shuffle_test: bool = False,
                    drop_last: bool = False,
                    ) -> Tuple[DataLoader, DataLoader]:
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
