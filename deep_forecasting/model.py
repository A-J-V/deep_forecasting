import torch
from torch import nn


class MLPTimeBlock(nn.Module):
    """
    Accept tensors and pass them through linear, ReLU, and dropout.

    This block is a time-wise fully-connected layer that expects time-steps
    to be the final dimensions of the input, and treats the time-steps in the
    lookback period as features.
    """

    def __init__(self,
                 features: int,
                 dropout: float,
                 ):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features=features,
                                           out_features=features),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 )

    def forward(self, x):
        return self.mlp(x)


class MLPFeatureBlock(nn.Module):
    """
    Accept tensors and pass them through linear, ReLU, dropout, linear, dropout.

    This block is a feature-wise fully-connected layer that expects features
    to be the final dimensions of the input, and treats the features as
    features.
    """

    def __init__(self,
                 features: int,
                 dropout: float,
                 ):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features=features,
                                           out_features=features),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=features,
                                           out_features=features),
                                 nn.Dropout(dropout),
                                 )

    def forward(self, x):
        return self.mlp(x)


class MixerLayer(nn.Module):
    """ Extract information across time and features. """

    def __init__(self,
                 lookback: int,
                 features: int,
                 dropout: float,
                 ):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(features)
        self.mlp_time = MLPTimeBlock(lookback, dropout)
        self.batchnorm2 = nn.BatchNorm1d(features)
        self.mlp_feat = MLPFeatureBlock(features, dropout)

    def forward(self, x):
        identity = x
        x = self.batchnorm1(x)
        x = self.mlp_time(x)
        x = x + identity
        identity = x
        x = self.batchnorm2(x)
        x = torch.transpose(x, 1, 2)
        x = self.mlp_feat(x)
        x = torch.transpose(x, 1, 2)
        x = x + identity
        return x


class TemporalProjection(nn.Module):
    """
    Final layer to project the data in the model back into the
    original format to serve as a forecast.
    """

    def __init__(self,
                 seq_len,
                 forecast=1,
                 ):
        super().__init__()
        self.fc = nn.Linear(in_features=seq_len,
                            out_features=forecast
                            )

    def forward(self, x):
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x


class TSMixer(nn.Module):
    """ Pytorch model based roughly on Google Research's 2023 paper on TSMixer. """

    def __init__(self,
                 lookback: int,
                 features: int,
                 forecast: int,
                 blocks: int,
                 dropout: float,
                 num_aux: int = 0,
                 ):
        """
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
        """
        assert features > 0, "Cannot have a model with no features!"
        assert forecast > 0, "Cannot have a model with no forecasts!"
        assert num_aux > 0, "Must have at least one auxiliary feature!"

        super().__init__()
        self.features = features
        self.num_aux = num_aux
        self.model = nn.Sequential(*[
            nn.Sequential(*(MixerLayer(lookback, features, dropout) for _ in range(blocks))),
            TemporalProjection(lookback, forecast)
        ])

    def forward(self, x):
        x = self.model(x)
        # Will not return auxiliary features in the prediction!
        return x[:, :, :self.features-self.num_aux]
