"""Contains the PyTorch modules that comprise the architecture of the main network in this package."""

import torch
from torch import nn
from typing import Dict, List


class MLPTimeBlock(nn.Module):
    """An MLP along the time-step dimension."""

    def __init__(self,
                 features: int,
                 dropout: float,
                 ):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features=features,
                                           out_features=features),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 )

    def forward(self, x):
        # x is [batch_size, time series, time step]
        return self.mlp(x)


class MLPFeatureBlock(nn.Module):
    """An MLP along the feature dimension (a 'mixer' step)."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 dropout: float,
                 ):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features=in_features,
                                           out_features=hidden_features),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=hidden_features,
                                           out_features=out_features),
                                 nn.Dropout(dropout),
                                 )

    def forward(self, x):
        # x is [batch size, time step, time series]
        return self.mlp(x)


class GroupedMixerBlock(nn.Module):
    def __init__(self,
                 lookback: int,
                 group_registry: Dict[str, List[int]],
                 covariate_registry: Dict[str, List[int]],
                 aux_indices: List[int],
                 hidden_features: int,
                 dropout: float
                 ):
        """ A block that facilitates an MLP-mixer-like operations over different clusters of time series'."""
        super().__init__()

        # Save registries
        self.group_registry = group_registry
        self.covariate_registry = covariate_registry
        self.aux_indices = aux_indices

        # Flatten all group-only indices
        self.group_indices = [idx for grp in group_registry.values() for idx in grp]

        # Flatten all covariate indices (for norm2 shaping)
        self.cov_indices = [idx for covs in covariate_registry.values() for idx in covs]

        # Count up features for norm2
        self.input_feature_count = len(self.group_indices)
        self.aux_feature_count = len(self.aux_indices)
        self.cov_feature_count = len(self.cov_indices)
        total_feats = (self.input_feature_count +
                       self.aux_feature_count +
                       self.cov_feature_count
                       )

        # Time-mixing block
        self.norm1 = nn.LayerNorm(lookback)  # normalize along last dim = time
        self.mlp_time = MLPTimeBlock(lookback, dropout)

        # Feature-mixing prep
        self.norm2 = nn.LayerNorm(total_feats)  # normalize across all series+aux+cov

        # One MLP per group: in = group+aux+cov, out = group
        self.group_mlps = nn.ModuleDict()
        for gid, gidx in group_registry.items():
            n_cov = len(covariate_registry.get(gid, []))
            self.group_mlps[gid] = MLPFeatureBlock(
                in_features=len(gidx) + self.aux_feature_count + n_cov,
                hidden_features=hidden_features,
                dropout=dropout,
                out_features=len(gidx)
            )

    def forward(self, x):
        # x: [B, series_total, time]
        identity = x

        # 1) Time mixing (shared)
        x = self.norm1(x)
        x = self.mlp_time(x)
        x = x + identity

        # 2) Prepare residual for feature mixing (group-only)
        #    slice out only the group series (no aux, no cov)
        identity = x[:, self.group_indices, :]

        # 3) Feature mixing prep: norm across series+aux+cov
        x = torch.transpose(x, 1, 2)  # → [B, time, series_total]
        x = self.norm2(x)

        # 4) Group-wise MLPs (each sees group+aux+cov, outputs group-only)
        group_outs = []
        for gid, gidx in self.group_registry.items():
            covs = self.covariate_registry.get(gid, [])
            in_idx = gidx + self.aux_indices + covs
            xg = x[:, :, in_idx]  # [B, time, group+aux+cov]
            out_g = self.group_mlps[gid](xg)  # [B, time, group]
            group_outs.append(out_g)

        # 5) Recombine groups; they should already be ordered correctly
        x_cat = torch.cat(group_outs, dim=2)  # [B, time, sum(group_sizes)]
        x = torch.transpose(x_cat, 1, 2)  # [B, series_group_only, time]

        # 6) Final residual
        return x + identity


class TemporalProjection(nn.Module):
    """Projection from timesteps to forecast."""

    def __init__(self,
                 seq_len,
                 forecast=1,
                 ):
        super().__init__()
        self.fc = nn.Linear(in_features=seq_len,
                            out_features=forecast
                            )

    def forward(self, x):
        # x is [batch_size, time series, time steps]
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        # output [batch_size, forecasted steps, time_series]
        return x


class HierarchicalTimeSeriesMixer(nn.Module):
    def __init__(self,
                 lookback: int,
                 group_registry: Dict[str, List[int]],
                 covariate_registry: Dict[str, List[int]],
                 aux_indices: List[int],
                 hidden_features: int,
                 forecast: int,
                 blocks: int,
                 dropout: float,
                 ):

        super().__init__()
        self.model = nn.Sequential(*[
            nn.Sequential(*(GroupedMixerBlock(
                lookback,
                group_registry,
                covariate_registry if i == 0 else {},
                aux_indices if i == 0 else [],
                hidden_features,
                dropout) for i in range(blocks)
                           )
                         ),
            TemporalProjection(lookback, forecast)
        ])

    def forward(self, x):
        x = self.model(x)
        # Does not return covariate or auxiliary features in the prediction!
        return x
