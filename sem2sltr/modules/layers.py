import math

import torch
from torch import nn, Tensor


class PosEmb(nn.Module):
    def __init__(self, emb, position_emb):
        super(PosEmb, self).__init__()
        self.emb = emb
        self.position_emb = position_emb

    def forward(self, x, mask):
        return self.position_emb(self.emb(x, mask))


class MaskedNorm(nn.Module):
    """ @ from Camgoz signjoey
        Original Code from:
        https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, norm_type, num_groups, num_features):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])


class SpatialEmbeddings(nn.Module):
    """
    @ from Camgoz signjoey
    Simple Linear Projection Layer
    (For encoder outputs to predict glosses)
    """

    # pylint: disable=unused-argument
    def __init__(
            self,
            embedding_dim: int,
            input_size: int,
            num_heads: int,
            norm_type: str = "batch",
            scale: bool = False,
            activation=nn.ReLU()
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.norm_type = norm_type
        self.ln = nn.Linear(self.input_size, self.embedding_dim)
        self.norm = MaskedNorm(
            norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
        )

        self.activation = activation
        self.scale = scale
        self.scale_factor = math.sqrt(self.embedding_dim)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        :param mask: frame masks
        :param x: input frame features
        :return: embedded representation for `x`
        """
        x = self.ln(x)

        if self.norm_type:
            x = self.norm(x, mask)
        if self.activation:
            x = self.activation(x)
        x = x * self.scale_factor if self.scale else x
        return x
