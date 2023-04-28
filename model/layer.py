import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation
import math

INF = 1e4


class Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x / torch.sqrt(variance + self.eps)


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


def attention_normalize(a, mask=None, dim=-1, method="softmax"):
    """不同的注意力归一化方案
    softmax：常规/标准的指数归一化；
    squared_relu：来自 https://arxiv.org/abs/2202.10447 ；
    softmax_plus：来自 https://kexue.fm/archives/8823 。
    """
    if method == "softmax":
        return torch.softmax(a, dim=dim)
    else:
        if mask is not None:
            assert mask.ndim == 3
            l = mask.sum(-1, keepdim=True)
        else:
            l = torch.ones_like(a) * a.shape[-2]
        if method == "squared_relu":
            return torch.relu(a) ** 2 / l
        elif method == "softmax_plus":
            scale = torch.log(l) / np.log(512)
            # mask: 1 for not padding, 0 for padding
            # padding position's scale is 1
            if mask is not None:
                scale = scale.masked_fill(mask == 0, 1.0)
            return torch.softmax(a * scale, dim=dim)
    return a


class ScaleOffset(nn.Module):
    """简单的仿射变换层（最后一维乘上gamma向量并加上beta向量）
    说明：1、具体操作为最后一维乘上gamma向量并加上beta向量；
         2、如果直接指定scale和offset，那么直接常数缩放和平移；
    """

    def __init__(
            self,
            hidden_size=768,
            scale=True,
            offset=True,
    ):
        super().__init__()
        self.scale = scale
        self.offset = offset

        if self.scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        if self.offset:
            self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs):
        if self.scale:
            inputs = inputs * self.weight
        if self.offset:
            inputs = inputs + self.bias

        return inputs


class GatedAttentionUnit(nn.Module):
    """门控注意力单元
    链接：https://arxiv.org/abs/2202.10447
    介绍：https://kexue.fm/archives/8934
    说明：没有加入加性相对位置编码，个人认为是不必要的；如果觉得有必要，
         可以自行通过a_bias传入。
    """

    def __init__(
            self,
            hidden_size=768,
            intermediate_size=1536,
            attention_key_size=128,
            activation="swish",
            use_bias=False,
            normalization="softmax_plus",
            attention_scale=True,
            attention_dropout=0.1,
    ):
        super().__init__()
        self.activation = get_activation(activation)
        self.intermediate_size = intermediate_size
        self.attention_key_size = attention_key_size
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.causal = True

        self.i_dense = nn.Linear(
            hidden_size, intermediate_size*2 + attention_key_size, bias=self.use_bias
        )
        self.o_dense = nn.Linear(intermediate_size, hidden_size, bias=self.use_bias)

        self.q_scaleoffset = ScaleOffset(attention_key_size, offset=True)
        self.k_scaleoffset = ScaleOffset(attention_key_size, offset=True)

    @staticmethod
    def apply_rotary(x, sinusoidal_pos=None):
        if sinusoidal_pos is None:
            return x
        sin, cos = sinusoidal_pos
        # x.shape [batch, seq_len, 2]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # [cos_nθ, -sin_nθ] [x1]
        # [sin_nθ,  cos_nθ] [x2]
        # => [x1 * cos_nθ - x2 * sin_nθ, x1 * sin_nθ + x2 * cos_nθ]
        # 苏神的rotary，使用了下面的计算方法。
        # return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2, -1)
        # 考虑到矩阵乘法torch.einsum("bmd,bnd->bmn", q, k)，因此可以直接在最后一个维度拼接（无需奇偶交错）
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            sinusoidal_pos=None,
    ):
        # print(hidden_states.shape, hidden_states[0])
        seq_len, device = hidden_states.shape[-2], hidden_states.device
        # 投影变换
        x = self.i_dense(hidden_states)
        u, v, qk = torch.split(
            self.activation(x),
            [self.intermediate_size, self.intermediate_size, self.attention_key_size],
            dim=-1,
        )
        q, k = self.q_scaleoffset(qk), self.k_scaleoffset(qk)

        # 加入RoPE
        q, k = self.apply_rotary(q, sinusoidal_pos), self.apply_rotary(
            k, sinusoidal_pos
        )

        # Attention
        a = torch.einsum("bmd,bnd->bmn", q, k)

        if self.attention_scale:
            a = a / self.attention_key_size ** 0.5

        if attention_mask is not None:
            a = a.masked_fill(attention_mask == 0, -INF)

        A = attention_normalize(a, attention_mask, dim=-1, method=self.normalization)

        A = F.dropout(A, p=self.attention_dropout, training=self.training)

        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).triu(1)
            A = A.masked_fill(causal_mask, -INF)

        # 计算输出
        o = self.o_dense(u * torch.einsum("bmn,bnd->bmd", A, v))

        # outputs = (o, A) if output_attentions else (o,)
        return o


class GAULayer(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.gau = GatedAttentionUnit(
            config.hidden_size,
            config.intermediate_size,
            config.attention_key_size,
            config.activation,
            config.use_bias,
            config.normalization,
            config.attention_scale,
            config.attention_dropout,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.eps, bias=True)
        self.hidden_dropout = config.hidden_dropout

        if config.deepnorm:
            self.alpha = math.pow(2.0 * config.num_layers, 0.25)
        else:
            self.alpha = 1.0

    def residual_connection(self, x, res):

        return self.norm(res*self.alpha + x)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            sinusoidal_pos=None,
    ):
        # 投影变换
        gau_output = self.gau(
            hidden_states, attention_mask, sinusoidal_pos
        )

        # dropout and residual
        o = F.dropout(gau_output, p=self.hidden_dropout, training=self.training)
        o = self.residual_connection(hidden_states, o)

        return o
