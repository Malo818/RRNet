import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Dropout


class FeatureFusion(nn.Module):
    def __init__(self, D_MODEL, NHEAD, LAYER_NAMES, ATTENTION):       # (D_MODEL=256, NHEAD=8, LAYER_NAMES=['Sa', 'SCa', 'MCa'] * 4, ATTENTION = 'full' )
        super(FeatureFusion, self).__init__()

        self.d_model = D_MODEL
        self.nhead = NHEAD
        self.layer_names = LAYER_NAMES
        s2s = Single2SingleFusion(D_MODEL, NHEAD, ATTENTION)
        self.layers_Sa = nn.ModuleList([copy.deepcopy(s2s) for _ in range(len(self.layer_names)//3)])
        s2m = Single2MultipleFusion(D_MODEL, NHEAD, ATTENTION)
        self.layers_SXa = nn.ModuleList([copy.deepcopy(s2m) for _ in range(2*len(self.layer_names)//3)])
        self.layers = nn.ModuleList()
        for i in range(len(self.layers_Sa)):
            self.layers.append(self.layers_Sa[i])
            self.layers.append(self.layers_SXa[2 * i])
            self.layers.append(self.layers_SXa[2 * i + 1])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1):          # image [4bs,100,256]       pc [mbs,100,256]
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        bs = feat0.size(0)//4
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'Sa':
                feat0 = layer(feat0)
                feat1 = layer(feat1)
            elif name == 'SCa':
                feat1 = layer(feat1, feat1, bs)
            elif name == 'MCa':
                feat0 = layer(feat0, feat1, bs)
                feat1 = layer(feat1, feat0, bs)
            else:
                raise KeyError
        return feat0, feat1

class Single2SingleFusion(nn.Module):
    def __init__(self, d_model, nhead, attention='full'):
        super(Single2SingleFusion, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention0()
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        mbs = x.size(0)  # m*batch size
        query = key = value = x
        query = F.normalize(self.q_proj(query).view(mbs, -1, self.nhead, self.dim))  # [mbs, N, (H, D)]
        key = self.k_proj(key).view(mbs, -1, self.nhead, self.dim)                   # [mbs, N, (H, D)]
        value = self.v_proj(value).view(mbs, -1, self.nhead, self.dim)               # [mbs, N, (H, D)]
        message = self.attention(query, key, value)                       # [mbs, N, (H, D)]
        message = self.merge(message.view(mbs, -1, self.nhead*self.dim))  # [mbs, N, C]
        message = self.norm1(message)
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        return x + message

class FullAttention0(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        QK = torch.einsum("mqhd,mwhd->mqwh", queries, keys)  # [mbs, N, N, H]
        softmax_temp = 1. / queries.size(3) ** .5            # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)
        queried_values = torch.einsum("mqwh,mehd->mqhd", A, values)  # [mbs, N, H, D]
        return queried_values.contiguous()


class Single2MultipleFusion(nn.Module):
    def __init__(self, d_model, nhead, attention='full'):
        super(Single2MultipleFusion, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention1()
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, y, bs):

        query = x.reshape(bs, -1, self.nhead*self.dim)
        key = value = y.reshape(bs, -1, self.nhead * self.dim)

        query = F.normalize(self.q_proj(query).view(bs, -1, self.nhead, self.dim))  # [bs, m1N, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)                   # [bs, m2N, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)               # [bs, m2N, (H, D)]

        message = self.attention(query, key, value)                      # [bs, m1N, (H, D)]
        message = self.merge(message.view(-1, 100, self.nhead*self.dim))  # [m1bs, N, C]
        message = self.norm1(message)

        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        return x + message

class FullAttention1(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.k_proj_in = nn.Linear(32, 32, bias=False)
        self.v_proj_in = nn.Linear(32, 32, bias=False)
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)


    def forward(self, queries, keys, values):                              #[bs, mxN, (H, D)]
        # First fusion
        QK = torch.einsum("bqhd,bwhd->bqwh", queries, keys)                #[bs, m1N, m2N, H]
        softmax_temp = 1. / queries.size(3)**.5                           # sqrt(D)
        sQK_separation = softmax_temp * QK.view(QK.size(0), QK.size(1), -1, 100, QK.size(3))
        A = torch.softmax(sQK_separation, dim=3)
        values_separation = values.view(values.size(0), -1, 100, values.size(2), values.size(3))
        if self.use_dropout:
            A = self.dropout(A)
        queried_values = torch.einsum("bqwnh,bwnhd->bqwhd", A, values_separation) #[bs, m1N, m2, H, D]

        # Secondary fusion
        key_in = self.k_proj_in(queried_values)    # [bs, m1N, m2, H, D]
        value_in = self.v_proj_in(queried_values)  # [bs, m1N, m2, H, D]
        QK_in = torch.einsum("bqhd,bqwhd->bqwh", queries, key_in)
        A_in = torch.softmax(softmax_temp * QK_in, dim=2)
        if self.use_dropout:
            A_in = self.dropout(A_in)
        queried_values_in = torch.einsum("bqwh,bqwhd->bqhd", A_in, value_in)  # [bs, m1N, H, D]

        return queried_values_in.contiguous()


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1
class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()




