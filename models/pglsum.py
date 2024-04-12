import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import sys
import h5py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SelfAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, freq=10000, heads=1, pos_enc=None):
        super(SelfAttention, self).__init__()

        self.permitted_encodings = ["absolute", "relative"]
        if pos_enc is not None:
            pos_enc = pos_enc.lower()
            assert pos_enc in self.permitted_encodings, f"Supported encodings: {*self.permitted_encodings,}"

        self.input_size = input_size
        self.output_size = output_size
        self.heads = heads
        self.pos_enc = pos_enc
        self.freq = freq
        self.Wk, self.Wq, self.Wv = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(self.heads):
            self.Wk.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
            self.Wq.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
            self.Wv.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
        self.out = nn.Linear(in_features=output_size, out_features=input_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=0.5)

    def getAbsolutePosition(self, T):
        freq = self.freq
        d = self.input_size

        pos = torch.arange(T).to(self.Wk[0].weight.device)
        i = torch.arange(T // 2).to(self.Wk[0].weight.device)

        pos = pos.view(-1, 1)
        pos = pos.repeat(1, i.shape[0])
        i = i.repeat(pos.shape[0], 1)

        AP = torch.zeros(T, T, device=self.Wk[0].weight.device)
        AP[pos, 2*i] = torch.sin(pos / freq ** ((2 * i) / d))
        AP[pos, 2*i+1] = torch.cos(pos / freq ** ((2 * i) / d))
        return AP

    def getRelativePosition(self, T):
        freq = self.freq
        d = 2 * T
        min_rpos = -(T - 1)

        i = torch.arange(T).to(self.Wk[0].weight.device)
        j = torch.arange(T).to(self.Wk[0].weight.device)

        i = i.view(-1, 1)
        i = i.repeat(1, i.shape[0])
        j = j.repeat(i.shape[0], 1)

        r_pos = j - i - min_rpos

        RP = torch.zeros(T, T, device=self.Wk[0].weight.device)
        idx = torch.arange(T // 2).to(self.Wk[0].weight.device)
        RP[:, 2*idx] = torch.sin(r_pos[:, 2*idx] / freq ** ((i[:, 2*idx] + j[:, 2*idx]) / d))
        RP[:, 2*idx+1] = torch.cos(r_pos[:, 2*idx+1] / freq ** ((i[:, 2*idx+1] + j[:, 2*idx+1]) / d))
        return RP

    def forward(self, x):
        outputs = []
        for head in range(self.heads):
            K = self.Wk[head](x)
            Q = self.Wq[head](x)
            V = self.Wv[head](x)

            energies = torch.matmul(Q, K.transpose(1, 0))
            if self.pos_enc is not None:
                if self.pos_enc == "absolute":
                    AP = self.getAbsolutePosition(T=energies.shape[0])
                    energies = energies + AP
                elif self.pos_enc == "relative":
                    RP = self.getRelativePosition(T=energies.shape[0])
                    energies = energies + RP

            att_weights = self.softmax(energies)
            _att_weights = self.drop(att_weights)
            y = torch.matmul(_att_weights, V)

            outputs.append(y)
        y = self.out(torch.cat(outputs, dim=1))
        return y, att_weights.clone()


class MultiAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, freq=10000, pos_enc=None,
                 num_segments=None, heads=1, fusion=None):
        super(MultiAttention, self).__init__()

        self.attention = SelfAttention(input_size=input_size, output_size=output_size,
                                       freq=freq, pos_enc=pos_enc, heads=heads)

        self.num_segments = num_segments
        if self.num_segments is not None:
            assert self.num_segments >= 2, "num_segments must be None or 2+"
            self.local_attention = nn.ModuleList()
            for _ in range(self.num_segments):
                self.local_attention.append(SelfAttention(input_size=input_size, output_size=output_size//num_segments,
                                                          freq=freq, pos_enc=pos_enc, heads=4))

        self.permitted_fusions = ["add", "mult", "avg", "max"]
        self.fusion = fusion
        if self.fusion is not None:
            self.fusion = self.fusion.lower()
            assert self.fusion in self.permitted_fusions, f"Fusion method must be: {*self.permitted_fusions,}"

    def forward(self, x):
        weighted_value, attn_weights = self.attention(x)

        if self.num_segments is not None and self.fusion is not None:
            segment_size = math.ceil(x.shape[0] / self.num_segments)
            for segment in range(self.num_segments):
                left_pos = segment * segment_size
                right_pos = (segment + 1) * segment_size
                local_x = x[left_pos:right_pos]
                weighted_local_value, attn_local_weights = self.local_attention[segment](local_x)

                weighted_value[left_pos:right_pos] = F.normalize(weighted_value[left_pos:right_pos].clone(), p=2, dim=1)
                weighted_local_value = F.normalize(weighted_local_value, p=2, dim=1)
                if self.fusion == "add":
                    weighted_value[left_pos:right_pos] += weighted_local_value
                elif self.fusion == "mult":
                    weighted_value[left_pos:right_pos] *= weighted_local_value
                elif self.fusion == "avg":
                    weighted_value[left_pos:right_pos] += weighted_local_value
                    weighted_value[left_pos:right_pos] /= 2
                elif self.fusion == "max":
                    weighted_value[left_pos:right_pos] = torch.max(weighted_value[left_pos:right_pos].clone(),
                                                                   weighted_local_value)

        return weighted_value, attn_weights


class PGL_SUM(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, freq=10000, pos_enc=None,
                 num_segments=None, heads=1, fusion=None):
        super(PGL_SUM, self).__init__()

        self.attention = MultiAttention(input_size=input_size, output_size=output_size, freq=freq,
                                        pos_enc=pos_enc, num_segments=num_segments, heads=heads, fusion=fusion)
        self.linear_1 = nn.Linear(in_features=input_size, out_features=input_size)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, frame_features):
        residual = frame_features
        weighted_value, attn_weights = self.attention(frame_features)
        y = weighted_value + residual
        y = self.drop(y)
        y = self.norm_y(y)

        y = self.linear_1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        y = self.sigmoid(y)
        y = y.view(1, -1)

        return y, attn_weights


if __name__ == '__main__':
    pass
    # Uncomment for a quick proof of concept
    model = PGL_SUM(input_size=256, output_size=256, num_segments=3, fusion="Add").cuda()
    _input = torch.randn(500, 256).cuda()  # [seq_len, hidden_size]
    output, weights = model(_input)
    print(f"Output shape: {output.shape}\tattention shape: {weights.shape}")

