import os
import sys
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(sys.path)
from video_summarizer.models import Trainer


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
        self.drop = nn.Dropout(p=0.5)

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

    def forward(self, x):
        """
        Input
          x: (seq_len, batch_size, input_size)
        Output
          y: (seq_len, batch_size, 1)
        """
        seq_len, batch_size, input_size = x.shape
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, input_size)

        residual = x
        weighted_value, attn_weights = self.attention(x)
        y = weighted_value + residual
        y = self.drop(y)
        y = self.norm_y(y)

        y = self.linear_1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        y = self.sigmoid(y)
        y = y.permute(1, 0, 2)  # (seq_len, batch_size, 1)

        return y

class PGL_SUMTrainer(Trainer):
    def _init_model(self):
        model = PGL_SUM(
            input_size=self.hps.input_size,
            output_size=self.hps.input_size,
            freq=int(self.hps.extra_params.get("freq", 10000)),
            pos_enc=self.hps.extra_params.get("pos_enc"),
            num_segments=int(self.hps.extra_params["num_segments"]) if "num_segments" in self.hps.extra_params else None,
            heads=int(self.hps.extra_params.get("heads", 1)),
            fusion=self.hps.extra_params.get("fusion")
        )

        cuda_device = self.hps.cuda_device
        if self.hps.use_cuda:
            self.log.info(f"Setting CUDA device: {cuda_device}")
            torch.cuda.set_device(cuda_device)
        if self.hps.use_cuda:
            model.cuda()
        return model

    def train(self, fold):
        self.model.train()
        train_keys, _ = self._get_train_test_keys(fold)
        self.draw_gtscores(fold, train_keys)

        criterion = nn.MSELoss()
        if self.hps.use_cuda:
            criterion = criterion.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.hps.lr, weight_decay=self.hps.weight_decay)

        # To record performances of the best epoch
        best_corr, best_avg_f_score, best_max_f_score = -1.0, 0.0, 0.0

        # For each epoch
        for epoch in range(self.hps.epochs):
            train_avg_loss = []
            dist_scores = {}
            random.shuffle(train_keys)

            # For each training video
            for key in train_keys:
                dataset = self.dataset[key]
                seq = dataset["features"][...]
                seq = torch.from_numpy(seq).unsqueeze(1)  # (seq_len, 1, input_size)
                target = dataset["gtscore"][...]
                target = torch.from_numpy(target).view(-1, 1, 1)  # (seq_len, 1, 1)

                # Normalize frame scores
                target -= target.min()
                target /= target.max() - target.min()

                if self.hps.use_cuda:
                    seq, target = seq.cuda(), target.cuda()

                scores = self.model(seq)
                loss = criterion(scores, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_avg_loss.append(float(loss))
                dist_scores[key] = scores.detach().cpu().numpy()

            # Average training loss value of epoch
            train_avg_loss = np.mean(np.array(train_avg_loss))
            self.log.info(f"Epoch: {f'{epoch+1}/{self.hps.epochs}':6}   "
                            f"Loss: {train_avg_loss:.05f}")
            self.hps.writer.add_scalar(f"{self.dataset_name}/Fold_{fold+1}/Train/Loss", train_avg_loss, epoch)

            # Evaluate performances on test keys
            if epoch % self.hps.test_every_epochs == 0:
                avg_corr, (avg_f_score, max_f_score) = self.test(fold)
                self.model.train()
                self.hps.writer.add_scalar(f"{self.dataset_name}/Fold_{fold+1}/Test/Correlation", avg_corr, epoch)
                self.hps.writer.add_scalar(f"{self.dataset_name}/Fold_{fold+1}/Test/F-score_avg", avg_f_score, epoch)
                self.hps.writer.add_scalar(f"{self.dataset_name}/Fold_{fold+1}/Test/F-score_max", max_f_score, epoch)
                best_avg_f_score = max(best_avg_f_score, avg_f_score)
                best_max_f_score = max(best_max_f_score, max_f_score)
                if avg_corr > best_corr:
                    best_corr = avg_corr
                    self.best_weights = self.model.state_dict()

        # Log final scores
        self.draw_scores(fold, dist_scores)

        return best_corr, best_avg_f_score, best_max_f_score


if __name__ == "__main__":
    model = PGL_SUM()
    print("Trainable parameters in model:", sum([_.numel() for _ in model.parameters() if _.requires_grad]))

    print()
    print("Possible flags for PGL_SUM:")
    print("freq: an integer describing the frequency used in positional encodings. Default=10000")
    print("pos_enc: \"absolute\" or \"relative\". Type of positional encoding to use. Default=None")
    print("num_segments: an integer describing the number of segments for local attention. Default=None")
    print("heads: an integer describing the number of attention heads. Default=1")
    print("fusion: \"add\", \"mult\", \"avg\", or \"max\". Method to fuse global and local attention. Default=None")

    x = torch.randn(10, 3, 1024)
    y = model(x)
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    assert y.shape[2] == 1
