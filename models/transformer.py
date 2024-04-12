#-*- coding:utf-8 -*-

import os
import sys
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from video_summarizer.models import Trainer

"""
Attention Is All You Need
https://arxiv.org/abs/1706.03762
https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
"""

class Transformer(nn.Module):
    def __init__(self, input_size=1024, encoder_layers=6, attention_heads=8, more_residuals=False, max_length=None, pos_embed="simple", epsilon=1e-5, weight_init=None):
        super(Transformer, self).__init__()

        # feature dimension that is the the dimensionality of the key, query and value vectors
        # as well as the hidden dimension for the FF layers
        self.input_size = input_size

        # Optional positional embeddings
        self.max_length = max_length
        if self.max_length:
            self.pos_embed_type = pos_embed

            if self.pos_embed_type == "simple":
                self.pos_embed = torch.nn.Embedding(self.max_length, self.input_size)
            elif self.pos_embed_type == "attention":
                self.pos_embed = torch.zeros(self.max_length, self.input_size)
                for pos in np.arange(self.max_length):
                    for i in np.arange(0, self.input_size, 2):
                        self.pos_embed[pos, i] = np.sin(pos / (10000 ** ((2 * i)/self.input_size)))
                        self.pos_embed[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/self.input_size)))
            else:
                self.max_length = None

        # Optional: Add a residual connection between before/after the Encoder layers
        self.more_residuals = more_residuals

        # Common steps
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = torch.nn.LayerNorm(self.input_size, epsilon)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=attention_heads, dim_feedforward=self.input_size, dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, num_layers=encoder_layers, norm=self.layer_norm)

        self.k1 = nn.Linear(in_features=self.input_size, out_features=self.input_size)
        self.k2 = nn.Linear(in_features=self.input_size, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # Weights initialization
        if weight_init:
            if weight_init.lower() in ["he", "kaiming"]:
                for i in np.arange(self.transformer_encoder.num_layers):
                    init.kaiming_uniform_(self.transformer_encoder.layers[i].linear1.weight)
                    init.kaiming_uniform_(self.transformer_encoder.layers[i].linear2.weight)
                init.kaiming_uniform_(self.k1.weight)
                init.kaiming_uniform_(self.k2.weight)
            elif weight_init.lower() == "xavier":
                for i in np.arange(self.transformer_encoder.num_layers):
                    init.xavier_uniform_(self.transformer_encoder.layers[i].linear1.weight)
                    init.xavier_uniform_(self.transformer_encoder.layers[i].linear2.weight)
                init.xavier_uniform_(self.k1.weight)
                init.xavier_uniform_(self.k2.weight)

    def forward(self, x):
        """
        Input
          x: (seq_len, batch_size, input_size)
        Output
          y: (seq_len, batch_size, 1)
        """
        seq_len, batch_size, input_size = x.shape
        x = x.permute(1, 0, 2) # (batch_size, seq_len, input_size)

        if self.max_length is not None:
            assert self.max_length >= seq_len, "input sequence has higher length than max_length"
            if self.pos_embed_type == "simple":
                pos_tensor = torch.arange(seq_len).repeat(1, batch_size).view([batch_size, seq_len]).to(x.device)
                x += self.pos_embed(pos_tensor)
            elif self.pos_embed_type == "attention":
                x += self.pos_embed[:seq_len, :].repeat(1, batch_size).view(batch_size, seq_len, input_size).to(x.device)

        x = x.permute(1, 0, 2) # (seq_len, batch_size, input_size)
        encoder_out = self.transformer_encoder.forward(x)
        
        if self.more_residuals:
            encoder_out += x

        y = self.k1(encoder_out)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.layer_norm(y)
        y = self.k2(y)
        y = self.sigmoid(y)

        return y


class TransformerTrainer(Trainer):
    def _init_model(self):
        model = Transformer(
            encoder_layers=int(self.hps.extra_params.get("encoder_layers", 6)),
            attention_heads=int(self.hps.extra_params.get("attention_heads", 8)),
            more_residuals=self.hps.extra_params.get("more_residuals", False),
            max_length=int(self.hps.extra_params["max_pos"]) if "max_pos" in self.hps.extra_params else None,
            pos_embed=self.hps.extra_params.get("pos_embed", "simple"),
            epsilon=float(self.hps.extra_params.get("epsilon", 1e-5)), 
            weight_init=self.hps.extra_params.get("weight_init", None)
        )

        cuda_device = self.hps.cuda_device
        if self.hps.use_cuda:
            self.log.info(f"Setting CUDA device: {cuda_device}")
            torch.cuda.set_device(cuda_device)
        if self.hps.use_cuda:
            model.cuda()
        self.model = torch.nn.DataParallel(model)
        return self.model

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
                seq = torch.from_numpy(seq).unsqueeze(1) # (seq_len, 1, input_size)
                target = dataset["gtscore"][...]
                target = torch.from_numpy(target).view(-1, 1, 1) # (seq_len, 1, 1)

                # Normalize frame scores
                target -= target.min()
                target /= target.max() - target.min()
                
                if self.hps.use_cuda:
                    seq, target = seq.cuda(), target.cuda()

                scores = self.model(seq)
                # scores = scores.to(torch.float64)

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
    model = Transformer()
    print("Trainable parameters in model:", sum([_.numel() for _ in model.parameters() if _.requires_grad]))

    # print()
    # print("Possible flags for Transformer:")
    # print("encoder_layers: an integer describing the number of encoder layers to use in the transformer. Default=6")
    # print("attention_heads: an integer describing the number of attention heads to use in the transformer. Default=8")
    # print("max_pos: an integer describing the maximum length of a sequence (e.g. the number of frames in each video). Specify to use positional encodings. Default=None")
    # print("more_residuals. Specify to add a residual connection between before and after the encoder layers. Default=False")
    # print("max_pos: an integer describing the maximum length of a sequence (e.g. the number of frames in each video). Specify to use positional encodings. Default=None")
    # print("pos_embed: \"simple\" or \"attention\". Whether to use simple (embedding of the position of the image in sequence) or attention-based cos-sin positional encodings. Specify `max_pos` to use positional encodings. Default=simple")
    # print("epsilon: a float added to the denominator for numerical stability when performing layer normalization. Default=1e-5")
    # print("weight_init: \"xavier\" or \"he\"/\"kaiming\". Whether to use Xavier weight initialization, Kaiming initialization, or none. Default=None")

    x = torch.randn(10, 3, 1024)
    y = model(x)
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    assert y.shape[2] == 1
