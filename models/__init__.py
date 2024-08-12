#-*- coding:utf-8 -*-

import os
import sys
import numpy as np
import h5py
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eval import generate_summary, evaluate_summary, generate_scores, evaluate_scores

class Trainer:
    """Abstract class handling the training process"""
    def __init__(self, hps, splits_file):
        self.hps = hps
        self.log = hps.logger
        self.splits_file = splits_file
        # print('datasertttt',hps.dataset_of_file )
        self.dataset = h5py.File(hps.dataset_of_file[splits_file], "r")
        self.dataset_name = hps.dataset_name_of_file[splits_file]
        self._init_model()
        
    def reset(self):
        """Reset between two folds of the cross-validation"""
        self.model = self._init_model()
        torch.cuda.empty_cache()
        if self.hps.use_cuda:
            self.model.cuda()
        return self

    def _get_train_test_keys(self, fold):
        """Train/Test keys from current split file and fold"""
        self.fold = fold
        self.split = self.hps.splits_of_file[self.splits_file][fold]
        return self.split["train_keys"][:], self.split["test_keys"][:]

    def _init_model(self):
        """Initialize here your model"""
        raise Exception("_init_model has not been implemented")

    def train(self, fold):
        """Train model on train_keys"""
        raise Exception("train has not been implemented")

    def test(self, fold):
        """Test model on test_keys"""
        self.model.eval()
        _, test_keys = self._get_train_test_keys(fold)
        summary = {}
        with torch.no_grad():
            for key in test_keys:
                seq = self.dataset[key]['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(1) # (seq_len, batch_size, input_size)

                if self.hps.use_cuda:
                    seq = seq.cuda()

                y = self.model(seq) # (seq_len, batch_size, 1)
                summary[key] = y.squeeze().detach().cpu().numpy() # (seq_len,)

        avg_corr = self._eval_scores(summary, test_keys)
        avg_f_score, max_f_score = self._eval_summary(summary, test_keys)
        return avg_corr, (avg_f_score, max_f_score)

    def _eval_scores(self, machine_summary_activations, test_keys):
        """Evaluate the importances scores using ranking correlation.
        Input
          machine_summary_activations: dictionnary of predictions {key: (seq_len,), ...}
          test_keys: list of keys to evaluate on
        Output
          avg_corr: average correlation over the test keys (1,)
        """
        avg_corrs = []
        for key in test_keys:
            d = self.dataset[key]
            probs = machine_summary_activations[key]

            if "user_scores" not in d and self.dataset_name != "fvs":
                raise Exception(f"No /user_scores in video {key} for score evaluation in {self.dataset_name}, "
                                "make sure you have up-to-date .h5 dataset files.")

            user_scores = d["user_scores"][...] if self.dataset_name != "fvs" else d["gtscore"][...]
            n_frames = d["n_frames"][()]
            positions = d["picks"][...]

            machine_scores = generate_scores(probs, n_frames, positions)
            avg_corr = evaluate_scores(machine_scores, user_scores, metric="spearmanr")
            avg_corrs.append(avg_corr)
        
        avg_corr = np.mean(avg_corrs)
        return avg_corr

    def _eval_summary(self, machine_summary_activations, test_keys):
        """Evaluate the final summary using the F-score.
        Input
          machine_summary_activations: dictionnary of predictions {key: (seq_len,), ...}
          test_keys: list of keys to evaluate on
        Output
          avg_f_score: average F1 over the test keys (1,)
          max_f_score: max F1 over the test keys (1,)
        """
        avg_f_scores, max_f_scores = [], []
        for key in test_keys:
            d = self.dataset[key]
            probs = machine_summary_activations[key]

            if "change_points" not in d:
                raise Exception(f"No /change_points in video {key} for summary evaluation, "
                                "make sure you have up-to-date .h5 dataset files.")

            cps = d["change_points"][...]
            num_frames = d["n_frames"][()]
            nfps = d["n_frame_per_seg"][...].tolist()
            positions = d["picks"][...]
            user_summary = d["user_summary"][...]

            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions, self.hps.summary_proportion, self.hps.selection_algorithm)
            avg_f_score, max_f_score = evaluate_summary(machine_summary, user_summary)
            avg_f_scores.append(avg_f_score)
            max_f_scores.append(max_f_score)

        avg_f_score = np.mean(avg_f_scores)
        max_f_score = np.mean(max_f_scores)
        return avg_f_score, max_f_score

    def draw_gtscores(self, fold, keys, norm=True):
        """Draw datasets ground truth scores distribution in Tensorboard histograms"""
        for key in keys:
            d = self.dataset[key]
            i = int(key.split("_")[1])
            gtscore = d["gtscore"][...]
            if norm:
                gtscore -= gtscore.min()
                gtscore /= gtscore.max() - gtscore.min()
            self.hps.writer.add_histogram(
                f"{self.dataset_name}/Fold_{fold+1}/Train/gtscores", 
                gtscore, i)

    def draw_scores(self, fold, dist_scores):
        """Draw predicted scores distribution in Tensorboard histograms"""
        for key, scores in dist_scores.items():
            i = int(key.split("_")[1])
            self.hps.writer.add_histogram(
                f"{self.dataset_name}/Fold_{fold+1}/Train/final_scores",
                scores, i)

    def predict_dataset(self, pred_path, reload = True):
        """Predict on all videos in the dataset and save in hdfs5 file"""
        # Load best weights
        if reload:
            self.model.load_state_dict(self.best_weights)
        self.model.eval()
        
        # Create or open result hdfs5 file
        with h5py.File(pred_path, "w") as f:
            dataset_file = os.path.basename(self.hps.dataset_of_file[self.splits_file])
            g = f.create_group(dataset_file)
            
            # Get machine summary for each key
            for key in self.dataset.keys():
                # Get video data
                d = self.dataset[key]
                seq = d["features"][...]
                cps = d["change_points"][...]
                n_frames = d["n_frames"][()]
                nfps = d["n_frame_per_seg"][...].tolist()
                positions = d["picks"][...]
                user_summary = d["user_summary"][...]
                
                # Predict scores and compute machine summary/scores
                seq = torch.from_numpy(seq).unsqueeze(1)
                if self.hps.use_cuda:
                    seq = seq.cuda()
                scores = self.model(seq).squeeze().detach().cpu().numpy()
                machine_summary = generate_summary(scores, cps, n_frames, nfps, positions, self.hps.summary_proportion, self.hps.selection_algorithm)
                machine_scores = generate_scores(scores, n_frames, positions)
                
                # Save in hdfs5 file
                k = g.create_group(key)
                k.create_dataset("scores", data=scores)
                k.create_dataset("user_summary", data=user_summary)
                k.create_dataset("machine_summary", data=machine_summary)
                k.create_dataset("machine_scores", data=machine_scores)
    
    def predict_sample(self,pred_path, load = True, custom_weights = None):
        """Predict on all videos in the sample and save in hdfs5 file
        Args:
          pred_path: path to save the predictions
          load: if True, load the custom weights
          custom_weights: custom weights path to load"""
        # Load best weights
        if load:
            self.model.load_state_dict(self.custom_weights)
        self.model.eval()
        # Create or open result hdfs5 file
        with h5py.File(pred_path, "w") as f:
            dataset_file = os.path.basename(self.hps.dataset_of_file[self.splits_file])
            g = f.create_group(dataset_file)
            for key in self.dataset.keys():
                # Get video data
                d = self.dataset[key]
                seq = d["features"][...]
                cps = d["change_points"][...]
                n_frames = d["n_frames"][()]
                nfps = d["n_frame_per_seg"][...].tolist()
                positions = d["picks"][...]
                user_summary = d["user_summary"][...]
                
                # Predict scores and compute machine summary/scores
                seq = torch.from_numpy(seq).unsqueeze(1)
                if self.hps.use_cuda:
                    seq = seq.cuda()
                scores = self.model(seq).squeeze().detach().cpu().numpy()
                machine_summary = generate_summary(scores, cps, n_frames, nfps, positions, self.hps.summary_proportion, self.hps.selection_algorithm)
                machine_scores = generate_scores(scores, n_frames, positions)
                
                # Save in hdfs5 file
                k = g.create_group(key)
                k.create_dataset("scores", data=scores)
                k.create_dataset("user_summary", data=user_summary)
                k.create_dataset("machine_summary", data=machine_summary)
                k.create_dataset("machine_scores", data=machine_scores)
            
        pass

    def save_best_weights(self, weights_path):
        """Dump current best weights"""
        if self.best_weights is None:
            raise Exception("best_weights property is empty, can't save model's weights")
        torch.save(self.best_weights, weights_path)

    def load_weights(self, weights_path):
        """Load weights"""
        print(weights_path)
        self.model.load_state_dict(torch.load(weights_path))
