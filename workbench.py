import h5py
from utils.config import HParameters
from utils import Proportion
import torch
import models
from torchsummary import summary
from models.rand import RandomTrainer
from models.logistic import LogisticRegressionTrainer
from models.vasnet import VASNetTrainer
from models.transformer import TransformerTrainer, Transformer
from models.dsn import DSNTrainer
from models.sumgan import SumGANTrainer
from models.sumgan_att import SumGANAttTrainer
import h5py

filename = 'sample_feature/sample_GoogleNet.h5'
# filename = 'logs/1707737585_TransformerTrainer/summe_splits.json_preds.h5'
def infer(filename = 'sample_feature/sample_GoogleNet.h5'):
    data = None
    with h5py.File(filename, 'r') as h5file:
        for k in h5file.keys():
            data = h5file[k][:]
            data = torch.from_numpy(data)
            with torch.no_grad():
                pred = model.model(data[None, ...])
                print(f'{k} -> x_dims {data.size()} -> y_dims {pred.size()}')


hps = HParameters()
print(hps)
hps.extra_params = {'encoder_layers': 6}
def load_model(model_weights_path):
    # Load model weights
    model_state_dict = torch.load(model_weights_path)
    # Create an instance of the model class
    model = TransformerTrainer(hps, splits_file='splits/summe_splits.json')
    # model.load_state_dict(model_state_dict)
    # Set model to evaluation mode
    # model.eval()
    return model


model_weights_path = '/mnt/g/Github/video_summarizer/logs/1707737585_TransformerTrainer/summe_splits.json.pth'
model = load_model(model_weights_path)
# model.load_weights(model_weights_path)
print(f'model loaded successfully from {model_weights_path}')
infer()
