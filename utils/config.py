import os, sys
import shutil
import inspect
import logging
import datetime
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import parse_splits_filename
from models.rand import RandomTrainer
from models.logistic import LogisticRegressionTrainer
from models.vasnet import VASNetTrainer
from models.transformer import TransformerTrainer
from models.dsn import DSNTrainer
from models.sumgan import SumGANTrainer
from models.sumgan_att import SumGANAttTrainer
from models.pglsum import PGL_SUM


class HParameters:
    def __init__(self):
        self.use_cuda = 'yes'
        self.cuda_device = 1
        self.weight_decay = 0.00001
        self.lr = 0.00005
        self.epochs = 50
        self.test_every_epochs = 10

        # dataset
        self.datasets = [
           'datasets/summarizer_dataset_summe_google_pool5.h5',
           'datasets/summarizer_dataset_tvsum_google_pool5.h5',
           'datasets/fvs.h5'
        ]

        # default split files to be trained/tested on
        self.splits_files = 'all'

        # default model
        self.model_class = RandomTrainer

        # Dict containing extra parameters, possibly model-specific
        self.extra_params = None

        # summary length
        self.summary_proportion = 0.15

        # video segmentの選択
        self.selection_algorithm = 'knapsack'

        # logger default level is INFO
        # self.log_level = logging.INFO
        self.log_level = 'INFO'

        # Call _init() to initialize other attributes,use this to test the class
        # self._init()

# def load_from_args(self, args):
#         # any key from flags
#         for key in args:
#             val = args[key]
#             if val is not None:
#                 if hasattr(self, key) and isinstance(getattr(self, key), list):
#                     val = val.split(',')
#                 setattr(self, key, val)
    def load_from_args(self, args):
        # any key from flags
        for key in args:
            val = args[key]
            attr = hasattr(self, key)
            print(key, val, attr)

            if key == 'log_level':
                val = val.upper()  # Ensure log_level is in uppercase
            elif val is not None and hasattr(self, key) and isinstance(getattr(self, key), list):
                val = val.split(',')
                setattr(self, key, val)
            elif val is not None and hasattr(self, key) and isinstance(val, dict):
                setattr(self, key, val)
        
        # pick model
        self.model_class = {
            'random': RandomTrainer,
            'logistic': LogisticRegressionTrainer,
            'vasnet': VASNetTrainer,
            'transformer': TransformerTrainer,
            'dsn': DSNTrainer,
            'sumgan': SumGANTrainer,
            'sumgan_att': SumGANAttTrainer,
            'pglsum': PGL_SUM,
            None: RandomTrainer
        }.get(args.get('model', None), None)
        if self.model_class is None:
            raise KeyError(f"{args['model']} model is not unknown")

        # other dynamic properties
        self._init(splits_files=args.get('splits_files',"all"))

    def _init(self,splits_files='all'):
        # 実験名と出力先を指定
        log_dir = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        log_dir += '_' + self.model_class.__name__
        self.log_path = os.path.join('logs', log_dir)

        # Tensor Board
        self.writer = SummaryWriter(self.log_path)

        # cudaの扱いについて

        if self.use_cuda == 'default':
            self.use_cuda = torch.cuda.is_available()
        elif self.use_cuda == 'yes' or self.use_cuda:
            self.use_cuda = True
        else:
            self.use_cuda = False

        # deviceの指定
        if self.use_cuda:
            num_cuda_devices = torch.cuda.device_count()
            if self.cuda_device >= num_cuda_devices:
                # Adjust cuda_device if it's out of range
                self.cuda_device = num_cuda_devices - 1
                print("Warning: cuda_device index out of range. Adjusted to", self.cuda_device)
            torch.cuda.set_device(self.cuda_device)
        #     # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        # split file takes from args.spilt_files, or its going to be default to "all"
        self.splits_files = splits_files
        if self.splits_files == 'all':
            self.splits_files = [
                'splits/tvsum_splits.json',
                'splits/summe_splits.json',
                'splits/fvs_splits.json']
        elif self.splits_files == 'tvsum':
            self.splits_files = ['splits/tvsum_splits.json']
        elif self.splits_files == 'summe':
            self.splits_files = ['splits/summe_splits.json']
        if self.splits_files == 'fvs':
            self.splits_files = ['splits/fvs_splits.json'] 
        elif self.splits_files == 'dataset':
            self.splits_files = ['splits/dataset_splits.json']

        # file nameの管理リスト
        self.dataset_name_of_file = {}
        self.dataset_of_file = {}
        self.splits_of_file = {}

        for splits_file in self.splits_files:
            dataset_name, splits = parse_splits_filename(splits_file)
            # print("dataset_name", dataset_name)
            # print("splits", splits)
            self.dataset_name_of_file[splits_file] = dataset_name
            self.dataset_of_file[splits_file] = self.get_dataset_by_name(dataset_name).pop()
            # self.dataset_list = self.get_dataset_by_name(dataset_name)
            # if dataset_name is list:
            #     self.dataset_of_file[splits_file] = self.get_dataset_by_name(dataset_name).pop()
            # else:
            #     self.dataset_of_file[splits_file] = self.get_dataset_by_name(dataset_name)
            # print(self.dataset_of_file[splits_file])
            self.splits_of_file[splits_file] = splits
        
        # destination for weights and predictions on dataset
        self.weights_path = {}
        self.pred_path = {}
        for splits_file in self.splits_files:
            weights_file = f'{os.path.basename(splits_file)}.pth'
            self.weights_path[splits_file] = os.path.join(self.log_path, weights_file)
            pred_file = f"{os.path.basename(splits_file)}_preds.h5"
            self.pred_path[splits_file] = os.path.join(self.log_path, pred_file)

        # logの保管先のディレクトリが存在しない場合にディレクトリを生成
        os.makedirs(self.log_path, exist_ok=True)

        # logger
        self.logger = logging.getLogger("video_summarizer")
        fmt = logging.Formatter("%(asctime)s::%(levelname)s: %(message)s", "%H:%M:%S")
        ch = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(self.log_path, "train.log"))
        ch.setFormatter(fmt)
        fh.setFormatter(fmt)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        self.logger.setLevel(getattr(logging, self.log_level.upper()))

        # modelをlog directoryの保存
        src = inspect.getfile(self.model_class)
        dst = os.path.join(self.log_path, os.path.basename(src))
        shutil.copyfile(src, dst)
    
    def get_dataset_by_name(self, dataset_name):
        for d in self.datasets:
            if dataset_name in d:
                return [d]
        return None
    
    def __str__(self):
        # ハイパーパラメータを表示
        vars = ["use_cuda", "cuda_device", "log_level","model_class",
                "weight_decay", "lr", "epochs",
                "summary_proportion", "selection_algorithm",
                "log_path", "splits_files", "extra_params"]
        info_str = ""

        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            if var == "cuda_device":
                val = "GPU" if val == 0 else "CPU"
            if var == "model_class":
                val = val.__name__
            info_str += "["+str(i)+"] "+var+": "+str(val)
            info_str += "\n" if i < len(vars)-1 else ""

        return info_str

    def get_full_hps_dict(self):
        """Returns the list of hyperparameters as a flat dict"""
        vars = ["weight_decay", "lr", "epochs"]

        hps = {}
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            hps[var] = val

        return hps
    def save_to_file(self, path):
        """ save the hyperparameters to a file(json), to load the model later"""
        # with open(path, 'w') as f:
        #     json.dump(self.__dict__, f)
        pass

if __name__ == "__main__":
    # Check default values
    hps = HParameters()
    # print(hps.)
    print(hps)
    # Check update with args works well
    args = {
        "root": "root_dir",
        "datasets": "set1,set2,set3",
        "splits": "split1, split2",
        "new_param_float": 1.23456
    }
    hps.load_from_args(args)
    print(hps)
