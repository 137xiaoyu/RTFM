from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import random
from utils import save_best_record
from model import Model
# from model_CIL import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(137)

    args = option.parser.parse_args()

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = Model(args.feature_size, args.batch_size)

    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint)
        print(f'{args.ckpt} loaded')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    auc, results = test(test_loader, model, args, None, device)

    raw_features = results['raw_features'].numpy()
    features = results['features'].numpy()
    labels = results['labels'].numpy()

    feature_save_dir = './features_84.289/'
    if not os.path.exists(feature_save_dir):
        os.makedirs(feature_save_dir)

    np.savetxt(os.path.join(feature_save_dir, args.dataset + '_raw_features.txt'), raw_features)
    np.savetxt(os.path.join(feature_save_dir, args.dataset + '_features.txt'), features)
    np.savetxt(os.path.join(feature_save_dir, args.dataset + '_labels.txt'), labels)
