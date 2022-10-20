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

viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(137)

    args = option.parser.parse_args()
    # config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch + 10, eta_min=0)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = './log/'   # put your own path here
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    auc = test(test_loader, model, args, viz, device)
    print('Random initialized AUC: {:.4f}\n'.format(auc))

    for epoch in range(args.max_epoch):

        train(train_aloader, train_nloader, model, args.batch_size, optimizer, viz, device, args)

        scheduler.step()

        auc = test(test_loader, model, args, viz, device)
        test_info["epoch"].append(epoch + 1)
        test_info["test_AUC"].append(auc)

        if test_info["test_AUC"][-1] > best_AUC:
            best_AUC = test_info["test_AUC"][-1]
            torch.save(model.state_dict(), './ckpt/' + args.model_name + '{}-i3d.pkl'.format(epoch + 1))
            save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(epoch + 1)))

        print(f'\nEpoch {epoch + 1}/{args.max_epoch}\tauc {auc * 100:.3f}\tbest {best_AUC * 100:.3f}')

    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
