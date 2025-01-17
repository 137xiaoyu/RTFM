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
    # viz = Visualizer(env=args.dataset + ' 10 crop', use_incoming_socket=False)
    config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = Model(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = './log/'   # put your own path here
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    auc, results = test(test_loader, model, args, None, device)

    # viz.lines('scores', results['scores'])
    # viz.lines('gt', results['gt'])
    # viz.plot_lines('pr_auc', results['pr_auc'])
    # viz.plot_lines('auc', results['auc'])

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        losses = train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, None, device, args)

        if (step - 1) % args.plot_freq == 0:
            # viz.plot_lines('loss', losses['loss'].item())
            # viz.plot_lines('loss_a2b', losses['loss_a2b'].item())
            # viz.plot_lines('loss_a2n', losses['loss_a2n'].item())
            # viz.plot_lines('loss_n2b', losses['loss_n2b'].item())
            # viz.plot_lines('loss_n2a', losses['loss_n2a'].item())
            # viz.plot_lines('cls_loss', losses['cls_loss'].item())
            # viz.plot_lines('cls_loss2', losses['cls_loss2'].item())
            # viz.plot_lines('rtfm_loss', losses['rtfm_loss'].item())
            # viz.plot_lines('smooth loss', losses['smooth loss'].item())
            # viz.plot_lines('sparsity loss', losses['sparsity loss'].item())
            pass

        if step % 5 == 0 and step > 200:

            auc, results = test(test_loader, model, args, None, device)

            # viz.lines('scores', results['scores'])
            # viz.plot_lines('pr_auc', results['pr_auc'])
            # viz.plot_lines('auc', results['auc'])

            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model.state_dict(), './ckpt/' + args.dataset + '{}-i3d-{:.3f}.pkl'.format(step, best_AUC * 100))
                # save_best_record(test_info, os.path.join(output_path, args.dataset + '{}-step-AUC.txt'.format(step)))

            print(f'\nauc {auc * 100:.3f}\tbest {best_AUC * 100:.3f}')

    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
