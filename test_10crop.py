import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)  # (b, 10, n, f)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, logits, feat_magnitudes, features, attn, neg_log_likelihood = model(inputs=input)
            logits = torch.squeeze(logits, 2)  # (1, n, 1) -> (1, n)
            logits = torch.squeeze(logits, 0)  # (1, n) -> (n)
            sig = logits
            pred = torch.cat((pred, sig))

        if args.dataset == 'sht':
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == 'ucf':
            gt = np.load('list/gt-ucf.npy')
        elif args.dataset == 'xdv':
            gt = np.load('list/gt-xdv.npy')
        elif args.dataset == 'my_ucf':
            gt = np.load('list/my_gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        # np.save('fpr.npy', fpr)
        # np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        # np.save('precision.npy', precision)
        # np.save('recall.npy', recall)
        # viz.plot_lines('pr_auc', pr_auc)
        # viz.plot_lines('auc', rec_auc)
        # viz.lines('scores', pred)
        # viz.lines('gt', gt)
        # viz.lines('attn', attn[0])
        # viz.lines('roc', tpr, fpr)
        # viz.lines('p-r', precision, recall)

        results = {
            'pr_auc': pr_auc,
            'auc': rec_auc,
            'scores': pred,
            'gt': gt,
            'roc': (tpr, fpr),
            'p-r': (precision, recall)
        }

        return rec_auc, results


# def test(dataloader, model, args, viz, device):
#     with torch.no_grad():
#         model.eval()
#         pred = torch.zeros(0, device=device)

#         for i, input in enumerate(dataloader):
#             input = input.to(device)
#             input = input.permute(0, 2, 1, 3)
#             bs, ncrops, t, f = input.size()
#             seq_len = torch.sum(torch.max(torch.abs(input.view(-1, t, f)), dim=2)[0] > 0, 1)
#             mmil_logits, visual_logits, visual_rep = model(input, seq_len)
#             logits = torch.squeeze(visual_logits, 1)
#             logits = torch.mean(logits, 0)
#             sig = logits
#             pred = torch.cat((pred, sig))

#         if args.dataset == 'shanghai':
#             gt = np.load('list/gt-sh.npy')
#         else:
#             gt = np.load('list/gt-ucf.npy')

#         pred = list(pred.cpu().detach().numpy())
#         pred = np.repeat(np.array(pred), 16)

#         fpr, tpr, threshold = roc_curve(list(gt), pred)
#         # np.save('fpr.npy', fpr)
#         # np.save('tpr.npy', tpr)
#         rec_auc = auc(fpr, tpr)
#         print('auc : ' + str(rec_auc))

#         precision, recall, th = precision_recall_curve(list(gt), pred)
#         pr_auc = auc(recall, precision)
#         # np.save('precision.npy', precision)
#         # np.save('recall.npy', recall)
#         viz.plot_lines('pr_auc', pr_auc)
#         viz.plot_lines('auc', rec_auc)
#         viz.lines('scores', pred)
#         viz.lines('roc', tpr, fpr)
#         return rec_auc
