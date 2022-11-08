import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
from CMA_MIL import CMAL
from CMIL import CMIL


def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin

    def forward(self, feat_n, feat_a):
        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        return self.alpha * loss_rtfm


# class RTFM_loss(torch.nn.Module):
#     def __init__(self, alpha, margin):
#         super(RTFM_loss, self).__init__()
#         self.alpha = alpha
#         self.margin = margin
#         self.criterion = torch.nn.MarginRankingLoss(margin)

#     def forward(self, fm_n, fm_a):
#         fm_n = torch.mean(fm_n, dim=1)  # (b, 3) -> (b)
#         fm_a = torch.mean(fm_a, dim=1)  # (b, 3) -> (b)

#         loss_rtfm = self.criterion(fm_a, fm_n, torch.ones_like(fm_a)) / fm_a.shape[0]

#         return self.alpha * loss_rtfm


def train(nloader, aloader, model, batch_size, optimizer, viz, device, args):
    with torch.set_grad_enabled(True):
        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)
        label = torch.cat((nlabel, alabel), 0).cuda()  # (b)

        input = torch.cat((ninput, ainput), 0).to(device)

        # bs, ncrops, t, f = input.size()
        seq_len = torch.sum(torch.max(torch.abs(input[:, 0]), dim=2)[0] > 0, 1)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores, fm_select_abn, fm_select_nor, feat_magnitudes, features, attn, neg_log_likelihood, cls_scores = model(input, labels=torch.cat((nlabel, alabel), 0).cuda())  # b*32  x 2048
        score = torch.cat((score_normal, score_abnormal), 0).squeeze()  # (b, 1) mean of (b, 3, 1) -> (b)
        cls_scores = cls_scores.squeeze()  # (b, 1, 1) -> (b)

        scores = scores.view(batch_size * 32 * 2)  # (b, n, 1) -> (b * n)

        abn_scores = scores[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        cls_criterion = torch.nn.BCELoss()
        rtfm_criterion = RTFM_loss(0.0001, 100)
        # rtfm_criterion = RTFM_loss(0.2, 100)
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)
        cls_loss = cls_criterion(score, label)
        cls_loss2 = cls_criterion(cls_scores, label)

        rtfm_loss = rtfm_criterion(feat_select_normal, feat_select_abn)
        # rtfm_loss = rtfm_criterion(fm_select_nor, fm_select_abn)
        loss_a2b, loss_a2n, loss_n2b, loss_n2a = CMIL(label, scores.view(batch_size * 2, 32), seq_len, features.view(batch_size * 2, 10, 32, -1).mean(dim=1))

        loss_a2b = loss_a2b * 0.1
        loss_a2n = loss_a2n * 0.2
        loss_n2b = loss_n2b * 0.1
        loss_n2a = loss_n2a * 0.2

        cost = cls_loss + loss_smooth + loss_sparse

        # viz.plot_lines('loss', cost.item())
        # viz.plot_lines('loss_a2b', loss_a2b.item())
        # viz.plot_lines('loss_a2n', loss_a2n.item())
        # viz.plot_lines('cls_loss', cls_loss.item())
        # viz.plot_lines('rtfm_loss', rtfm_loss.item())
        # viz.plot_lines('neg_log_likelihood', neg_log_likelihood.item())
        # viz.plot_lines('smooth loss', loss_smooth.item())
        # viz.plot_lines('sparsity loss', loss_sparse.item())
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    losses = {
        'loss': cost,
        'loss_a2b': loss_a2b,
        'loss_a2n': loss_a2n,
        'loss_n2b': loss_n2b,
        'loss_n2a': loss_n2a,
        'cls_loss': cls_loss,
        'cls_loss2': cls_loss2,
        'rtfm_loss': rtfm_loss,
        'neg_log_likelihood': neg_log_likelihood,
        'smooth loss': loss_smooth,
        'sparsity loss': loss_sparse
    }

    return losses


# def train(nloader, aloader, model, batch_size, optimizer, viz, device):
#     criterion = torch.nn.BCELoss()
#     lamda_a2b = 1.0
#     lamda_a2n = 1.0
#     with torch.set_grad_enabled(True):
#         model.train()

#         ninput, nlabel = next(nloader)
#         ainput, alabel = next(aloader)

#         input = torch.cat((ninput, ainput), 0).to(device)
#         bs, ncrops, t, f = input.size()
#         label = torch.cat([torch.repeat_interleave(nlabel, ncrops), torch.repeat_interleave(alabel, ncrops)], dim=0).to(device)

#         seq_len = torch.sum(torch.max(torch.abs(input.view(-1, t, f)), dim=2)[0] > 0, 1)

#         mmil_logits, visual_logits, visual_rep = model(input, seq_len)
#         visual_logits = visual_logits.squeeze()
#         mmil_logits = mmil_logits.squeeze()
#         clsloss = criterion(mmil_logits, label)
#         cmaloss_a2b, cmaloss_a2n = CMAL(mmil_logits, visual_logits, seq_len, visual_rep)
#         total_loss = clsloss + lamda_a2b * cmaloss_a2b + lamda_a2n * cmaloss_a2n
#         # total_loss = clsloss

#         viz.plot_lines('loss', total_loss.item())
#         viz.plot_lines('cmaloss_a2b', cmaloss_a2b.item())
#         viz.plot_lines('cmaloss_a2n', cmaloss_a2n.item())
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
