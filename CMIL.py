import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from InfoNCE import InfoNCE


def CMIL(scores, visual_logits, seq_len, visual_rep):
    """
    Args:
        labels: (b)
        visual_logits: (b, n)
        seq_len: (b)
        visual_rep: (b, n, d)
    """
    visual_abn = torch.zeros(0).cuda()  # tensor([])
    visual_bgd = torch.zeros(0).cuda()  # tensor([])
    visual_nor = torch.zeros(0).cuda()
    for i in range(visual_logits.shape[0]):
        if scores[i] > 0.5:
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(torch.div(seq_len[i], 16, rounding_mode='trunc') + 1), largest=False)
            cur_visual_inverse_rep_topk = visual_rep[i][cur_visual_inverse_topk_indices]
            # cur_dim = cur_visual_inverse_rep_topk.size()
            # cur_visual_inverse_rep_topk = torch.mean(cur_visual_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_bgd = torch.cat((visual_bgd, cur_visual_inverse_rep_topk), 0)

            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(torch.div(seq_len[i], 16, rounding_mode='trunc') + 1), largest=True)
            cur_visual_rep_topk = visual_rep[i][cur_visual_topk_indices]
            # cur_dim = cur_visual_rep_topk.size()
            # cur_visual_rep_topk = torch.mean(cur_visual_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_abn = torch.cat((visual_abn, cur_visual_rep_topk), 0)
        else:
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(torch.div(seq_len[i], 16, rounding_mode='trunc') + 1), largest=False)
            cur_visual_inverse_rep_topk = visual_rep[i][cur_visual_inverse_topk_indices]
            # cur_dim = cur_visual_inverse_rep_topk.size()
            # cur_visual_inverse_rep_topk = torch.mean(cur_visual_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_bgd = torch.cat((visual_bgd, cur_visual_inverse_rep_topk), 0)

            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(torch.div(seq_len[i], 16, rounding_mode='trunc') + 1), largest=True)
            cur_visual_rep_topk = visual_rep[i][cur_visual_topk_indices]
            # cur_dim = cur_visual_rep_topk.size()
            # cur_visual_rep_topk = torch.mean(cur_visual_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_nor = torch.cat((visual_nor, cur_visual_rep_topk), 0)
    visual_abn_center = visual_abn.mean(dim=0, keepdim=True).expand(visual_abn.shape[0], -1)  # (1, d) expand
    visual_nor_center = visual_nor.mean(dim=0, keepdim=True).expand(visual_abn.shape[0], -1)  # (1, d) expand
    cmals = InfoNCE(negative_mode='unpaired')
    if visual_nor.size(0) == 0 or visual_abn.size(0) == 0:
        return torch.tensor(0.0), torch.tensor(0.0)
    else:
        loss_a2b = cmals(visual_abn, visual_abn_center, visual_bgd)
        loss_a2n = cmals(visual_abn, visual_abn_center, visual_nor)
        loss_n2b = cmals(visual_nor, visual_nor_center, visual_bgd)
        loss_n2a = cmals(visual_nor, visual_nor_center, visual_abn)
        return loss_a2b, loss_a2n, loss_n2b, loss_n2a
