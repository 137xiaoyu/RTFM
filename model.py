import torch
import torch.nn as nn
import torch.nn.init as torch_init
from Transformer import *
torch.set_default_tensor_type('torch.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)



class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)


    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        residual = out

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)

        out3 = self.conv_3(out)
        out_d = torch.cat((out1, out2, out3), dim = 1)
        out = self.conv_4(out)
        out = self.non_local(out)
        out = torch.cat((out_d, out), dim=1)
        out = self.conv_5(out)   # fuse all the features together
        out = out + residual
        out = out.permute(0, 2, 1)
        # out: (B, T, 1)

        return out


class TemporalConsensus(nn.Module):
    def __init__(self, len_feature, hid_dim):
        super().__init__()

        nhead = 4
        dropout = 0.1
        ffn_dim = hid_dim
        bn = nn.BatchNorm1d
        self.hid_dim = hid_dim
        self.len_feature = len_feature

        self.fc_v = nn.Linear(self.len_feature, hid_dim)
        self.cma = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))

        # self.fc_v2 = nn.Linear(self.len_feature, hid_dim)
        self.conv_1 = nn.Sequential(nn.Conv1d(in_channels=self.hid_dim, out_channels=self.hid_dim, kernel_size=3, stride=1, dilation=1, padding=1), bn(self.hid_dim), nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv1d(in_channels=self.hid_dim, out_channels=self.hid_dim, kernel_size=3, stride=1, dilation=2, padding=2), bn(self.hid_dim), nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv1d(in_channels=self.hid_dim, out_channels=self.hid_dim, kernel_size=3, stride=1, dilation=4, padding=4), bn(self.hid_dim), nn.ReLU())

    def forward(self, x):
        # x: (B, T, F)

        x1 = self.fc_v(x)
        out_long = self.cma(x1)

        # x2 = self.fc_v2(x)
        x2 = x1.permute(0, 2, 1)
        out1 = self.conv_1(x2) + x2
        out2 = self.conv_2(x2) + x2
        out3 = self.conv_3(x2) + x2
        out_short = torch.cat([out1, out2, out3], dim=1)
        out_short = out_short.permute(0, 2, 1)

        return torch.cat([out_long, out_short], dim=2)


class AttentionMIL(nn.Module):

    def __init__(self, D, input_dim):
        super(AttentionMIL, self).__init__()
        self.L = input_dim
        self.D = D
        self.K = 1

        self.softmax = nn.Softmax(dim=2)

        self.attention = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K))

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

    def forward(self, x):
        # (b, n, L)
        A = self.attention(x)  # (b, n, 1)
        A = torch.transpose(A, 2, 1)  # (b, 1, n)
        A = self.softmax(A)  # softmax over N

        M = torch.bmm(A, x)  # (b, 1, L)

        Y_prob = self.classifier(M).squeeze(-1).squeeze(-1)  # (b, 1, 1) -> (b)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A.squeeze(1)

    def calculate(self, X, Y):
        Y = Y.float()

        Y_prob, Y_hat, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return neg_log_likelihood.mean(), A, error, Y_prob, Y_hat


class Model(nn.Module):
    def __init__(self, n_features, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        hid_dim = 128
        self.temporal_consensus = TemporalConsensus(n_features, hid_dim)
        self.fc = nn.Linear(hid_dim * 4, 1)

        # self.attn_model = AttentionMIL(256, hid_dim * 4)

        # self.Aggregate = Aggregate(len_feature=n_features)
        # self.fc1 = nn.Linear(n_features, 512)
        # self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs, labels=None):

        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        bs, ncrops, t, f = out.size()

        out = out.view(-1, t, f)

        out = self.temporal_consensus(out)
        f = out.shape[-1]

        # out = self.Aggregate(out)

        out = self.drop_out(out)

        neg_log_likelihood = torch.tensor(0.)
        attn = torch.zeros(bs, t)
        # if labels != None:
        #     neg_log_likelihood, attn, _, _, _ = self.attn_model.calculate(out.view(bs, ncrops, t, f).mean(1), labels)
        # else:
        #     _, _, attn = self.attn_model(out.view(bs, ncrops, t, f).mean(1))

        features = out
        scores = self.sigmoid(self.fc(features))
        # scores = self.relu(self.fc1(features))
        # scores = self.drop_out(scores)
        # scores = self.relu(self.fc2(scores))
        # scores = self.drop_out(scores)
        # scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)

        normal_features = features[0:self.batch_size*10]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size*10:]
        abnormal_scores = scores[self.batch_size:]

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        # feat_magnitudes = torch.norm(features, p=1, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        select_idx = torch.ones_like(nfea_magnitudes)
        select_idx = self.drop_out(select_idx)

        #######  process abnormal videos -> select top3 feature magnitude  #######
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
        # idx_abn = torch.topk(attn[self.batch_size:] * self.drop_out(torch.ones_like(attn[self.batch_size:])), k_abn, dim=1)[1]
        # idx_abn = torch.topk(scores[self.batch_size:] * self.drop_out(torch.ones_like(scores[self.batch_size:])), k_abn, dim=1)[1].squeeze(-1)
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2,3)

        total_select_abn_feature = torch.zeros(0, device=inputs.device)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude


        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes)
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        # idx_normal = torch.topk(attn[:self.batch_size] * self.drop_out(torch.ones_like(attn[:self.batch_size])), k_nor, dim=1)[1]
        # idx_normal = torch.topk(scores[:self.batch_size] * self.drop_out(torch.ones_like(scores[:self.batch_size])), k_nor, dim=1)[1].squeeze(-1)
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0, device=inputs.device)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        """
        Returns:
            score_abnormal: (b, 1) mean of (b, 3, 1)
            score_normal: (b, 1) mean of (b, 3, 1)
            feat_select_abn: (b * 10, 3, d)
            feat_select_normal: (b * 10, 3, d)
            scores: (b, n, 1)
            feat_magnitudes: (b * 10, n, 1)
            feat_magnitudes: (b * 10, n, d)
            attn: (b, n)
        以上不是准确的shape, 可以看出意义
        """
        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores, feat_magnitudes, features, attn, neg_log_likelihood
