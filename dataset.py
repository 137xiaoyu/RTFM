import os
import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        if self.dataset == 'sht':
            if test_mode:
                self.rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
            else:
                self.rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
        elif self.dataset == 'ucf':
            if test_mode:
                self.rgb_list_file = 'list/ucf-i3d-test.list'
            else:
                self.rgb_list_file = 'list/ucf-i3d.list'
        elif self.dataset == 'xdv':
            if test_mode:
                self.rgb_list_file = 'list/xdv-i3d-test.list'
            else:
                self.rgb_list_file = 'list/xdv-i3d.list'
        elif self.dataset == 'my_ucf':
            if test_mode:
                self.rgb_list_file = 'list/my_ucf-i3d-test.list'
            else:
                self.rgb_list_file = 'list/my_ucf-i3d.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'sht':
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
                    print(self.list)
            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)
            elif self.dataset == 'xdv':
                if self.is_normal:
                    self.list = self.list[1905:]
                    print('normal list for xdv')
                    print(self.list)
                else:
                    self.list = self.list[:1905]
                    print('abnormal list for xdv')
                    print(self.list)
            elif self.dataset == 'my_ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1

        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)  # (t, 10, f) or (t, f)

        # if self.dataset == 'sht':
        #     features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        #     features = np.array(features, dtype=np.float32)  # (t, 10, f)
        # elif self.dataset == 'ucf':
        #     tmp_list = [self.list[index].strip('\n')]
        #     root, ext = os.path.splitext(tmp_list[0])
        #     for i in range(1, 10):
        #         tmp_list.append(root + '__' + str(i) + ext)
        #     features = []
        #     for path in tmp_list:
        #         features.append(np.load(path, allow_pickle=True))
        #     # if tmp_list[0] in [
        #     #         'D:/137/dataset/VAD/UCF_Crime/features/I3D/Train/RGB\\Training_Normal_Videos_Anomaly\\Normal_Videos653_x264.npy',
        #     #         'D:/137/dataset/VAD/UCF_Crime/features/I3D/Train/RGB\\Stealing\\Stealing074_x264.npy',
        #     #         'D:/137/dataset/VAD/UCF_Crime/features/I3D/Train/RGB\\Shooting\\Shooting044_x264.npy'
        #     # ]:
        #         # features[0] = features[0][:-1]
        #     try:
        #         features = np.stack(features).astype(np.float32).transpose(1, 0, 2)  # (10, t, f) -> (t, 10, f)
        #     except:
        #         features[0] = features[0][:-1]
        #         features = np.stack(features).astype(np.float32).transpose(1, 0, 2)  # (10, t, f) -> (t, 10, f)
        # elif self.dataset == 'xdv':
        #     tmp_list = [self.list[index].strip('\n')]
        #     root, ext = os.path.splitext(tmp_list[0])
        #     root = root[:-1]
        #     for i in range(1, 5):
        #         tmp_list.append(root + str(i) + ext)
        #     features = []
        #     for path in tmp_list:
        #         features.append(np.load(path, allow_pickle=True))
        #     features = np.stack(features).astype(np.float32).transpose(1, 0, 2)  # (10, t, f) -> (t, 10, f)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # (t, 10, f) -> (10, t, f)
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
