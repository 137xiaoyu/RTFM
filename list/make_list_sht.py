import os
import glob

if __name__ == '__main__':
    abnormal_path = 'D:/137/dataset/VAD/ShanghaiTech/annotations/test_frame_mask/'
    root_path = 'D:/137/dataset/VAD/ShanghaiTech/features/SH_Train_ten_crop_i3d/'
    # root_path = 'D:/137/dataset/VAD/ShanghaiTech/features/SH_Test_ten_crop_i3d/'
    # abnormal_path = '/home/wucx/dataset/VAD/ShanghaiTech/annotations/test_frame_mask/'
    # root_path = '/home/wucx/dataset/VAD/ShanghaiTech/features/SH_Train_ten_crop_i3d/'
    # root_path = '/home/wucx/dataset/VAD/ShanghaiTech/features/SH_Test_ten_crop_i3d/'
    files = sorted(glob.glob(os.path.join(root_path, '*.npy')))

    abnormal_files = sorted(glob.glob(os.path.join(abnormal_path, '*.npy')))
    abnormal_files = [os.path.splitext(os.path.basename(ab_file))[0] + '_i3d' for ab_file in abnormal_files]

    count = 0
    with open('./list/shanghai-i3d-train-10crop.list', 'w+') as f:
    # with open('./list/shanghai-i3d-test-10crop.list', 'w+') as f:
        normal = []
        for file in files:
            if os.path.splitext(os.path.basename(file))[0] not in abnormal_files:
                normal.append(file)
            else:
                count += 1
                newline = file + '\n'
                f.write(newline)

        for file in normal:
            newline = file + '\n'
            f.write(newline)

    print(count)
