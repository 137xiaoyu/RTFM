import os
import glob

if __name__ == '__main__':
    # root_path = 'D:/137/dataset/VAD/UCF_Crime/features/I3D-10crop/Train/'
    root_path = 'D:/137/dataset/VAD/UCF_Crime/features/I3D-10crop/Test/'
    # root_path = '/home/wucx/dataset/VAD/UCF_Crime/features/I3D-10crop/Train/'
    # root_path = '/home/wucx/dataset/VAD/UCF_Crime/features/I3D-10crop/Test/'
    files = sorted(glob.glob(os.path.join(root_path, '*.npy')))

    count = 0
    # with open('./list/ucf-i3d.list', 'w+') as f:
    with open('./list/ucf-i3d-test.list', 'w+') as f:
        normal = []
        for file in files:
            if 'Normal_' in file:
                normal.append(file)
            else:
                count += 1
                newline = file + '\n'
                f.write(newline)

        for file in normal:
            newline = file + '\n'
            f.write(newline)

    print(count)
