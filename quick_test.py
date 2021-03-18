import mmcv
import numpy as np

from mmpose.core.camera import SimpleCamera


def main():
    cam_params = mmcv.load('data/h36m/annotations/cameras.pkl')
    data = np.load('data/h36m/annotations/h36m_train_new.npz')

    param1 = cam_params[('S1', '54138969')]
    cam1 = SimpleCamera(param=param1)
    X_c1 = data['S'][0, :, :3] * 1000
    # param2 = cam_params[('S1', '55011271')]
    # cam2 = SimpleCamera(param=param2)
    # X_c2 = data['S'][277, :, :3] * 1000

    X_p1 = cam1.camera_to_pixel(X_c1)
    P1 = data['part'][0, :, :2]
    print(X_c1)
    print('#' * 30)
    print(X_p1)
    print('#' * 30)
    print(P1)
    print('#' * 30)
    print(np.abs(X_p1 - P1))


if __name__ == '__main__':
    main()
