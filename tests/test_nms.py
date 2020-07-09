import numpy as np

from mmpose.core.post_processing.nms import oks_nms, soft_oks_nms


def test_soft_oks_nms():
    oks_thr = 0.9
    kpts = []
    kpts.append({
        'keypoints': np.tile(np.array([10, 10, 0.9]), [17, 1]),
        'area': 100,
        'score': 0.9
    })
    kpts.append({
        'keypoints': np.tile(np.array([10, 10, 0.9]), [17, 1]),
        'area': 100,
        'score': 0.4
    })
    kpts.append({
        'keypoints': np.tile(np.array([100, 100, 0.9]), [17, 1]),
        'area': 100,
        'score': 0.7
    })

    keep = soft_oks_nms([kpts[i] for i in range(len(kpts))], oks_thr)
    assert (keep == np.array([0, 2, 1])).all()

    keep = oks_nms([kpts[i] for i in range(len(kpts))], oks_thr)
    assert (keep == np.array([0, 2])).all()
