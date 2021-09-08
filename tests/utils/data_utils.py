# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def convert_db_to_output(db, batch_size=2, keys=None, is_3d=False):
    outputs = []
    len_db = len(db)
    for i in range(0, len_db, batch_size):
        keypoints_dim = 3 if is_3d else 2
        keypoints = np.stack([
            np.hstack([
                db[j]['joints_3d'].reshape((-1, 3))[:, :keypoints_dim],
                db[j]['joints_3d_visible'].reshape((-1, 3))[:, :1]
            ]) for j in range(i, min(i + batch_size, len_db))
        ])

        image_paths = [
            db[j]['image_file'] for j in range(i, min(i + batch_size, len_db))
        ]
        bbox_ids = [j for j in range(i, min(i + batch_size, len_db))]
        box = np.stack([
            np.array([
                db[j]['center'][0], db[j]['center'][1], db[j]['scale'][0],
                db[j]['scale'][1],
                db[j]['scale'][0] * db[j]['scale'][1] * 200 * 200, 1.0
            ],
                     dtype=np.float32)
            for j in range(i, min(i + batch_size, len_db))
        ])

        output = {}
        output['preds'] = keypoints
        output['boxes'] = box
        output['image_paths'] = image_paths
        output['output_heatmap'] = None
        output['bbox_ids'] = bbox_ids

        if keys is not None:
            keys = keys if isinstance(keys, list) else [keys]
            for key in keys:
                output[key] = [
                    db[j][key] for j in range(i, min(i + batch_size, len_db))
                ]

        outputs.append(output)

    return outputs
