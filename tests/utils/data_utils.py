# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmpose.core.bbox.transforms import bbox_cs2xywh


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

        # Get bbox directly or from center+scale
        if 'bbox' in db[i]:
            boxes = np.stack(
                [db[j]['bbox'] for j in range(i, min(i + batch_size, len_db))])
        else:
            assert 'center' in db[i] and 'scale' in db[i]
            boxes = np.stack([
                bbox_cs2xywh(db[j]['center'], db[j]['scale'])
                for j in range(i, min(i + batch_size, len_db))
            ])

        # Add bbox area and score
        boxes = np.concatenate(
            (boxes, np.prod(boxes[:, 2:4], axis=1,
                            keepdims=True), np.ones((boxes.shape[0], 1))),
            axis=1)

        output = {}
        output['preds'] = keypoints
        output['boxes'] = boxes
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
