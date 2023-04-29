import os
import mmengine
import argparse
import numpy as np
import mmcv

def parse_examples(data_file):
    if not os.path.isfile(data_file):
        print(f'Error: file {data_file} does not exist!')
        return None

    with open(data_file) as fid:
        data_examples = [example.strip() for example in fid if example != '']

    return data_examples

def images_info(object_path, data_examples):
    all_images_path = os.path.join(object_path, 'rgb')
    all_filenames = [
        filename for filename in os.listdir(all_images_path)
        if '.png' in filename and filename.replace('.png', '') in data_examples
        ]
    image_paths = [
        os.path.join(all_images_path, filename) for filename in all_filenames
        ]
    images = []
    for id, image_path in enumerate(image_paths):
        img = mmcv.imread(image_path)
        height = img.shape[0]
        width = img.shape[1]
        images.append(dict(file_name=all_filenames[id],
                           height=height,
                           width=width,
                           id=id))
    return images

def project_points_3D_to_2D(points_3D, rotation_vector, translation_vector,
                            camera_matrix):
    points_3D = points_3D.reshape(3,1)
    rotation_vector = rotation_vector.reshape(3,3)
    translation_vector = translation_vector.reshape(3,1)
    pixel = camera_matrix.dot(
        rotation_vector.dot(points_3D)+translation_vector)
    pixel /= pixel[-1]
    points_2D = pixel[:2]
    
    return points_2D

def insert_np_cam_calibration(filtered_infos):
    for info in filtered_infos:
        info['cam_K_np'] = np.reshape(np.array(info['cam_K']), newshape=(3, 3))

    return filtered_infos

def get_bbox_from_mask(mask, mask_value=None):
    if mask_value is None:
        seg = np.where(mask != 0)
    else:
        seg = np.where(mask == mask_value)
    # check if mask is empty
    if seg[0].size <= 0 or seg[1].size <= 0:
        return np.zeros((4, ), dtype=np.float32), False
    min_x = np.min(seg[1])
    min_y = np.min(seg[0])
    max_x = np.max(seg[1])
    max_y = np.max(seg[0])

    return np.array([min_x, min_y, max_x-min_x, max_y-min_y], dtype=np.float32)

def annotations_info(object_path, data_examples, gt_dict, info_dict,
                     model_info_dict, obj_id):
    all_images_path = os.path.join(object_path, 'rgb')
    all_filenames = [
        filename for filename in os.listdir(all_images_path)
        if '.png' in filename and filename.replace('.png', '') in data_examples
    ]
    image_paths = [
        os.path.join(all_images_path, filename) for filename in all_filenames
    ]
    mask_paths = [
        image_path.replace('rgb', 'mask') for image_path in image_paths
    ]
    
    example_ids = [int(filename.split('.')[0]) for filename in all_filenames]
    filtered_gt_lists = [gt_dict[key] for key in example_ids]
    filtered_gts = []
    for gt_list in filtered_gt_lists:
        all_annos = [anno for anno in gt_list if anno['obj_id'] == int(obj_id)]
        if len(all_annos) <= 0:
            print('\nError: No annotation found!')
            filtered_gts.append(None)
        elif len(all_annos) > 1:
            print('\nWarning: found more than one annotation.\
                    using only the first annotation')
            filtered_gts.append(all_annos[0])
        else:
            filtered_gts.append(all_annos[0])
            
    filtered_infos = [info_dict[key] for key in example_ids]
    info_list = insert_np_cam_calibration(filtered_infos)
    
    id = 0
    annotations = []
    # 获取bbox与keypoints
    for gt, info, mask_path in zip(filtered_gts, info_list, mask_paths):
        mask = mmcv.imread(mask_path)
        annotation = {}
        annotation['category_id'] = 1
        annotation['segmentation'] = []
        annotation['iscrowd'] = 0
        annotation['image_id'] = id
        annotation['id'] = id # 因为图片中只有一个物体，所以image_id=id
        bbox = get_bbox_from_mask(mask)
        annotation['bbox'] = bbox
        annotation['area'] = bbox[2] * bbox[3]
        annotation['num_keypoints'] = 8
        
        # keypoints中 不存在的关键点为[0,0] 关键点的第三位是0 没有标注点 1 遮挡点 2正常点
        min_x = model_info_dict[int(obj_id)]['min_x']
        min_y = model_info_dict[int(obj_id)]['min_y']
        min_z = model_info_dict[int(obj_id)]['min_z']
        max_x = min_x + model_info_dict[int(obj_id)]['size_x']
        max_y = min_y + model_info_dict[int(obj_id)]['size_y']
        max_z = min_z + model_info_dict[int(obj_id)]['size_z']
        corners = np.array([[max_x, max_y, min_z],
                            [max_x, max_y, max_z],
                            [max_x, min_y, min_z],
                            [max_x, min_y, max_z],
                            [min_x, max_y, min_z],
                            [min_x, max_y, max_z],
                            [min_x, min_y, min_z],
                            [min_x, min_y, max_z]])
        corners = [
            project_points_3D_to_2D(corner, np.array(gt['cam_R_m2c']),
                                    np.array(gt['cam_t_m2c']),
                                    info['cam_K_np'])
            for corner in corners]
        corners = np.array(corners).reshape(8,2)
        tmp = np.array([2]*8).reshape(8,1)
        corners = np.hstack((corners, tmp))
        corners = corners.reshape(-1)
        annotation['keypoints'] = corners
        
        id += 1
        annotations.append(annotation)
    return annotations

def parse_args():
    parser = argparse.ArgumentParser(description='Create_linemod_json')
    parser.add_argument('--root', help='root path')
    parser.add_argument('--id', type=str, help='object id, for example: 01')
    parser.add_argument('--mode', type=str, help='mode, for example: train')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    object_path = os.path.join(args.root, f'data/{args.id}/')
    data_examples = parse_examples(object_path + args.mode + '.txt')
    gt_dict = mmengine.load(object_path + 'gt.yml')
    info_dict = mmengine.load(object_path + 'info.yml')
    obj_id = args.id
    model_info_path = args.root + 'models/'
    model_info_dict = mmengine.load(model_info_path + 'models_info.yml')
    
    # images
    images = images_info(object_path, data_examples)
    
    # annotations
    annotations = annotations_info(object_path, data_examples, 
                                   gt_dict, info_dict, model_info_dict,
                                   obj_id)
    
    # categories
    object = [{
        'supercatgory': 'ape',
        'id': 1,
        'name': 'ape',
        'keypoints': [
            'min_min_min', 'min_min_max',
            'min_max_min', 'min_max_max',
            'max_min_min', 'max_min_max',
            'max_max_min', 'max_max_max'],
        'skeleton': [[0, 4], [1, 5], [3, 7], [6, 2],
                     [0, 2], [1, 3], [7, 5], [4, 6],
                     [0, 1], [7, 6], [5, 4], [2, 3]],
    }]
    
    # remove invalid data
    linemod_coco = {
        'categories': object,
        'images': images,
        'annotations': annotations
    }
    out_file = args.root + 'json/linemod_preprocessed_'+ args.mode + '.json'
    mmengine.dump(linemod_coco, out_file)

if __name__ == '__main__':
    main()