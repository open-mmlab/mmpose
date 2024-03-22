import cv2
import numpy as np

def get_3D_corners(model_info_dict, obj_id):
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
    return corners


def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32') 

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    points_2D = points_2D.astype(np.float32)[0:8]
    points_3D = (points_3D).astype(np.float32)
    _, R_exp, t = cv2.solvePnP(points_3D,
                               points_2D.reshape((-1,1,2)),
                               cameraMatrix,
                               distCoeffs)                            

    R, _ = cv2.Rodrigues(R_exp)
    return R, t


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


def compute_projection(points_3D, transformation, internal_calibration):
    points_3D = points_3D.T
    tmp = np.array([1.]*8).reshape(1, 8)
    points_3D = np.concatenate((points_3D, tmp))
    
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

