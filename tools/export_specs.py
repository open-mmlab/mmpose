import argparse
import copy
import glob
import json
import os
import os.path as osp
import shutil
import time
from typing import Dict

from mmcv import Config, DictAction

YAML_DEFAULT_MIN_KP_SCORE = -99999.0
YAML_DEFAULT_OPT_BATCH_SIZE = 4
YAML_DEFAULT_MAX_BATCH_SIZE = 16

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--project_name',
        type=str,
        help='name of the project for lv usage, should match the folder name structure'
             'in the ml_models repo: e.g. \"brummer\"',
        required=True)
    parser.add_argument(
        '--author',
        type=str,
        help='full name of the Author of this training: e.g. \"Christian Holland\"',
        required=True)
    parser.add_argument(
        '--jira_task',
        type=str,
        help='shortened name of the Jira task for this training: e.g. \"OR-1926\"',
        required=True)
    args = parser.parse_args()


    return args

def read_json(path: str):
    return json.loads(
        open(path,
             "rb").read())

def write_string_as_file(path: str, string: str) -> None:
    f = open(path, "w")
    f.write(string)
    f.close()

def recreate_dir(dir_name: str):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def write_detector_yaml(cfg: Config, coco_file: Dict, write_dir: str, name: str) -> None:
    detector_name = "KeypointTrtDetector"
    heatmap_shape = cfg["data_cfg"]["heatmap_size"]
    input_shape = cfg["data_cfg"]["image_size"]
    export_name = f"{name}.onnx"
    onnx_file_path = f"/data/ml_models/models/keypoint_trt/{cfg.project_name}/{export_name}"
    kp_names = coco_file["categories"][0]["keypoints"]
    yaml_contents =  f"- !<{detector_name}>\n" \
                     f"  name: {name}\n" \
                     f"  heatmap_shape: {heatmap_shape}\n" \
                     f"  input_shape: {input_shape}\n" \
                     f"  onnx_file_path: {onnx_file_path}\n" \
                     f"  kp_names: {kp_names}\n" \
                     f"  min_kp_score: {YAML_DEFAULT_MIN_KP_SCORE}\n" \
                     f"  opt_batch_size: {YAML_DEFAULT_OPT_BATCH_SIZE}\n" \
                     f"  max_batch_size: {YAML_DEFAULT_MAX_BATCH_SIZE}\n"
    write_path = os.path.join(write_dir, "detectors.yaml")
    write_string_as_file(write_path, yaml_contents)

def write_info_file(cfg: Config, write_dir: str) -> None:
    result_file = os.path.join(cfg.work_dir, "best.json")
    result_contents = read_json(result_file)
    mAP = result_contents["best_score"]
    train_img_folder = cfg["data"]["train"]["img_prefix"]
    val_img_folder = cfg["data"]["val"]["img_prefix"]
    num_tain_img = len(glob.glob(f"{train_img_folder}/*"))
    num_val_img = len(glob.glob(f"{val_img_folder}/*"))
    date_trained = time.strftime("%m.%d.%Y %H:%M:%S")
    info_file = f" -- Keypoint training info -- \n" \
                f"mAP score: {mAP}\n" \
                f"Num train images: {num_tain_img}\n" \
                f"Num val images: {num_val_img}\n" \
                f"Date trained: {date_trained}\n" \
                f"Jira task: {cfg.jira_task}\n" \
                f"Author: {cfg.author}\n"
    write_path = os.path.join(write_dir, "model_info.txt")
    write_string_as_file(write_path, info_file)

def copy_training_specs(cfg, write_dir):
    shutil.copy(cfg.filename, os.path.join(write_dir, "config.txt"))



def export_for_lv(args):
    cfg = Config.fromfile(args.config)

    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    if args.project_name:
        cfg.project_name = args.project_name
    else:
        cfg.project_name = cfg["data"]["train"]["ann_file"].split("/")[1]
    cfg.jira_task = args.jira_task
    cfg.author = args.author
    export_folder = os.path.join(cfg.work_dir, "export")
    recreate_dir(export_folder)
    coco_file = read_json(cfg["data"]["train"]["ann_file"])
    model_name = f"keypoint_detector_{cfg.project_name}_{time.strftime('%y%m%d')}"
    write_detector_yaml(cfg=cfg, coco_file=coco_file, write_dir=export_folder, name=model_name)
    write_info_file(cfg=cfg, write_dir=export_folder)
    copy_training_specs(cfg=cfg, write_dir=export_folder)
    print(f"Training info exported successfully to: {export_folder}")
    model_output_path = os.path.join(export_folder, f"{model_name}.onnx")
    return model_output_path



if __name__ == '__main__':
    args = parse_args()
    export_for_lv(args)
