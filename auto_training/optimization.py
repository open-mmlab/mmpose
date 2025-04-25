# Copyright (c) OpenMMLab. All rights reserved.
import optuna
import os
import os.path as osp
import argparse
import os
import os.path as osp
from objective import train
from multiprocessing import Process
import subprocess


def objective(trial):
    # # Define the hyperparameter search space
    res = trial.suggest_categorical("resolution", [384])  # square resolutions
    augmentation_index = trial.suggest_categorical("augmentation_index", [0])  # Indices for different augmentations
    batch_size = trial.suggest_categorical("batch_size", [64])  # Batch size
    repeat_times = trial.suggest_categorical("repeat_times", [1])  # Batch size
    resnet_depth = trial.suggest_categorical("resnet_depth", [18])  # backbone resnet depth
    backbone_type = trial.suggest_categorical("backbone_type", ["resnet"])  # backbone type

    try:

        process = subprocess.Popen(
            [
                "python", "objective.py",
                "--res", str(res),
                "--augmentation_index", str(augmentation_index),
                "--batch_size", str(batch_size),
                "--repeat_times", str(repeat_times),
                "--resnet_depth", str(resnet_depth),
                "--backbone_type", str(backbone_type),
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        for line in iter(process.stdout.readline, ''):
            print(line, end='')

        process.stdout.close()
        process.wait()

    except Exception as e:
        print(f"Error: {e}")
        print(
            "Params with error: res - {res}, augmentation_index - {augmentation_index}, "
            "batch_size - {batch_size}, repeat_times - {repeat_times}"
        )
        return np.float("inf")

    return 0

    # Return the metric to be maximized oDeviceRequestdockerarams)
    print("Best value:", study.best_value)


def main():
    study = optuna.create_study(
        direction="maximize",
        study_name="mmpose_optimization",
        storage="sqlite:///mmpose_optimization.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=100)


if __name__ == "__main__":
    main()
