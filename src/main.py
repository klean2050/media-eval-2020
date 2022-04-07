"""
This repo is part of the USC SAIL submission to
the 2020 MediaEval Emotions and Themes in Music task. 

Base implementations for the three models provided here
are forked from the sota-music-tagging-models repository
(https://github.com/minzwon/sota-music-tagging-models) by Won et al.

See their model comparison paper at https://arxiv.org/abs/2006.00751
"""

import os, argparse, numpy as np

from train import Solver
from data_loader import get_audio_loader


def main(config):
    # path to save checkpoint
    os.makedirs(config.model_save_path, exist_ok=True)
    # Set audio segment length (in samples)
    config.input_length = 4.6 * 16000  # 4.6 seconds * 16000 Hz

    # get data loder
    train_loader, class_weights = get_audio_loader(
        config.data_path,
        config.batch_size,
        config.splits_path,
        config.sampling_type,
        split="TRAIN",
        input_length=config.input_length,
        num_workers=config.num_workers,
    )
    solver = Solver(train_loader, class_weights, config)
    solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Parameters for the training procedure
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_tensorboard", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_step", type=int, default=50)

    # Loss function to use
    parser.add_argument(
        "--loss_function",
        type=str,
        default="bce",
        choices=["bce", "focal_loss", "cb_focal_loss", "db_focal_loss"],
    )
    # Whether to augment with mixup
    parser.add_argument("--use_mixup", type=int, default=0)
    parser.add_argument(
        "--sampling_type",
        type=str,
        default="standard",
        choices=["standard", "class_aware"],
    )

    # "Model load path" can point to a pretrained model
    parser.add_argument("--model_load_path", type=str, default=".")
    # "Model save path" is the output directory for best model and curves
    parser.add_argument("--model_save_path", type=str, default=".")
    # "Data path" should point to a directory structure of .npy files. See data_loader.py
    parser.add_argument("--data_path", type=str, default="./data")
    # "Splits path" should point to a directory with files that define the training, validation, and test splits
    parser.add_argument("--splits_path", type=str, default="./splits/")

    config = parser.parse_args()

    os.makedirs(config.model_save_path, exist_ok=True)
    np.save(
        os.path.join(config.model_save_path, "config_parameters.npy"),
        config,
        allow_pickle=True,
    )

    main(config)
