import random
import numpy as np
import torch
import pandas as pd
import torch.fft
from pathlib import Path
from functools import partial
from torch.distributions import Normal
import argparse

from experiments.mri.fastmri_plus import (
    PathologiesSliceDataset,
    PathologyTransform,
    get_pathology_info,
    populate_slice_filter,
)
from models.mri_model import ModelModule


# fastMRI, NeurIPS2020 splits
fastmri_data_folder = Path("/home/timsey/HDD/data/fastMRI/singlecoil")
train = fastmri_data_folder / "singlecoil_train"
val = fastmri_data_folder / "singlecoil_val"
test = fastmri_data_folder / "singlecoil_test"

# fastMRI+
plus_data_folder = Path("/home/timsey/Projects/fastmri-plus/Annotations/")
pathology_path = plus_data_folder / "knee.csv"
checked_path = plus_data_folder / "knee_file_list.csv"

# -------------------------
# ------ Hyperparams ------
# -------------------------


def create_arg_parser():
    parser = argparse.ArgumentParser()

    # General params
    parser.add_argument(
        "--data_seed",
        default=0,
        type=int,
        help="Seed for randomness in dataset creation.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed for randomness outside of dataset creation.",
    )
    # Data args
    parser.add_argument(
        "--train_dir",
        default="/home/timsey/HDD/data/fastMRI/singlecoil/singlecoil_train",
        type=Path,
        help="Path to fastMRI singlecoil knee train data .h5 file dir.",
    )
    parser.add_argument(
        "--val_dir",
        default="/home/timsey/HDD/data/fastMRI/singlecoil/singlecoil_val",
        type=Path,
        help="Path to fastMRI singlecoil knee val data .h5 file dir.",
    )
    parser.add_argument(
        "--test_dir",
        default="/home/timsey/HDD/data/fastMRI/singlecoil/singlecoil_test",
        type=Path,
        help="Path to fastMRI singlecoil knee test data .h5 file dir.",
    )
    parser.add_argument(
        "--pathology_path",
        default="/home/timsey/Projects/fastmri-plus/Annotations/knee.csv",
        type=Path,
        help="Path to fastMRI+ knee annotations csv.",
    )
    parser.add_argument(
        "--checked_path",
        default="/home/timsey/Projects/fastmri-plus/Annotations/knee_file_list.csv",
        type=Path,
        help="Path to fastMRI+ `knee_file_list.csv`.",
    )
    # Experiment params
    # sample_rates = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    parser.add_argument(
        "--sample_rates", default=[1], type=int, nargs="+", help="Sample rates to use."
    )
    parser.add_argument(
        "--num_exp",
        default=1,
        type=int,
        help="Number of datasets to combine into one experiment.",
    )
    parser.add_argument(
        "--crop_size", default=320, type=int, help="MRI image crop size (square)."
    )
    parser.add_argument("--batch_size", default=64, type=int, help="Train batch size.")
    parser.add_argument(
        "--num_workers", default=20, type=int, help="Number of dataloader workers."
    )
    # Unet params
    parser.add_argument(
        "--in_chans", default=1, type=int, help="Unet encoder in channels."
    )
    parser.add_argument(
        "--chans", default=16, type=int, help="Unet encoder first-layer channels."
    )
    parser.add_argument(
        "--num_pool_layers",
        default=4,
        type=int,
        help="Unet encoder number of pool layers.",
    )
    parser.add_argument(
        "--drop_prob", default=0.0, type=float, help="Unet encoder dropout probability."
    )
    # Learning params
    parser.add_argument(
        "--num_epochs", default=5, type=int, help="Number of training epochs."
    )
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate.")
    parser.add_argument(
        "--total_lr_gamma",
        default=1.0,
        type=float,
        help="lr decay factor (exponential decay).",
    )
    return parser


def c2st_e_prob1(y, prob1, num_batches=None):
    # H0
    # p(y|x) of MLE under H0: p(y|x) = p(y), is just the empirical frequency of y in the test data.
    emp_freq_class0 = 1 - (y[y == 1]).sum() / y.shape[0]
    emp_freq_class1 = (y[y == 1]).sum() / y.shape[0]

    # prob1 is probability of class 1 given by model
    pred_prob_class0 = 1 - prob1
    pred_prob_class1 = prob1

    if num_batches is None:
        log_eval = torch.sum(
            y * torch.log(pred_prob_class1 / emp_freq_class1)
            + (1 - y) * torch.log(pred_prob_class0 / emp_freq_class0)
        ).double()
        e_val = torch.exp(log_eval)

    else:
        e_val = 0
        ratios = y * torch.log(pred_prob_class1 / emp_freq_class1) + (
            1 - y
        ) * torch.log(pred_prob_class0 / emp_freq_class0)
        ind = torch.randperm(ratios.shape[0])
        ratios = ratios[ind]
        ratio_batches = [ratios[i::num_batches] for i in range(num_batches)]
        for i in range(num_batches):
            e_val = e_val + torch.exp(torch.sum(ratio_batches[i]))
        e_val = e_val / num_batches

    # E-value
    return e_val


def c2st_prob1(y, prob1):
    # H0: accuracy=0.5 vs H1: accuracy>0.5
    y_hat = (prob1 > 0.5).long()
    accuracy = torch.sum(y == y_hat) / y.shape[0]
    n_te = y.shape[0]
    stat = 2 * np.sqrt(n_te) * (accuracy - 0.5)
    pval = 1 - Normal(0, 1).cdf(stat)
    return pval


if __name__ == "__main__":
    args = create_arg_parser().parse_args()

    input_shape = (args.crop_size, args.crop_size)

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----------------------------
    # ------ Pathology data ------
    # ----------------------------

    # Skip final row because it's "Knee data" for some reason
    pathology_df = pd.read_csv(pathology_path, index_col=None, header=0)
    check_df = pd.read_csv(
        checked_path,
        names=["file"],
        index_col=None,
        header=None,
        skipfooter=1,
        engine="python",
    )

    folders_to_check = [train, val, test]
    not_checked, no_pathologies, any_pathologies, all_pathologies = get_pathology_info(
        folders_to_check, pathology_df, check_df
    )
    print(len(not_checked), len(no_pathologies), len(any_pathologies))
    print(len(all_pathologies), all_pathologies)

    if len(not_checked) == 0:
        clean_volumes = {**no_pathologies, **any_pathologies}
    else:
        clean_volumes = None
        raise NotImplementedError(
            "Haven't thought of what to do if there are unchecked volumes yet."
        )

    slice_filter = partial(
        partial(populate_slice_filter, clean_volumes), all_pathologies
    )

    # ----------------------------
    # ------ fastMRI data ------
    # ----------------------------

    pathology_transform = PathologyTransform(crop_size=args.crop_size)

    for sample_rate in sample_rates:
        print(f"\n ----- Sample rate: {sample_rate} ----- ")
        train_dataset = PathologiesSliceDataset(
            root=train,
            challenge="singlecoil",  # Doesn't do anything right now, because pathologies labeled using RSS.
            transform=pathology_transform,
            raw_sample_filter=slice_filter,  # For populating slices with pathologies.
            pathology_df=pathology_df,  # For volume metadata and for populating slices with pathologies.
            clean_volumes=clean_volumes,  # For equalising clean VS. pathology volumes.
            seed=args.data_seed,
            use_center_slices_only=True,
            sample_rate=sample_rate,
            num_datasets=args.num_exp,
        )
        val_dataset = PathologiesSliceDataset(
            root=val,
            challenge="singlecoil",  # Doesn't do anything right now, because pathologies labeled using RSS.
            transform=pathology_transform,
            raw_sample_filter=slice_filter,  # For populating slices with pathologies.
            pathology_df=pathology_df,  # For volume metadata and for populating slices with pathologies.
            clean_volumes=clean_volumes,  # For equalising clean VS. pathology volumes.
            seed=args.data_seed,
            use_center_slices_only=True,
            sample_rate=sample_rate,
            num_datasets=args.num_exp,
        )
        test_dataset = PathologiesSliceDataset(
            root=test,
            challenge="singlecoil",  # Doesn't do anything right now, because pathologies labeled using RSS.
            transform=pathology_transform,
            raw_sample_filter=slice_filter,  # For populating slices with pathologies.
            pathology_df=pathology_df,  # For volume metadata and for populating slices with pathologies.
            clean_volumes=clean_volumes,  # For equalising clean VS. pathology volumes.
            seed=args.data_seed,
            use_center_slices_only=True,
            sample_rate=sample_rate,
            num_datasets=args.num_exp,
        )

        # One experiment per dataset
        for dataset_ind in range(args.num_exp):
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                worker_init_fn=None,
                shuffle=True,
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=256,
                num_workers=args.num_workers,
                worker_init_fn=None,
                shuffle=False,
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=256,
                num_workers=args.num_workers,
                worker_init_fn=None,
                shuffle=False,
            )

            # --------------------------------------
            # ------ Model, training, testing ------
            # --------------------------------------

            module = ModelModule(
                args.in_chans,
                args.chans,
                args.num_pool_layers,
                args.drop_prob,
                input_shape,
                args.lr,
                args.total_lr_gamma,
                args.num_epochs,
            )

            train_losses, val_losses, val_accs, extra_output, total_time = module.train(
                train_loader, val_loader, print_every=1, eval_every=1
            )
            print(f"Total time: {total_time:.2f}s")

            # val_loss_x = [key for key in sorted(val_losses.keys(), key=lambda x: int(x))]
            # val_loss_y = [val_losses[key] for key in val_loss_x]
            # val_acc_y = [val_accs[key] for key in val_loss_x]
            test_loss, test_acc, test_extra_output = module.test(test_loader)

            # --------------------------
            # ------ Sample tests ------
            # --------------------------

            # Targets are 1-dimensional
            targets = test_extra_output["targets"]
            # logits are 1-dimensional
            test_logit1 = test_extra_output["logits"]
            # test_prob1 is probability of class 1 given by model
            test_prob1 = torch.sigmoid(test_logit1)

            e_val = c2st_e_prob1(targets, test_prob1).item()
            print(f" 1 / E-value: {1 / e_val:.4f} (actual: {1 / e_val})")
            p_val_c2st = c2st_prob1(targets, test_prob1).item()
            print(f"     p-value: {p_val_c2st:.4f} (actual: {p_val_c2st})")

            if dataset_ind < args.num_exp - 1:
                train_dataset.next_dataset()
                val_dataset.next_dataset()
                test_dataset.next_dataset()
