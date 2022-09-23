import random
from datetime import datetime
import numpy as np
import torch
import json
import pandas as pd
import torch.fft
from pathlib import Path
from functools import partial
from torch.distributions import Normal
import argparse
import matplotlib.pyplot as plt

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


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
    parser.add_argument(
        "--save_dir",
        default="/home/timsey/Projects/c2st-e/results/mri/",
        type=Path,
        help="Dir for saving results.",
    )
    parser.add_argument(
        "--type1",
        default=True,
        type=str2bool,
        help="Do type1 error test (same distribution) if True, or type2 (different distribution) if False.",
    )
    parser.add_argument(
        "--num_test_batches",
        default=1,
        type=int,
        help="Number of test batches to use. Can compute Type-I/II errors when using multiple.",
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
    # Sample rate 1 corresponds to binary stratified: Train 4436, Val 1632, Test 1168 slices.
    parser.add_argument(
        "--sample_rates",
        default=[1],
        type=float,
        nargs="+",
        help="Sample rates to use.",
    )
    parser.add_argument(
        "--num_exp",
        default=1,
        type=int,
        help="Number of datasets to combine into one experiment.",
    )
    parser.add_argument("--crop_size", default=320, type=int, help="MRI image crop size (square).")
    parser.add_argument("--batch_size", default=64, type=int, help="Train batch size.")
    parser.add_argument("--num_workers", default=20, type=int, help="Number of dataloader workers.")
    # Unet params
    parser.add_argument("--in_chans", default=1, type=int, help="Unet encoder in channels.")
    parser.add_argument("--chans", default=16, type=int, help="Unet encoder first-layer channels.")
    parser.add_argument(
        "--num_pool_layers",
        default=4,
        type=int,
        help="Unet encoder number of pool layers.",
    )
    parser.add_argument("--drop_prob", default=0.0, type=float, help="Unet encoder dropout probability.")
    # Learning params
    parser.add_argument("--num_epochs", default=5, type=int, help="Number of training epochs.")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate.")
    parser.add_argument(
        "--total_lr_gamma",
        default=1.0,
        type=float,
        help="lr decay factor (exponential decay).",
    )
    parser.add_argument(
        "--do_early_stopping",
        default=True,
        type=str2bool,
        help="Whether to do early stopping on validation data.",
    )
    parser.add_argument(
        "--quick_test",
        default=False,
        type=str2bool,
        help="Do quick run-test (ignore values of these runs).",
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
            y * torch.log(pred_prob_class1 / emp_freq_class1) + (1 - y) * torch.log(pred_prob_class0 / emp_freq_class0)
        ).double()
        e_val = torch.exp(log_eval).item()

    else:
        # e_val = 0
        ratios = y * torch.log(pred_prob_class1 / emp_freq_class1) + (1 - y) * torch.log(
            pred_prob_class0 / emp_freq_class0
        )
        ind = torch.randperm(ratios.shape[0])
        ratios = ratios[ind]
        ratio_batches = [ratios[i::num_batches] for i in range(num_batches)]
        e_val = [torch.exp(torch.sum(ratio_batches[i]).double()).item() for i in range(num_batches)]
        # for i in range(num_batches):
        #     e_val = e_val + torch.exp(torch.sum(ratio_batches[i]))
        # e_val = e_val / num_batches

    # E-value OR list of E-values if num_batches is not None
    return e_val


def c2st_prob1(y, prob1, num_batches=None):
    # H0: accuracy=0.5 vs H1: accuracy>0.5
    y_hat = (prob1 > 0.5).long()
    matches = y == y_hat
    if num_batches is None:
        accuracy = torch.sum(matches) / y.shape[0]
        n_te = y.shape[0]
        stat = 2 * np.sqrt(n_te) * (accuracy - 0.5)
        pval = 1 - Normal(0, 1).cdf(stat).item()
    else:
        pval = []
        ind = torch.randperm(y_hat.shape[0])
        matches = matches[ind]
        matches_batches = [matches[i::num_batches] for i in range(num_batches)]
        for batch in matches_batches:
            batch_acc = torch.sum(matches) / batch.shape[0]
            n_te = batch.shape[0]
            stat = 2 * np.sqrt(n_te) * (batch_acc - 0.5)
            pval.append(1 - Normal(0, 1).cdf(stat).item())

    # p-value OR list of p-values if num_batches is not None
    return pval


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    if len(args.sample_rates) > 1 and args.num_exp > 1:
        raise ValueError(
            "Cannot run multiple sample rates AND partition dataset. "
            "Make `sample_rates` contain only one element or set `num_exp` to 1."
        )

    # Save args
    date_string = f"{datetime.now():%Y-%m-%d}"
    time_string = f"{datetime.now():%H:%M:%S}"
    save_dir = args.save_dir / date_string / time_string
    save_dir.mkdir(parents=True, exist_ok=False)
    args_dict = {key: str(value) for key, value in vars(args).items()}
    with open(save_dir / "args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    if args.quick_test:
        args.num_epochs = 1
        args.sample_rates = [1]  # Will ignored most of data in dataset creation already
        args.data_seed = 0
        args.seed = 0

    input_shape = (args.crop_size, args.crop_size)

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        raise NotImplementedError("Haven't thought of what to do if there are unchecked volumes yet.")

    slice_filter = partial(partial(populate_slice_filter, clean_volumes), all_pathologies)

    # ----------------------------
    # ------ fastMRI data ------
    # ----------------------------

    pathology_transform = PathologyTransform(crop_size=args.crop_size)

    # TODO: combine multiple seeds? Or just do this post hoc. This is for confidence intervals.
    results_dict = {}
    for sample_rate in args.sample_rates:
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
            quick_test=args.quick_test,
            type1=args.type1,
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
            quick_test=args.quick_test,
            type1=args.type1,
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
            sample_rate=1,  # always use all test data: batches for type-I/II error.
            num_datasets=args.num_exp,
            quick_test=args.quick_test,
            type1=args.type1,
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
                args.do_early_stopping,
                args.type1,
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

            if args.num_test_batches == 1:
                e_val = c2st_e_prob1(targets, test_prob1, num_batches=None)
                print(f" 1 / E-value: {1 / e_val:.4f} (actual: {1 / e_val})")
                p_val_c2st = c2st_prob1(targets, test_prob1, num_batches=None)
                print(f"     p-value: {p_val_c2st:.4f} (actual: {p_val_c2st})")
            else:  # multiple values
                e_val = c2st_e_prob1(targets, test_prob1, num_batches=args.num_test_batches)
                e_val_recip = [1 / e_val for e_val in e_val]
                print(f" 1 / E-values: {e_val_recip}")
                p_val_c2st = c2st_prob1(targets, test_prob1, num_batches=args.num_test_batches)
                print(f"     p-values: {p_val_c2st}")

            result_label = (dataset_ind, sample_rate)
            # float OR list of floats
            results_dict[result_label] = {"e_val": e_val, "p_val": p_val_c2st}

            if dataset_ind < args.num_exp - 1:
                train_dataset.next_dataset()
                val_dataset.next_dataset()
                test_dataset.next_dataset()

    # Saving results
    with open(save_dir / "results.json", "w") as f:
        json.dump({str(key): val for key, val in results_dict.items()}, f, indent=4)

    if len(args.sample_rates) > 1 or args.num_exp > 1:  # Multiple experiments
        if len(args.sample_rates) > 1:  # Multiple sample rates
            save_name = save_dir / "per_sample_rate.png"
            # Sample rate 1 corresponds to binary stratified: Train 4436, Val 1632, Test 1168 slices.
            xlabel = "data fraction (of 4436 train, 1632 val, 1168 test)"
            keys = [key for key in sorted(results_dict.keys(), key=lambda x: float(x[1]))]  # Sort by sample rate
            x_vals = [float(key[1]) for key in keys]  # sample rate
            e_values_recip = np.array([1 / np.array(results_dict[key]["e_val"]) for key in keys])  # 1 / e-value
            p_values = np.array([results_dict[key]["p_val"] for key in keys])

            if args.num_test_batches > 1:  # multiple e-/p-values (one per test batch), compute type-I/II errors
                # Fraction of incorrect decisions
                threshold = 0.05
                # TODO: check computation correctness
                if args.type1:
                    title = "Type-I (1)"
                    # Rejection of null is incorrect. Error = mean(#rejections).
                    e_errors = np.array([np.mean(e_ps < threshold) for e_ps in e_values_recip])
                    p_errors = np.array([np.mean(p_ps < threshold) for p_ps in p_values])
                else:
                    title = "Type-II (2)"
                    # Rejection of null is correct. Error = 1 - mean(#rejections).
                    e_errors = np.array([1 - np.mean(e_ps < threshold) for e_ps in e_values_recip])
                    p_errors = np.array([1 - np.mean(p_ps < threshold) for p_ps in p_values])

                plt.figure(figsize=(8, 5))
                plt.plot(x_vals, e_errors, c="orange", marker="o", label="e-value")
                plt.plot(x_vals, p_errors, c="b", marker="o", label="p-value")
                plt.title(f"{title} error as a function of dataset size", fontsize=18)
                plt.ylabel("error value", fontsize=15)
                plt.xlabel(f"{xlabel}", fontsize=15)
                plt.xlim(0, x_vals[-1] * 1.1)
                plt.legend()
                plt.tight_layout()
                plt.savefig(save_name)
            else:  # single e-/p-value, plot as function of dataset size. No stat errors reported.
                plt.figure(figsize=(8, 5))
                plt.plot(x_vals, e_values_recip, c="orange", marker="o", label="1 / e-value")
                plt.plot(x_vals, p_values, c="b", marker="o", label="p-value")
                plt.title("Statistics as function of dataset size", fontsize=18)
                plt.ylabel("statistic value", fontsize=15)
                plt.xlabel(f"{xlabel}", fontsize=15)
                plt.xlim(0, x_vals[-1] * 1.1)
                plt.legend()
                plt.tight_layout()
                plt.savefig(save_name)

        else:  # Multiple partitions: meta-analysis bit
            print("...WARNING: code not complete here...")
            # TODO: Add error plotting
            title = "Statistics per partition"
            save_name = save_dir / "per_partition.png"
            xlabel = "partition"
            # TODO: plots per partition now, but want to plot combined
            keys = [key for key in sorted(results_dict.keys(), key=lambda x: int(x[0]))]  # Sort by dataset_ind
            x_vals = np.array([int(key[0]) for key in keys])  # partition number
            e_values_recip = [1 / results_dict[key]["e_val"] for key in keys]  # 1 / e-value
            p_values = [results_dict[key]["p_val"] for key in keys]

            plt.figure(figsize=(8, 5))
            plt.bar(
                x_vals - 0.1,
                e_values_recip,
                width=0.2,
                color="orange",
                align="center",
                label="1 / e-value",
            )
            plt.bar(
                x_vals + 0.1,
                p_values,
                width=0.2,
                color="b",
                align="center",
                label="p-value",
            )
            plt.title(f"{title}", fontsize=18)
            plt.ylabel("statistic value", fontsize=15)
            plt.xlabel(f"{xlabel}", fontsize=15)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_name)

    else:  # Single experiment
        save_name = save_dir / "learning_curves.png"
        val_loss_x = [key for key in sorted(val_losses.keys(), key=lambda x: int(x))]
        val_loss_y = [val_losses[key] for key in val_loss_x]
        val_acc_y = [val_accs[key] for key in val_loss_x]

        key = list(results_dict.keys())[0]
        # float OR array of floats (one for every test batch)
        e_value_recip = 1 / np.array(results_dict[key]["e_val"])
        p_value = np.array(results_dict[key]["p_val"])

        # Learning curves
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses, c="b", marker="o", label="train")
        plt.plot(val_loss_x, val_loss_y, c="orange", marker="o", label="val")
        plt.title("Learning curves", fontsize=18)
        plt.ylabel("loss", fontsize=15)
        plt.xlabel("epoch", fontsize=15)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(val_loss_x, val_acc_y, c="orange", marker="o", label="val")
        plt.axhline(
            test_acc,
            xmin=0.01,
            xmax=0.99,
            c="green",
            ls="--",
            label="test (last epoch)",
        )
        plt.title(f"Accuracies", fontsize=18)
        plt.ylabel("accuracy", fontsize=15)
        plt.xlabel("epoch", fontsize=15)
        plt.legend()

        plt.subplot(1, 3, 3)
        if isinstance(e_value_recip, np.float64):  # Single test batch
            x_vals = np.array([0])
            e_error_bit = ""
            p_error_bit = ""
        else:  # array: multiple test batches
            # Fraction of incorrect decisions
            threshold = 0.05
            # TODO: check computation correctness
            if args.type1:
                title = "Type-I (1)"
                # Rejection of null is incorrect. Error = mean(#rejections).
                e_error = np.mean(e_value_recip < threshold)
                p_error = np.mean(p_value < threshold)
            else:
                title = "Type-II (2)"
                # Rejection of null is correct. Error = 1 - mean(#rejections).
                e_error = 1 - np.mean(e_value_recip < threshold)
                p_error = 1 - np.mean(p_value < threshold)

            e_error_bit = f", {title} error: {e_error:.2f}"
            p_error_bit = f", {title} error: {p_error:.2f}"
            x_vals = np.array(list(range(len(e_value_recip))))

        plt.bar(
            x_vals - 0.1,
            e_value_recip,
            width=0.2,
            color="orange",
            align="center",
            label=f"1 / e-value{e_error_bit}",
        )
        plt.bar(
            x_vals + 0.1,
            p_value,
            width=0.2,
            color="b",
            align="center",
            label=f"p-value{p_error_bit}",
        )
        plt.title(f"Statistics per test batch.", fontsize=18)
        plt.ylabel("statistic value", fontsize=15)
        plt.xlabel("test batch index", fontsize=15)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_name)


# TODO:
#  - Compute Type-2 errors using test-set batches (sample_rate = 1 for test set always then).
#  - Compute Type-1 errors using same dist training (no-pathologies easiest probably?)
#   - Current implementation is messy: maybe do labels in dataset already, not in mri_model?
#  - Do confidence intervals of errors by training models with multiple seeds.
#  - Meta analysis experiments: multiple 'datasets' (train, val, test) splits, that all compute p- and e-values.
#     Here we don't need to compute type-I/type-II errors maybe? Otherwise we also have to split test again.

# TODO:
#  - Setup as single big dataset: split in train-test (100 times or so), split train in train-val, all stratified.
#  - Say big data contains 6000 points. We make type-I and type-II error splits.
#  - Type-I: get train-test split with 1000 of 0, 1000 of 1 in train, and same in test. Train model on 0, model on 1,
#     act like they are different classes within, of course.
#     Then compute e-value for this setting. Do this 100 times, compute 2 type-I errors for this.
#  - Type-II: get train-test split with 500 of 0, 500 of 1. Train model here (single e-value). Then compute Type-II
#     error from the 100 experiments and match with type-I.
#  - Of course, errors depend on alpha, so save e-value for each setting (dataset size: 100x type-Ia, type-Ib, type-II).
#  - Repeat for different data sizes.
#  - Then meta-analysis experiment?
