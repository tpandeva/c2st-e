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

from experiments.mri.fastmri_plus import get_pathology_info
from experiments.mri.sampling_dataset import (
    FilteredSlices,
    SampledSlices,
    populate_slice_filter_with_labels,
    PathologyLabelTransform,
)
from models.sampling_mri_model import SamplingModelModule


"""
# Base experiment (100 samples per size)

conda activate c2st
python trainMRI.py --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
     --dataset_sizes 200 400 1000 2000 3000 4000 5000 --settings 1a 1b 2


# Meta-analysis experiment (3 partitions, Type-II error). 
# Do loop so we have multiple meta-analysis experiments to analyse for Type-II error (e.g. 100).
# Set specific save dir for ease of combining multiple meta-analysis experiments.
# Set `seed` to None, because otherwise we use the same data split every time we rerun the script.

conda activate c2st
for i in {1..100}
do 
    python trainMRI.py --num_dataset_samples 0 --num_partitions 3 --num_epochs 30 --do_early_stopping True \
     --dataset_sizes 200 400 1000 --settings 2 --seed None \
     --save_dir /home/timsey/Projects/c2st-e/results/mri/meta_analysis_sept26
done

"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def str2none_int(v):
    if v is None:
        return v
    if v.lower() == "none":
        return None
    else:
        return int(v)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    # General params
    parser.add_argument(
        "--seed",
        default=0,
        type=str2none_int,
        help="Seed for randomness outside of dataset creation. Set None for no seed.",
    )
    parser.add_argument(
        "--save_dir",
        default="/home/timsey/Projects/c2st-e/results/mri/",
        type=Path,
        help="Dir for saving results.",
    )

    # Data args
    parser.add_argument(
        "--data_dir",
        default="/home/timsey/HDD/data/fastMRI/singlecoil/singlecoil_all",
        type=Path,
        help="Path to fastMRI singlecoil knee data .h5 file dir.",
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
    parser.add_argument(
        "--settings",
        default=["1a", "1b", "2"],
        type=str,
        nargs="+",
        help="Experiment settings to use (1a, 1b are Type-I error settings, 2 is Type-II error setting).",
    )
    parser.add_argument(
        "--dataset_sizes",
        default=[5000],
        type=int,
        nargs="+",
        help=(
            "Dataset sizes to train on. Stratified splits will be made. E.g. if set to 1000. Will sample train and "
            "test set of size 500. Compute 2 type-I experiments with size 500, and one type-II experiment with size "
            "500, randomly sampled from the two classes."
        ),
    )
    parser.add_argument(
        "--num_dataset_samples",
        default=1,
        type=int,
        help="Number of datasets to sample and train on for each dataset size (used for computing errors).",
    )
    parser.add_argument(
        "--num_partitions",
        default=1,
        type=int,
        help=(
            "Number of partitions to use (0 for not partitioning). Cannot be used in combination with "
            "num_dataset_samples (set either to 0)."
        ),
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
        help="lr decay factor (exponential decay): 1.0 corresponds to no lr decay.",
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


def c2st_e_prob1(y, prob1):
    # H0
    # p(y|x) of MLE under H0: p(y|x) = p(y), is just the empirical frequency of y in the test data.
    emp_freq_class0 = 1 - (y[y == 1]).sum() / y.shape[0]
    emp_freq_class1 = (y[y == 1]).sum() / y.shape[0]

    # prob1 is probability of class 1 given by model
    pred_prob_class0 = 1 - prob1
    pred_prob_class1 = prob1

    log_eval = torch.sum(
        y * torch.log(pred_prob_class1 / emp_freq_class1) + (1 - y) * torch.log(pred_prob_class0 / emp_freq_class0)
    ).double()
    e_val = torch.exp(log_eval).item()
    return e_val


def c2st_prob1(y, prob1):
    # H0: accuracy=0.5 vs H1: accuracy>0.5
    y_hat = (prob1 > 0.5).long()
    matches = y == y_hat
    accuracy = torch.sum(matches) / y.shape[0]
    n_te = y.shape[0]
    stat = 2 * np.sqrt(n_te) * torch.abs(accuracy - 0.5)
    pval = 1 - Normal(0, 1).cdf(stat).item()
    # p-value OR list of p-values if num_batches is not None
    return pval


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    if args.num_dataset_samples < 0 or args.num_partitions < 0:
        raise ValueError("`num_dataset_samples` and `num_partitions` must be >= 0.")
    if args.num_partitions > 0:
        if args.num_dataset_samples > 0:
            raise ValueError("Set either `num_partitions` or num_dataset_samples` to 0.")
        else:
            num_samples = args.num_partitions  # set samples to number of partitions
            num_partitions = args.num_partitions
    else:
        if args.num_dataset_samples == 0:
            raise ValueError("Set either `num_partitions` or num_dataset_samples` to int > 0.")
        else:
            num_samples = args.num_dataset_samples
            num_partitions = 0

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
        args.dataset_sizes = [100]  # Will ignored most of data in dataset creation already
        args.num_partitions = 0
        args.seed = 0
        num_samples = 1

    input_shape = (args.crop_size, args.crop_size)

    # Seeds
    if args.seed is not None:
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
    pathology_df = pd.read_csv(args.pathology_path, index_col=None, header=0)
    check_df = pd.read_csv(
        args.checked_path,
        names=["file"],
        index_col=None,
        header=None,
        skipfooter=1,
        engine="python",
    )

    not_checked, no_pathologies, any_pathologies, all_pathologies = get_pathology_info(
        [args.data_dir], pathology_df, check_df
    )
    print(len(not_checked), len(no_pathologies), len(any_pathologies))
    print(len(all_pathologies), all_pathologies)

    if len(not_checked) == 0:
        # True if clean, False if not clean
        clean_volumes = {**no_pathologies, **any_pathologies}
    else:
        clean_volumes = None
        raise NotImplementedError("Haven't thought of what to do if there are unchecked volumes yet.")

    slice_filter = partial(partial(populate_slice_filter_with_labels, clean_volumes), all_pathologies)

    # ----------------------------
    # ------ fastMRI data ------
    # ----------------------------

    # Create initial big dataset
    # All filtered data: {False: 3618, True: 8311}, can sample up to 2x 3618 = 7236.
    max_data = 7236
    pathology_transform = PathologyLabelTransform(crop_size=args.crop_size)
    # Just filter out stuff we really don't want to use. Then sample from this dataset.
    full_dataset = FilteredSlices(
        root=args.data_dir,
        raw_sample_filter=slice_filter,  # For populating slices with pathologies.
        pathology_df=pathology_df,  # For volume metadata and for populating slices with pathologies.
        use_center_slices_only=True,
        quick_test=args.quick_test,
    )

    results_dict = {}
    # Best if dataset sizes are divisible by 40.
    # Half these points are used for each test (2x Type-I, 1x Type-II)
    # Half of that is train-test split.
    # 20% of train data is made validation instead.
    # And all this is split over 2 classes.
    # So, 2 * 2 * 2 * 5 (20%) = 40 as divisor.
    for dataset_size in args.dataset_sizes:
        print(f"\n ----- Dataset size: {dataset_size} ----- ")
        size_results_dict = {dataset_ind: {} for dataset_ind in range(num_samples)}
        if isinstance(args.num_partitions, int):
            if dataset_size * args.num_partitions > max_data:  # Max dataset size
                raise RuntimeError(
                    f"Cannot partition data into {args.num_partition} bits of size {dataset_size}. Only {max_data} "
                    f"points available."
                )
        # Sample datasets of this size.
        slice_splits = None
        for dataset_ind in range(num_samples):
            print(f"\n  ----- Dataset index: {dataset_ind} ----- ")
            # Do this experiment a bunch of times so we can get an error estimate.
            # NOTE: Can also do partition split to treat as different experiments!
            if num_partitions > 0 and slice_splits is not None:
                # This means we are doing partitions, and are not on the first partition index anymore.
                # slice_splits will then contain data sampled from the next partition.
                slice_splits.next_partition()
            else:
                slice_splits = SampledSlices(
                    base_dataset=full_dataset,
                    dataset_size=dataset_size,
                    transform=pathology_transform,
                    num_partitions=num_partitions,  # Whether to do partition split (0, >0), and how many.
                )
            # Loop over Type-Ia, Ib, and II experiment settings.
            for setting, (train, val, test) in slice_splits.datasets_dict.items():
                if setting not in args.settings:
                    print(f"Skipping setting: {setting}.")
                    continue
                print(f"\n   ----- Setting: {setting} ----- ")
                train_loader = torch.utils.data.DataLoader(
                    dataset=train,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    worker_init_fn=None,
                    shuffle=True,
                )
                val_loader = torch.utils.data.DataLoader(
                    dataset=val,
                    batch_size=256,
                    num_workers=args.num_workers,
                    worker_init_fn=None,
                    shuffle=False,
                )
                test_loader = torch.utils.data.DataLoader(
                    dataset=test,
                    batch_size=256,
                    num_workers=args.num_workers,
                    worker_init_fn=None,
                    shuffle=False,
                )

                # --------------------------------------
                # ------ Model, training, testing ------
                # --------------------------------------

                module = SamplingModelModule(
                    args.in_chans,
                    args.chans,
                    args.num_pool_layers,
                    args.drop_prob,
                    input_shape,
                    args.lr,
                    args.total_lr_gamma,
                    args.num_epochs,
                    args.do_early_stopping,
                )

                (
                    train_losses,
                    val_losses,
                    val_accs,
                    extra_output,
                    total_time,
                ) = module.train(train_loader, val_loader, print_every=1, eval_every=1)
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

                e_val = c2st_e_prob1(targets, test_prob1)
                print(f" 1 / E-value: {1 / e_val:.4f} (actual: {1 / e_val})")
                p_val_c2st = c2st_prob1(targets, test_prob1)
                print(f"     p-value: {p_val_c2st:.4f} (actual: {p_val_c2st})")

                size_results_dict[dataset_ind][setting] = {
                    "e_val": e_val,
                    "p_val": p_val_c2st,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }

        results_dict[dataset_size] = size_results_dict
        # Saving results
        with open(save_dir / f"results_size{dataset_size}.json", "w") as f:
            json.dump({str(key): val for key, val in size_results_dict.items()}, f, indent=4)

    # Saving results
    with open(save_dir / "results.json", "w") as f:
        json.dump({str(key): val for key, val in results_dict.items()}, f, indent=4)
