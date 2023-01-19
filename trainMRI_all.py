import random
import time
from datetime import datetime
import numpy as np
import torch
import json
import pandas as pd
import torch.fft
from pathlib import Path
from functools import partial
from torch.distributions import Normal
from tests import mmd2_permutations
import argparse

from experiments.mri.fastmri_plus import get_pathology_info
from experiments.mri.sampling_dataset import (
    FilteredSlices,
    SampledSlices,
    populate_slice_filter_with_labels,
    PathologyLabelTransform,
)
from models.sampling_mri_model import SamplingModelModule
from models.sampling_mri_model_mmd import SamplingModelModuleMMD


"""
# Base experiment (100 samples per size)
# Save dir not necessary, but useful to find this back later.

conda activate c2st
python trainMRI_all.py --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
     --dataset_sizes 200 400 1000 2000 3000 4000 5000 --settings 1a 1b 2 --seed None \
     --save_dir /home/timsey/Projects/c2st-e/results/mri/base_exp_oct10


# Meta-analysis experiment (3 partitions). 
# We overload the num_dataset_samples loop to do partitions instead: set to 0 to activate partition behaviour.
# Do outer loop so we have multiple meta-analysis experiments to analyse for Type-I/II error (e.g. 100).
# Set specific save dir for ease of combining multiple meta-analysis experiments.
# Set `seed` to None, because otherwise we use the same data split every time we rerun the script.

conda activate c2st
for i in {1..100}
do 
    python trainMRI_all.py --num_dataset_samples 0 --num_partitions 3 --num_epochs 30 --do_early_stopping True \
     --dataset_sizes 200 400 1000 2000 --settings 1a 1b 2 --seed None \
     --save_dir /home/timsey/Projects/c2st-e/results/mri/meta_analysis_oct10
done



# Combined:

conda activate c2st
python trainMRI.py --num_dataset_samples 100 --num_partitions 0 --num_epochs 30 --do_early_stopping True \
     --dataset_sizes 200 400 1000 2000 3000 4000 5000 --settings 1a 1b 2 --seed None \
     --save_dir /home/timsey/Projects/c2st-e/results/mri/base_exp_oct7
     
for i in {1..100}
do 
    python trainMRI.py --num_dataset_samples 0 --num_partitions 3 --num_epochs 30 --do_early_stopping True \
     --dataset_sizes 200 400 1000 2000 --settings 1a 1b 2 --seed None \
     --save_dir /home/timsey/Projects/c2st-e/results/mri/meta_analysis_oct7
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
        help=(
            "Seed, use ONLY for debugging. Set to `None` for actual experiments (influences data sampling "
            "in potentially problematic ways, especially for meta-learning experiments)."
        ),
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
        "--patience",
        default=3,
        type=int,
        help="Patience (in epochs) to use for early stopping.",
    )
    parser.add_argument(
        "--quick_test",
        default=False,
        type=str2bool,
        help="Do quick run-test (ignore values of these runs).",
    )
    parser.add_argument(
        "--test_type",
        default="base",
        type=str,
        choices=["base", "anytime"],
        help="Type of test to run ('base' or 'anytime').",
    )
    parser.add_argument(
        "--num_skip_rounds",
        default=0,
        type=int,
        help="Number of rounds to skip model training on during anytime testing (to start with larger datasets).",
        # NOTE: This is different from only skipping the E-value computation!
    )
    parser.add_argument(
        "--cold_start",
        default=False,
        type=str2bool,
        help=(
            "Whether to retrain model from scratch every round of anytime testing (else continues "
            "from previous round). Not used when `do_online_learning` is True, since online learning implies "
            "starting from a previous trained state.",
        )
    )
    parser.add_argument(
        "--volumes_per_batch",
        default=10,
        type=int,
        help=(
            "Number of volumes to use per batch during anytime testing. Multiple "
            "of 2 (use half `positive` and half `negative` samples.",
        )
    )
    parser.add_argument(
        "--do_online_learning",
        default=False,
        type=str2bool,
        help="Whether to do online learning, else: accumulate batches for training data.",
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
    stat = 2 * np.sqrt(n_te) * (accuracy - 0.5)
    pval = 1 - Normal(0, 1).cdf(stat).item()
    # p-value OR list of p-values if num_batches is not None
    return pval


# def c2st_permutation(y, prob1, N_per):
#     # H0: accuracy=0.5 vs H1: accuracy>0.5
#     y_hat = (prob1 > 0.5).long()
#     matches = y == y_hat
#     accuracy = torch.sum(matches) / y.shape[0]
#     n_te = y.shape[0]
#     STAT_vector = np.zeros(N_per)
#     for r in range(N_per):
#         ind = np.random.choice(n_te, n_te, replace=False)
#         STAT_vector[r] = torch.sum(y[ind] == y_hat) / y.shape[0]
#     S_vector = np.sort(STAT_vector)
#     p_val = np.mean(accuracy.item() < S_vector)
#     return p_val


def test_le(x, y, N_per, alpha=0.05, sigmoid=False):
    N = x.shape[0]
    if sigmoid:
        # f = torch.nn.Softmax()
        f = torch.sigmoid
        x = f(x)
    N1 = y.sum().int()
    # STAT = abs((x[y == 1, 0]).type(torch.FloatTensor).mean() - (x[y == 0, 0]).type(torch.FloatTensor).mean())
    STAT = abs((x[y == 1]).type(torch.FloatTensor).mean() - (x[y == 0]).type(torch.FloatTensor).mean())

    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # STAT_vector[r] = abs(x[ind_X, 0].type(torch.FloatTensor).mean() - x[ind_Y, 0].type(torch.FloatTensor).mean())
        STAT_vector[r] = abs(x[ind_X].type(torch.FloatTensor).mean() - x[ind_Y].type(torch.FloatTensor).mean())
    S_vector = np.sort(STAT_vector)
    p_val = np.mean(STAT.item() < S_vector)
    threshold = S_vector[int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT, p_val


def train_model_and_do_base_tests(args, dataset_ind, setting, input_shape, train, val, test):
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
    print(f"\n Starting C2ST-E/P/L training...")
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
        patience=3,
    )

    (
        train_losses,
        val_losses,
        val_accs,
        extra_output,
        total_time,
    ) = module.train(train_loader, val_loader, print_every=1, eval_every=1)
    print(f" Total time: {total_time:.2f}s")
    test_loss, test_acc, test_extra_output = module.test(test_loader)

    # print(f"\n Starting MMD training...")
    # # MMD classifier training and testing
    # module = SamplingModelModuleMMD(
    #     args.in_chans,
    #     args.chans,
    #     args.num_pool_layers,
    #     args.drop_prob,
    #     input_shape,
    #     args.lr,
    #     args.total_lr_gamma,
    #     args.num_epochs,
    #     save_dir / str(dataset_size) / str(dataset_ind) / str(setting),  # For logit saving
    #     args.do_early_stopping,
    # )
    #
    # (
    #     train_losses_mmd,
    #     val_losses_mmd,
    #     extra_output_mmd,
    #     total_time_mmd,
    # ) = module.train(train_loader, val_loader, print_every=1, eval_every=1)
    # print(f" Total time: {total_time_mmd:.2f}s")
    # test_loss_mmd, Kxyxy, mmd_size, test_extra_output_mmd = module.test(test_loader)

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
    _, _, _, p_val_l = test_le(test_logit1, targets, 100, sigmoid=False)
    print(f"     p-value (L): {p_val_l:.4f} (actual: {p_val_l})")
    _, _, _, p_val_ls = test_le(test_logit1, targets, 100, sigmoid=True)
    print(f"     p-value (L - sigmoid): {p_val_ls:.4f} (actual: {p_val_ls})")
    # _, p_val_mmd, _ = mmd2_permutations(Kxyxy, mmd_size, permutations=200)
    # print(f"     p-value (MMD): {p_val_mmd:.4f} (actual: {p_val_mmd})")

    results = {
        "e_val": e_val,
        "p_val": p_val_c2st,
        "p_val_l": p_val_l,
        "p_val_ls": p_val_ls,
        # "p_val_mmd": p_val_mmd,
        "test_loss": test_loss,
        "test_acc": test_acc,
        # "test_loss_mmd": test_loss_mmd,
    }
    return results


def train_model_and_do_anytime_tests(args, dataset_ind, setting, input_shape, train, val, test):
    """Train and test model for anytime testing."""

    # Now get batches where each batch is a volume
    # Note that the volumes are either fully clean OR pathological.
    # For Type-1: each volume contains both (simulated) classes.
    # For Type-2: each volume contains only one (true) class.
    # This means we need at least two volumes per round! And check (for Type-2) whether both classes are there.
    # So training proceeds by grabbing two batches -- one for each class -- and training on them. We also grab two
    # batches for validation, and for testing. This is one round. For the next round, we use the validation batch for
    # training, the test batch for validation, and grab a new batch for testing. Repeat until we reject the null or
    # run out of data.
    # Easiest way to do this is probably to change the train/val/test loaders every round.
    def make_data_per_volume(train, val, test):
        data = train.raw_samples + val.raw_samples + test.raw_samples
        data_per_volume = {}
        for datum in data:
            volume = datum.fname.name
            if volume not in data_per_volume:
                data_per_volume[volume] = []
            data_per_volume[volume].append(datum)
        return data_per_volume

    data_per_volume = make_data_per_volume(train, val, test)
    if setting in ["1a", "1b"]:
        for volume in data_per_volume:
            # Need at least two samples in a volume, so we can guarantee a positive and negative slice per
            # volume for the Type-1 setting.
            if len(data_per_volume[volume]) == 1:
                print(
                    f"Reconstructed volume {volume} only has one sample! Make sure to use multiple volumes per batch."
                )
                if args.volumes_per_batch == 1:
                    raise ValueError(
                        "Can't run Type-1 setting with only one volume per batch, since "
                        "there is a reconstructed volume with only one sample! Set `volumes_per_batch` > 1."
                    )
    volumes = [key for key in data_per_volume.keys()]
    num_volumes = len(volumes)

    # Track which volumes are positive and which are negative (for Type-2)
    positive_volumes = []
    negative_volumes = []
    for i, volume in enumerate(volumes):
        # Type-1 volumes contain simulated positive and negative samples, so just assign equally.
        # Note that this may go wrong (lead to volumes with only one class) if we don't use the full dataset, since
        # the smaller datasets have randomly thrown out data without checking volume boundaries. I.e. it has randomly
        # thrown out parts of some (or all) volumes, stratified by True label. This happens in SampledSlices().
        # NOTE: The 'full' filtered 7236 dataset ({False: 3618, True: 8311} --> 2x3618 = 7236) technically has this
        # problem as well. Type-1 data is assigned a simulated label stratified on the slices in its
        # sample (i.e. dataset_size). This assignment does not respect volume boundaries. Hence, when reconstructing
        # volumes later (as we do here), we may end up with a volume that contains only the one simulated class.
        # Type-2 volumes are either fully positive or negative, so we never run into this problem.
        if setting in ["1a", "1b"]:
            labels = [sl.label for sl in data_per_volume[volume]]
            if np.sum(labels) in [0, len(labels)]:  # This is the error mentioned above!
                # This volume is either all simulated positive or all negative. Easy fix: change the label of one slice.
                # Note that this makes the sl.label attribute inconsistent with the sl.slice_pathologies attribute, but
                # that's fine since we won't use the latter anymore. NOTE: Dangerous if we ever need the original!
                # NOTE: This doesn't seem to actually replace the attribute for some reason...
                # data_per_volume[volume][0]._replace(label=not data_per_volume[volume][0].label)
                print(f"Volume {volume} has only one class! Make sure to use multiple volumes!")
                if args.volumes_per_batch == 1:
                    raise ValueError("Not using multiple volumes per batch! Set `volumes_per_batch` > 1.")
            if i % 2 == 0:  # even
                positive_volumes.append(volume)
            else:  # odd
                negative_volumes.append(volume)
        elif setting == "2":  # Check whether the volume is positive or negative
            labels = [sl.label for sl in data_per_volume[volume]]
            if np.sum(labels) not in [0, len(labels)]:
                raise RuntimeError("Volume contains both positive and negative samples!")
            if data_per_volume[volume][0].label == 1:
                positive_volumes.append(volume)
            else:
                negative_volumes.append(volume)

    # Shuffle volumes to get random order for every run (for computing Type-1 and Type-2 errors later).
    random.shuffle(positive_volumes)
    random.shuffle(negative_volumes)

    # Model definition
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
        patience=args.patience,
    )

    # Results dict
    results_per_round = {
        "e_val": [],
        "running_e_val": [],
        "test_loss": [],
        "test_acc": [],
        "result": [],
    }
    halfnum_per_batch = args.volumes_per_batch // 2
    # Note that using this to control the number of rounds means we're throwing away volumes for Type 2.
    min_of_volumes = min(len(positive_volumes), len(negative_volumes))
    num_rounds = (min_of_volumes // halfnum_per_batch - 2 if min_of_volumes % halfnum_per_batch == 0
                  else min_of_volumes // halfnum_per_batch - 1)
    running_e_val = 1
    for i in range(num_rounds):
        print(f"Round {i} / {num_rounds - 1}")
        if i == 0:  # First round, assign volumes.
            train_volumes = [
                positive_volumes[:halfnum_per_batch],
                negative_volumes[:halfnum_per_batch],
            ]
            val_volumes = [
                positive_volumes[halfnum_per_batch: 2 * halfnum_per_batch],
                negative_volumes[halfnum_per_batch: 2 * halfnum_per_batch],
            ]
            test_volumes = [
                positive_volumes[2 * halfnum_per_batch: 3 * halfnum_per_batch],
                negative_volumes[2 * halfnum_per_batch: 3 * halfnum_per_batch],
            ]
        else:  # Subsequent rounds, use validation and test data from previous round.
            if args.do_online_learning:  # Online learning: use validation data from previous round as training data.
                # If we skip rounds when online learning, then we should accumulate data from skipped rounds. When
                #  i == args.skip_rounds, we have skipped args.skip_rounds rounds, so we do one final accumulation and
                #  train. E.g. if args.skip_rounds == 10, then we skip i == [0-9] (10 rounds), and train starting at
                #  i == 10 (round 11) using data from all [0-9] rounds. Then in round 12 (i == 11), we only use train
                #  data from a single new batch.
                if 0 < i <= args.num_skip_rounds:
                    train_volumes += val_volumes
                else:
                    train_volumes = val_volumes
            else:  # Combine train and validation data from previous round.
                train_volumes += val_volumes
            val_volumes = test_volumes
            test_volumes = [
                positive_volumes[(i + 2) * halfnum_per_batch: (i + 3) * halfnum_per_batch],
                negative_volumes[(i + 2) * halfnum_per_batch: (i + 3) * halfnum_per_batch]
            ]

        if i < args.num_skip_rounds:  # Just accumulate data for now
            continue

        # Overwrite dataset structures with these batches
        train.raw_samples = [sl for vol_list in train_volumes for vol in vol_list for sl in data_per_volume[vol]]
        val.raw_samples = [sl for vol_list in val_volumes for vol in vol_list for sl in data_per_volume[vol]]
        test.raw_samples = [sl for vol_list in test_volumes for vol in vol_list for sl in data_per_volume[vol]]

        # Check that we have positive and negative samples in each batch. Using more volumes per batch can help
        #  prevent these errors for Type-1 data settings.
        if i == 0:  # First round, check that we have positive and negative samples in each batch.
            assert len(set([sl.label for sl in train.raw_samples])) == 2, "Train batch contains only one class!"
            assert len(set([sl.label for sl in val.raw_samples])) == 2, "Val batch contains only one class!"
        # Later rounds, only test batches contain new data.
        assert len(set([sl.label for sl in test.raw_samples])) == 2, "Test batch contains only one class!"

        print("Train: {}, Val: {}, Test: {}".format(len(train), len(val), len(test)))
        # Create loaders for this round
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
        (
            train_losses,
            val_losses,
            val_accs,
            extra_output,
            total_time,
        ) = module.train(train_loader, val_loader, print_every=10, eval_every=1)
        # Do testing
        test_loss, test_acc, test_extra_output = module.test(test_loader)
        # Targets are 1-dimensional
        targets = test_extra_output["targets"]
        # logits are 1-dimensional
        test_logit1 = test_extra_output["logits"]
        # test_prob1 is probability of class 1 given by model
        test_prob1 = torch.sigmoid(test_logit1)

        e_val = c2st_e_prob1(targets, test_prob1)
        running_e_val *= e_val
        print(f"E-value: {e_val:.4f} (actual: {e_val})")
        print(f"Running E-value: {running_e_val:.4f} (actual: {running_e_val})")

        results_per_round["e_val"].append(e_val)
        results_per_round["running_e_val"].append(running_e_val)
        results_per_round["test_loss"].append(test_loss)
        results_per_round["test_acc"].append(test_acc)
        results_per_round["result"].append("reject" if e_val > 20 else "accept")
        if running_e_val > 20:  # exit loop if reject
            return results_per_round

        if not args.do_online_learning and args.cold_start:  # Retrain model from scratch every round
            # Apparently reinitialising is not so easy, so just redefine the model with same params.
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
                patience=args.patience,
            )

    return results_per_round


if __name__ == "__main__":
    start_time = time.perf_counter()
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
    if args.test_type == "anytime":
        assert args.volumes_per_batch % 2 == 0, "`volumes_per_batch` must be even for anytime testing."

    # Save args
    date_string = f"{datetime.now():%Y-%m-%d}"
    time_string = f"{datetime.now():%H:%M:%S.%f}"[:-3]  # milliseconds added
    save_dir = args.save_dir / date_string
    save_dir.mkdir(parents=True, exist_ok=True)  # OK if date folder exists already
    save_dir = save_dir / time_string
    save_dir.mkdir(parents=True, exist_ok=False)  # Not OK if time folder exists already
    args_dict = {key: str(value) for key, value in vars(args).items()}
    with open(save_dir / "args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    if args.quick_test:
        args.num_epochs = 1
        args.dataset_sizes = [200]
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
        ds_time = time.perf_counter()
        print(f"\n ----- Dataset size: {dataset_size} ----- ")
        size_results_dict = {dataset_ind: {} for dataset_ind in range(num_samples)}
        if isinstance(args.num_partitions, int):
            if dataset_size * args.num_partitions > max_data:  # Max dataset size
                raise RuntimeError(
                    f"Cannot partition data into {args.num_partitions} bits of size {dataset_size}. Only {max_data} "
                    f"points available."
                )
        # Sample datasets of this size.
        slice_splits = None
        for dataset_ind in range(num_samples):
            ind_time = time.perf_counter()
            print(f"\n  ----- Dataset index: {dataset_ind} ----- ")
            # Do this experiment a bunch of times, so we can get an error estimate.
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
                if args.test_type == "anytime":
                    size_results_dict[dataset_ind][setting] = train_model_and_do_anytime_tests(
                        args, dataset_ind, setting, input_shape, train, val, test
                    )
                else:
                    size_results_dict[dataset_ind][setting] = train_model_and_do_base_tests(
                        args, dataset_ind, setting, input_shape, train, val, test
                    )

            print(f"Dataset index {dataset_ind} took {time.perf_counter() - ind_time:.2f} seconds.")
            # Saving partial results
            size_dir = save_dir / f"size_{dataset_size}"
            size_dir.mkdir(parents=True, exist_ok=True)  # OK if date folder exists already
            print(f"Saving index {dataset_ind} results to {size_dir}")
            with open(size_dir / f"results_index{dataset_ind}.json", "w") as f:
                json.dump({str(key): val for key, val in size_results_dict[dataset_ind].items()}, f, indent=4)

        print(f"Dataset size {dataset_size} took {time.perf_counter() - ds_time:.2f} seconds.")
        # Saving size results
        print(f"Saving size {dataset_size} results to {save_dir}")
        with open(save_dir / f"results_size{dataset_size}.json", "w") as f:
            json.dump({str(key): val for key, val in size_results_dict.items()}, f, indent=4)
        results_dict[dataset_size] = size_results_dict

    # Saving final results
    print(f"Saving final results to {save_dir}")
    with open(save_dir / "results.json", "w") as f:
        json.dump({str(key): val for key, val in results_dict.items()}, f, indent=4)

    print(f"Full run took {time.perf_counter() - start_time:.2f} seconds.")