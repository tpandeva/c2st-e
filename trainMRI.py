import random
import numpy as np
import torch
import pandas as pd
import torch.fft
from pathlib import Path
from functools import partial
from torch.distributions import Normal

from experiments.mri.fastmri_plus import (
    PathologiesSliceDataset,
    PathologyTransform,
    get_pathology_info,
    populate_slice_filter,
)
from models.mri_model import ModelModule


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
    data_seed = 0
    seed = 0

    num_datasets = 1
    sample_rates = [1]

    crop_size = 320

    batch_size = 64
    num_workers = 20

    input_shape = (crop_size, crop_size)
    in_chans = 1
    chans = 16
    num_pool_layers = 4
    drop_prob = 0.0

    num_epochs = 1
    lr = 1e-5
    total_lr_gamma = 1

    # Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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

    slice_filter = partial(partial(populate_slice_filter, clean_volumes), all_pathologies)

    # ----------------------------
    # ------ fastMRI data ------
    # ----------------------------

    pathology_transform = PathologyTransform(crop_size=crop_size)

    for sample_rate in sample_rates:
        print(f" ----- Sample rate: {sample_rate} ----- ")
        train_dataset = PathologiesSliceDataset(
            root=train,
            challenge="singlecoil",  # Doesn't do anything right now, because pathologies labeled using RSS.
            transform=pathology_transform,
            raw_sample_filter=slice_filter,  # For populating slices with pathologies.
            pathology_df=pathology_df,  # For volume metadata and for populating slices with pathologies.
            clean_volumes=clean_volumes,  # For equalising clean VS. pathology volumes.
            seed=data_seed,
            use_center_slices_only=True,
            sample_rate=sample_rate,
            num_datasets=num_datasets,
        )
        val_dataset = PathologiesSliceDataset(
            root=val,
            challenge="singlecoil",  # Doesn't do anything right now, because pathologies labeled using RSS.
            transform=pathology_transform,
            raw_sample_filter=slice_filter,  # For populating slices with pathologies.
            pathology_df=pathology_df,  # For volume metadata and for populating slices with pathologies.
            clean_volumes=clean_volumes,  # For equalising clean VS. pathology volumes.
            seed=data_seed,
            use_center_slices_only=True,
            sample_rate=sample_rate,
            num_datasets=num_datasets,
        )
        test_dataset = PathologiesSliceDataset(
            root=test,
            challenge="singlecoil",  # Doesn't do anything right now, because pathologies labeled using RSS.
            transform=pathology_transform,
            raw_sample_filter=slice_filter,  # For populating slices with pathologies.
            pathology_df=pathology_df,  # For volume metadata and for populating slices with pathologies.
            clean_volumes=clean_volumes,  # For equalising clean VS. pathology volumes.
            seed=data_seed,
            use_center_slices_only=True,
            sample_rate=sample_rate,
            num_datasets=num_datasets,
        )

        # One experiment per dataset
        for dataset in range(num_datasets):
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                worker_init_fn=None,
                shuffle=True,
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=256,
                num_workers=num_workers,
                worker_init_fn=None,
                shuffle=False,
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=256,
                num_workers=num_workers,
                worker_init_fn=None,
                shuffle=False,
            )

            # --------------------------------------
            # ------ Model, training, testing ------
            # --------------------------------------

            module = ModelModule(
                in_chans,
                chans,
                num_pool_layers,
                drop_prob,
                input_shape,
                lr,
                total_lr_gamma,
                num_epochs,
            )
            print("Encoding size:", module.model.enc_size)

            train_losses, val_losses, val_accs, extra_output, total_time = module.train(
                train_loader, val_loader, print_every=1, eval_every=2
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
            print(f"1 / E-value: {1 / e_val:.4f} (actual: {1 / e_val})")
            p_val_c2st = c2st_prob1(targets, test_prob1).item()
            print(f"    p-value: {p_val_c2st:.4f} (actual: {p_val_c2st})")

            train_dataset.next_dataset()
            val_dataset.next_dataset()
            test_dataset.next_dataset()
