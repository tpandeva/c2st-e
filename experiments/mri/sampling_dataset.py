import os
import random
import copy
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import (
    Callable,
    Optional,
    Tuple,
    Union,
    NamedTuple,
    Dict,
    Any,
    Sequence,
)

import h5py
import numpy as np
import pandas as pd
import torch.fft
from .fastmri import (
    to_tensor,
    complex_abs,
    normalize_instance,
    complex_center_crop,
    fft2c,
    ifft2c,
)

from .fastmri_plus import et_query


def populate_slice_filter_with_labels(clean_volumes, all_pathologies, raw_sample):
    # Filter for populating slices with pathology information.
    # (pathology info for volume also in metadata)
    fname = raw_sample.fname
    slice_ind = raw_sample.slice_ind
    metadata = raw_sample.metadata

    pathologies_of_volume = metadata["pathologies"]
    # Pathologies in this slice
    pathologies_of_slice = pathologies_of_volume[
        (pathologies_of_volume["slice"] == slice_ind)
    ]
    # Replace empty list with n-hot of pathologies (needs to be n-hot for batching later)
    # `artifact` is also included as positive class.
    n_hot_pathologies = np.zeros(len(all_pathologies), dtype=int)
    for pathology in list(pathologies_of_slice["label"].values):
        n_hot_pathologies[all_pathologies.index(pathology)] += 1

    if n_hot_pathologies.sum() != len(list(pathologies_of_slice["label"].values)):
        raise RuntimeError("Pathologies got lost...")

    raw_sample = raw_sample._replace(slice_pathologies=n_hot_pathologies, label=(n_hot_pathologies.sum() > 0))

    # Keep slices belonging to clean volumes AND slices with pathologies,
    # BUT NOT slices in non-clean volumes that don't have pathologies.
    # This is fine in terms of data numbers, because we have many more pathology volumes than clean volumes.
    keep = clean_volumes[fname.name[:-3]] or len(pathologies_of_slice) > 0
    #     print(clean_volumes[fname.name[:-3]], len(pathologies_of_slice))
    return raw_sample, keep


class FastMRIPathologyDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]
    slice_pathologies: Sequence[str]
    label: Union[int, None]


class FilteredSlices:
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
        pathology_df: Optional[pd.DataFrame] = None,
        use_center_slices_only: Optional[bool] = None,
        seed: Optional[int] = None,
        quick_test: bool = False,
    ):
        """
        Args:
            root: Path to the dataset.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
        Added args:
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
            pathology_df: fastMRI+ pathologies dataframe.
            clean_volumes: {filename: volume clean True/False} dictionary.
            seed: random seed for shuffling operations (also affect sample_rate operations).
            use_center_slices_only: bool, whether to use only the center half of volumes.
            quick_test: ignore 99% of data for quick test.
        """

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        if seed is not None:
            random.seed(seed)

        self.recons_key = "reconstruction_rss"

        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        self.pathology_df = pathology_df

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        total_slices = 0
        total_slices_halved = 0
        total_slices_halved_filtered = 0
        files = list(Path(root).iterdir())
        if quick_test:
            random.shuffle(files)
            files = files[: int(len(files) * 0.03)]
        for fname in sorted(files):
            metadata, num_slices = self._retrieve_metadata(fname)
            total_slices += num_slices

            new_raw_samples = []
            for slice_ind in range(num_slices):
                if use_center_slices_only:
                    # Use only center half of slices, because edges contains more noise.
                    if (
                        slice_ind < num_slices // 4
                        or slice_ind > num_slices * 3 // 4
                    ):
                        continue
                total_slices_halved += 1
                raw_sample = FastMRIPathologyDataSample(fname, slice_ind, metadata, [], None)
                # Apply pathology filter here
                filtered_sample, keep = self.raw_sample_filter(raw_sample)
                if keep:
                    new_raw_samples.append(filtered_sample)
                    total_slices_halved_filtered += 1
            self.raw_samples += new_raw_samples
            del metadata[
                "pathologies"
            ]  # Delete df from sample for later collation.

        # Here we have obtained filtered data (only center samples from good volumes). Samples can come from here.

        if num_cols:
            self.raw_samples = [
                ex
                for ex in self.raw_samples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]

        print("All filtered data", self.get_label_counts(self.raw_samples))

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            pathologies = self.pathology_df[
                self.pathology_df["file"] == fname.name[:-3]
            ]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                "pathologies": pathologies,
                **hf.attrs,
            }

        return metadata, num_slices

    def __len__(self):
        return len(self.raw_samples)

    @staticmethod
    def get_label_counts(slices):
        true_counts = {False: 0, True: 0}
        counts = {False: 0, True: 0}
        for slic in slices:
            fname, dataslice, metadata, slice_pathologies, label = slic
            true_label = np.sum(slice_pathologies) > 0
            true_counts[true_label] += 1
            counts[label] += 1
        return true_counts, counts


class SampledSlices:
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        base_dataset,
        dataset_size: int,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            base_dataset: dataset to sample from.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            dataset_size: size of dataset to sample train-test from.
        """

        self.base_dataset = base_dataset
        self.transform = transform
        self.dataset_size = dataset_size

        # Get stratified split of size dataset_size form base_dataset of correct size.
        self.stratified_data = self.stratify_true_label(base_dataset.raw_samples, dataset_size)

        # ----- Assign data to partitions -----

        # Type-II error
        # Use half the full data for type two, to correspond with type1 data size
        data_copy2 = copy.deepcopy(self.stratified_data)  # in case we mutate
        sampled_data_for_type2, _ = self.stratified_split_true_label(data_copy2, dataset_size // 2)
        # Half train, half test
        train_split2, test_split2 = self.stratified_split_true_label(
            sampled_data_for_type2, len(sampled_data_for_type2) // 2
        )
        # Make val from test (could do k-fold or some stuff as well).
        train_split2, val_split2 = self.stratified_split_true_label(train_split2, int(0.8 * len(train_split2)))
        train = SliceDataset(train_split2, self.transform)
        val = SliceDataset(val_split2, self.transform)
        test = SliceDataset(test_split2, self.transform)
        type2_data = {train, val, test}

        # Type-I error
        # Have two options: true class 0 and true class 1 training. Do both.
        # Get data with label 0 and with label 1
        data_copy1 = copy.deepcopy(self.stratified_data)  # Deepcopy because label will be mutated later.
        true0_data, true1_data = self.split_data_by_true_label(data_copy1)  # 1a, 1b

        # Type-1a, use true0 data, split stratified, but with re-assigned label!
        train_split1a, test_split1a = self.stratified_split_false_labels(true0_data, False)
        # True label has been mutated, so we can do a true label split to get val data
        train_split1a, val_split1a = self.stratified_split_true_label(train_split1a, int(0.8 * len(train_split1a)))
        train = SliceDataset(train_split1a, self.transform)
        val = SliceDataset(val_split1a, self.transform)
        test = SliceDataset(test_split1a, self.transform)
        type1a_data = {train, val, test}

        # Type-1b
        train_split1b, test_split1b = self.stratified_split_false_labels(true1_data, True)
        # True label has been mutated, so we can do a true label split to get val data
        train_split1b, val_split1b = self.stratified_split_true_label(train_split1b, int(0.8 * len(train_split1b)))
        train = SliceDataset(train_split1b, self.transform)
        val = SliceDataset(val_split1b, self.transform)
        test = SliceDataset(test_split1b, self.transform)
        type1b_data = {train, val, test}

        print("all", self.get_label_counts(self.stratified_data))
        print("\n")
        print("type1_full", self.get_label_counts(data_copy1))
        print("type2_full", self.get_label_counts(data_copy2))
        print("\n")
        print("type1a_true0", self.get_label_counts(true0_data))
        print("type1b_true1", self.get_label_counts(true1_data))
        print("type2_sampled", self.get_label_counts(sampled_data_for_type2))
        print("\n")
        print("type1a_train", self.get_label_counts(train_split1a))
        print("type1a_val", self.get_label_counts(val_split1a))
        print("type1a_test", self.get_label_counts(test_split1a))
        print("\n")
        print("type1b_train", self.get_label_counts(train_split1b))
        print("type1b_val", self.get_label_counts(val_split1b))
        print("type1b_test", self.get_label_counts(test_split1b))
        print("\n")
        print("type2_train", self.get_label_counts(train_split2))
        print("type2_val", self.get_label_counts(val_split2))
        print("type2_test", self.get_label_counts(test_split2))
        print("\n")

        # Combined
        self.datasets_dict = {"1a": type1a_data, "1b": type1b_data, "2": type2_data}

    # raw_sample._replace(slice_pathologies=n_hot_pathologies)

    @staticmethod
    def stratify_true_label(slices, dataset_size):
        random.shuffle(slices)
        # Get stratified slices: 0, 1 each dataset_size // 2 instances.
        num_per_class = dataset_size // 2
        stratified_slices = []
        num_pos, num_neg = 0, 0
        for slic in slices:
            fname, dataslice, metadata, slice_pathologies, true_label = slic
            if true_label and num_pos < num_per_class:
                stratified_slices.append(slic)
                num_pos += 1
            elif not true_label and num_neg < num_per_class:
                stratified_slices.append(slic)
                num_neg += 1
            else:
                continue
        return stratified_slices

    @staticmethod
    def stratified_split_true_label(slices, size_of_first_split):
        random.shuffle(slices)
        # Split into (size_of_first_split, rest). Both splits contain half of each label.
        split1, split2 = [], []
        num_per_class_split1 = size_of_first_split // 2
        num_pos_split1, num_neg_split1 = 0, 0
        for slic in slices:
            fname, dataslice, metadata, slice_pathologies, true_label = slic
            if true_label:
                if num_pos_split1 < num_per_class_split1:
                    split1.append(slic)
                    num_pos_split1 += 1
                else:
                    split2.append(slic)
            else:
                if num_neg_split1 < num_per_class_split1:
                    split1.append(slic)
                    num_neg_split1 += 1
                else:
                    split2.append(slic)
        return split1, split2

    @staticmethod
    def split_data_by_true_label(slices):
        random.shuffle(slices)
        # Splits contain either only 0 or 1 as true label
        true0, true1 = [], []
        for slic in slices:
            fname, dataslice, metadata, slice_pathologies, true_label = slic
            if true_label:
                true1.append(slic)
            else:
                true0.append(slic)
        return true0, true1

    @staticmethod
    def stratified_split_false_labels(slices, expected_label):
        random.shuffle(slices)
        # Split slices into two parts (e.g. train-test), with a random (stratified) label.
        # Assumes all true labels are the same
        size_of_split = len(slices) // 2
        labels_to_flip0 = labels_to_flip1 = size_of_split // 2
        flipped_labels0, flipped_labels1 = 0, 0  # 0s and 1s that have flipped
        split0, split1 = [], []
        for slic in slices:
            fname, dataslice, metadata, slice_pathologies, true_label = slic
            if true_label != expected_label:
                raise RuntimeError("Expected different label!")
            if len(split0) < size_of_split:
                if flipped_labels0 < labels_to_flip0:  # Flip label
                    slic = slic._replace(label=~true_label)
                    flipped_labels0 += 1
                split0.append(slic)
            else:
                if flipped_labels1 < labels_to_flip1:
                    slic = slic._replace(label=~true_label)
                    flipped_labels1 += 1
                split1.append(slic)
        return split0, split1

    @staticmethod
    def get_label_counts(slices):
        true_counts = {False: 0, True: 0}
        counts = {False: 0, True: 0}
        for slic in slices:
            fname, dataslice, metadata, slice_pathologies, label = slic
            true_label = np.sum(slice_pathologies) > 0
            true_counts[true_label] += 1
            counts[label] += 1
        return true_counts, counts


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
            self,
            slices,
            transform: Optional[Callable] = None,
    ):
        """
        Args:
            base_dataset: dataset to sample from.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
        """

        self.transform = transform
        self.raw_samples = slices

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        print(self.raw_samples)
        fname, dataslice, metadata, slice_pathologies, label = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            image = hf["reconstruction_rss"][dataslice] if self.recons_key in hf else None
            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            sample = (
                kspace,
                image,
                None,
                None,
                attrs,
                fname.name,
                dataslice,
                slice_pathologies,
                label
            )
        else:
            sample = self.transform(
                kspace, image, attrs, fname.name, dataslice, slice_pathologies, label
            )

        return sample


class PathologyLabelTransform:
    """
    Data Transformer for pathology model with U-Net encoder.
    """

    def __init__(
        self,
        crop_size: Optional[int] = None,
    ):
        """
        Args:
            crop_size: int, size to crop to (square).
        """

        self.crop_size = crop_size

    def __call__(
        self,
        kspace: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_ind: int,
        pathology_labels: np.ndarray,
        label: bool,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
        str,
        int,
        np.ndarray,
        bool,
    ]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_ind: Serial number of the slice.
            pathology_labels: n-hot array of pathology labels.
        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        kspace = to_tensor(kspace)

        # inverse Fourier transform to get zero filled solution
        image = ifft2c(kspace)

        # crop input to correct size
        if target is not None:
            true_crop_size = (target.shape[-2], target.shape[-1])
        else:
            true_crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < true_crop_size[1]:
            true_crop_size = (image.shape[-2], image.shape[-2])

        # Crop image to stated resolution
        image = complex_center_crop(image, true_crop_size)

        # Proper image constructed from kspace. From here we can do cropping (also in kspace).
        if self.crop_size is not None:
            # to kspace
            image = fft2c(image)
            # crop in kspace
            image = complex_center_crop(image, (self.crop_size, self.crop_size))
            kspace = image
            # to image space
            image = ifft2c(image)

            # absolute value
        image = complex_abs(image)

        #         # apply Root-Sum-of-Squares if multicoil data
        #         if self.which_challenge == "multicoil":
        #             image = fastmri.rss(image)

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # Flip image (upside down): NOTE: this means kspace will not correspond to exactly image anymore.
        image = torch.flip(image, dims=(0,))

        # Image constructed from kspace
        return kspace, image, mean, std, attrs, fname, slice_ind, pathology_labels, label
