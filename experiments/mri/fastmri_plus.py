"""
This code has been partially adapted from the fastMRI repository, for which the following license holds:

Copyright (c) Facebook, Inc. and its affiliates. This source code is licensed under the MIT license.
"""

import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import h5py
import numpy as np
import pandas as pd
import torch
import torch.fft
from collections import defaultdict

from .fastmri import (
    to_tensor,
    complex_abs,
    normalize_instance,
    complex_center_crop,
    fft2c,
    ifft2c,
)


def get_pathology_info(folders_to_check, pathology_df, check_df):
    not_checked = defaultdict(bool)
    no_pathologies = defaultdict(bool)
    any_pathologies = defaultdict(bool)

    all_pathologies = set([])

    for folder in folders_to_check:
        for fname in folder.iterdir():
            name = fname.name[:-3]
            if name in not_checked or name in no_pathologies or name in any_pathologies:
                raise RuntimeError("Found volume in multiple partitions!")

            if name not in check_df["file"].values:
                not_checked[name] = 1
                continue

            pathologies = pathology_df[pathology_df["file"] == name]
            all_pathologies = all_pathologies | set(pathologies["label"].values)
            num_pathologies = len(pathologies)
            if num_pathologies == 0:
                no_pathologies[name] = True
            else:
                any_pathologies[name] = False
    return not_checked, no_pathologies, any_pathologies, list(all_pathologies)


def populate_slice_filter(clean_volumes, all_pathologies, raw_sample):
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

    raw_sample = raw_sample._replace(slice_pathologies=n_hot_pathologies)

    # Keep slices belonging to clean volumes AND slices with pathologies,
    # BUT NOT slices in non-clean volumes that don't have pathologies.
    # This is fine in terms of data numbers, because we have many more pathology volumes than clean volumes.
    keep = clean_volumes[fname.name[:-3]] or len(pathologies_of_slice) > 0
    #     print(clean_volumes[fname.name[:-3]], len(pathologies_of_slice))
    return raw_sample, keep


# --------------------------------------------
# ------ Below is based on fastMRI code ------
# --------------------------------------------


def et_query(
    root: etree.Element,
    qlist,
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]
    slice_pathologies: Sequence[str]


class PathologiesSliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
        pathology_df: Optional[pd.DataFrame] = None,
        clean_volumes: Optional[dict] = None,
        use_center_slices_only: Optional[bool] = None,
        seed: Optional[int] = None,
        num_datasets: int = 1,
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
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
            num_datasets: int, number of datasets to create.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        if seed is not None:
            random.seed(seed)

        self.num_datasets = num_datasets
        self.current_dataset_index = 0

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform

        # Pathologies were labeled using RSS, so always use RSS.
        #         self.recons_key = (
        #             "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        #         )
        self.recons_key = "reconstruction_rss"

        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        self.pathology_df = pathology_df

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        total_slices = 0
        total_slices_halved = 0
        total_slices_halved_filtered = 0
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
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
                    raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata, [])
                    # Apply pathology filter here
                    filtered_sample, keep = self.raw_sample_filter(raw_sample)
                    if keep:
                        new_raw_samples.append(filtered_sample)
                        total_slices_halved_filtered += 1
                self.raw_samples += new_raw_samples
                del metadata[
                    "pathologies"
                ]  # Delete df from sample for later collation.

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.raw_samples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.raw_samples = [
                raw_sample
                for raw_sample in self.raw_samples
                if raw_sample[0].stem in sampled_vols
            ]

        if num_cols:
            self.raw_samples = [
                ex
                for ex in self.raw_samples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]

        # Equalise number of pathology examples.
        # 1) Get min. of num clean / pathology slices for this data partition.
        # 2) Randomly throw out extra slices.
        print("Total", total_slices)
        print("Total halved", total_slices_halved)
        print("Total halved filtered", total_slices_halved_filtered)
        print("Remaining (should match above)", len(self.raw_samples))
        clean_counts = {True: 0, False: 0}
        for sample in self.raw_samples:
            fname = sample.fname
            name = fname.name[:-3]
            clean_true_false = clean_volumes[name]
            clean_counts[clean_true_false] += 1
        print("Clean vs. pathology counts", clean_counts)
        max_slices_class = max(clean_counts.keys(), key=(lambda k: clean_counts[k]))
        min_slices_class = min(clean_counts.keys(), key=(lambda k: clean_counts[k]))
        max_slices = clean_counts[max_slices_class]
        # min. of (num clean, num pathology)
        min_slices = clean_counts[min_slices_class]

        # Shuffle to randomize what we throw out.
        random.shuffle(self.raw_samples)
        # Go from the back so we don't mess up the loop when deleting stuff.
        for i in reversed(range(len(self.raw_samples))):
            if max_slices == min_slices:
                break  # Stop once equal number of slices in both classes.
            sample = self.raw_samples[i]
            fname = sample.fname
            name = fname.name[:-3]
            # Throw out if class of volume is not min_slices_class, until min_slices remain in each class.
            if clean_volumes[name] != min_slices_class:
                self.raw_samples.pop(i)
                max_slices -= 1
        print("Final remaining", len(self.raw_samples))

    def first_dataset(self):
        # Go to first dataset
        self.current_dataset_index = 0

    def next_dataset(self):
        # Go to next dataset
        if self.current_dataset_index < self.num_datasets - 1:
            self.current_dataset_index += 1
        else:
            raise RuntimeError("Already at last dataset.")

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

    def __getitem__(self, i: int):
        fname, dataslice, metadata, slice_pathologies = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            image = hf[self.recons_key][dataslice] if self.recons_key in hf else None

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
            )
        else:
            sample = self.transform(
                kspace, image, attrs, fname.name, dataslice, slice_pathologies
            )

        return sample


class PathologyTransform:
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
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
        str,
        int,
        np.ndarray,
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

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

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
        return kspace, image, mean, std, attrs, fname, slice_ind, pathology_labels
