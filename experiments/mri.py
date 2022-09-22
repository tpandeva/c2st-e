"""
Copyright (c) Facebook, Inc. and its affiliates.

Part of this source code is licensed under the MIT license.
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
    List,
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
    one_hot_pathologies = np.zeros(len(all_pathologies), dtype=int)
    for pathology in list(pathologies_of_slice["label"].values):
        one_hot_pathologies[all_pathologies.index(pathology)] += 1

    if one_hot_pathologies.sum() != len(list(pathologies_of_slice["label"].values)):
        raise RuntimeError("Pathologies got lost...")

    raw_sample = raw_sample._replace(slice_pathologies=one_hot_pathologies)

    # Keep slices belonging to clean volumes AND slices with pathologies,
    # BUT NOT slices in non-clean volumes that don't have pathologies.
    # This is fine in terms of data numbers, because we have many more pathology volumes than clean volumes.
    keep = clean_volumes[fname.name[:-3]] or len(pathologies_of_slice) > 0
    #     print(clean_volumes[fname.name[:-3]], len(pathologies_of_slice))
    return raw_sample, keep


# --------------------------------
# ------ fastMRI operations ------
# --------------------------------
