# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from compressai.registry import register_dataset

import numpy as np 


@register_dataset("FeatureFolder")
class FeatureFolder(Dataset):
    """Load an feature folder database. Training and testing feature samples
    are respectively stored in separate directories:
    Args:
        root (string): root directory of the dataset
        split (string): split mode ('train' or 'test')
    """

    def __init__(self, root, split="train", suffix='.npy'):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        # self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())
        self.samples = sorted([f for f in splitdir.iterdir() if f.is_file() and f.name.endswith(suffix)])
        #gcs
        # self.samples = self.samples[:320]

    def __getitem__(self, index):
        # Load feature, use float32 for training
        feat = np.load(self.samples[index]).astype(np.float32)
        feat = np.expand_dims(feat, axis=0) # (C,H,W)
        # print(feat.shape) # (1, 256, 256)
        return feat

    def __len__(self):
        return len(self.samples)


@register_dataset("ConcatFeatureFolder")
class ConcatFeatureFolder(Dataset):
    """
    Combine multiple FeatureFolder datasets into one.
    
    Args:
        roots (list): list of root directories, each with 'train/' and 'test/' subdirs
        split (str): 'train' or 'test'
    """

    def __init__(self, roots, split="train", suffix='.npy'):
        if isinstance(roots, str):
            roots = [roots]
        self.datasets = [FeatureFolder(root, split=split, suffix=suffix) for root in roots]
        self.concat_dataset = ConcatDataset(self.datasets)

    def __getitem__(self, index):
        return self.concat_dataset[index]

    def __len__(self):
        return len(self.concat_dataset)
