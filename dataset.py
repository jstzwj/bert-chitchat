import os
import time
import json
import pickle
import logging
from filelock import FileLock

import torch
import numpy as np

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, file_path: str, block_size: int=512, overwrite_cache=False):
        super(TextDataset, self).__init__()
        self.path = file_path
        assert os.path.isfile(file_path)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.data = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.data = []
                with open(file_path, encoding="utf-8") as f:
                    for each_line in f:
                        obj = json.loads(each_line)
                        tokenized_source = tokenizer.encode(obj['source'], truncation=True, max_length=block_size, padding=True)
                        tokenized_target = tokenizer.encode(obj['target'], truncation=True, max_length=block_size, padding=True)
                        self.data.append((tokenized_source, tokenized_target))

                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __getitem__(self, index:int) -> torch.Tensor:
        return {
            'source': torch.tensor(self.data[index][0], dtype=torch.long),
            'target': torch.tensor(self.data[index][1], dtype=torch.long)
        }
    def __len__(self):
        return len(self.data)