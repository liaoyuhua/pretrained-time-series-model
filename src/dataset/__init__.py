from typing import List, Dict
from multiprocessing import cpu_count
import logging
import warnings
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numba

logger = logging.getLogger(__name__)


# raise error if numba is not installed
try:
    import numba
except ImportError:
    raise ImportError(
        "Please install numba first, you can refer to https://numba.readthedocs.io/en/stable/user/installing.html"
    )


@numba.jit(nopython=True)
def _find_end_indices(
    diffs: np.ndarray, input_len: int, last_idx: List[int]
) -> np.ndarray:
    """
    Find the end indices of each input sequence.

    Args:
        diffs (np.ndarray): array of differences to next time step. nans should be filled up with ones
        input_len (int): length of the input sequence.
        last_idx (List[int]): list of last index of each group.

    Returns:
        np.ndarray: array of end indices
    """
    end_indices = []
    mask_indices = []  # 0 for not mask, 1 for mask

    for start_idx, _ in enumerate(diffs):
        end_idx = start_idx
        length = 1
        mask = [1] * input_len
        max_idx = last_idx[start_idx]
        if start_idx == max_idx:
            end_indices.append(start_idx)
            mask[length - 1] = 0
            mask_indices.append(mask)
            continue

        while length <= input_len and end_idx <= max_idx:
            mask[length - 1] = 0
            length += diffs[end_idx]
            end_idx += 1
        end_indices.append(end_idx - 1)
        mask_indices.append(mask)

    return np.asarray(end_indices), np.asarray(mask_indices)


class PretrainDataset(Dataset):
    """
    Required time column format:
    - year: 2019 / 2019-01-01 / 2019-01-01 00:00:00
    - month: 2019-01 / 2019-01-01 / 2019-01-01 00:00:00
    - day: 2019-01-01 / 2019-01-01 00:00:00
    - hour: 2019-01-01 01:00:00
    - minute: 2019-01-01 01:10:00
    - second: 2019-01-01 01:10:05

    Other formats are not ensured to work properly.

    TODO: add support for time series with covariates.
    """

    freq = ("y", "m", "d", "h", "t", "s")
    freq_name_map = {
        "y": "year",
        "m": "month",
        "d": "day",
        "h": "hour",
        "t": "minute",
        "s": "second",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        group_id: str,
        time_col: str,
        time_index: str,
        target: str,
        seq_len: int,
        min_count_per_sample: int,
        stride: int,
        freq: str = "d",
    ) -> None:
        """
        Args:
            data (pd.DataFrame): raw data with time index and target.
            group_id (str): group id column name like country, store name, etc.
            time_col (str): time column name like takeoffdate, orderdate, etc.
            time_index (str): time index column, which is the absolute time index
            for each group by ascending order.
            target (str): target column name.
            seq_len (int): length of the input sequence.
            min_count_per_sample (int): minimum number of valid data points in each sample sequence.
            stride (int): stride of sliding window, whcih should set appropriately to avoid overfitting.
            freq (str): frequency of the time series, default daily.
        """
        super().__init__()

        self.target = target
        self.group_id = group_id
        assert isinstance(time_col, str), "time_col must be a string"
        self.time_col = time_col
        assert isinstance(time_index, str), "time_index must be a string"
        self.time_index = time_index
        self.seq_len = seq_len
        self.min_count_per_sample = min_count_per_sample
        self.stride = stride
        self.freq = freq

        self.data, self.scaler, self.group_encoder = self._preprocess(data)
        self.index, self.mask = self._construct_index(
            self.data
        )  # waste a little memory for speed up
        self.tensors = self._to_tensor(self.data)

    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # why double underline for these two columns?
        # to avoid conflict with other raw data columns
        # why save special columns in data?
        # to save raw meta information for dataset
        data["__idx__"] = np.arange(len(data))
        data["__target__"] = data[self.target]
        data["__group_id__"] = data[self.group_id]

        # convert time index to datetime
        data[self.time_col] = pd.to_datetime(data[self.time_col])

        data.sort_values(
            by=[self.group_id, self.time_index], inplace=True, ignore_index=True
        )

        # normalize target
        scaler = StandardScaler()
        data[self.target] = scaler.fit_transform(
            data[self.target].values.reshape(-1, 1)
        )

        # encode group id
        encoder = LabelEncoder()
        data[self.group_id] = encoder.fit_transform(data[self.group_id].values)

        return data, scaler, encoder

    def _construct_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Construct index for each sample sequence.
        """
        g = data.groupby(self.group_id)

        df_index_first = g["__idx__"].transform("nth", 0).to_frame("idx_first")
        df_index_last = g["__idx__"].transform("nth", -1).to_frame("idx_last")

        df_index_diff_to_next = (
            -g["__idx__"].diff(-1).fillna(-1).astype(int).to_frame("idx_diff_to_next")
        )
        df_index = pd.concat(
            [df_index_first, df_index_last, df_index_diff_to_next], axis=1
        )
        df_index["temporal_idx"] = data[self.time_index]
        df_index["index_start"] = data["__idx__"]

        df_index["index_end"], np_mask = _find_end_indices(
            df_index["idx_diff_to_next"].values,
            self.seq_len,
            df_index["idx_last"].values,
        )

        # get the complete items of each sample
        df_index["count"] = df_index["index_end"] - df_index["index_start"] + 1

        # filter out samples with too few items
        np_mask = np_mask[df_index["count"] >= self.min_count_per_sample]
        df_index = df_index[df_index["count"] >= self.min_count_per_sample].reset_index(
            drop=True
        )
        assert len(df_index) == len(np_mask), "length of df_index and np_mask mismatch"

        return df_index, np_mask

    def _to_tensor(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Convert data to tensor to save time for numpy/dataframe-to-tensor.

        TODO: add support for time series with covariates.
        """
        target = torch.tensor(
            data[self.target].values.astype(np.float32), dtype=torch.float32
        )
        time_index = torch.tensor(
            data[self.time_index].values.astype(np.int64), dtype=torch.int64
        )
        group_id = torch.tensor(
            data[self.group_id].values.astype(np.int64), dtype=torch.long
        )
        mask = torch.tensor(self.mask.astype(np.int64), dtype=torch.long)

        tensors = dict(
            target=target,
            time_index=time_index,
            group_id=group_id,
            mask=mask,
        )

        return tensors

    def __getitem__(self, idx):
        index = self.index.iloc[idx]

        target = self.tensors["target"][
            index["index_start"] : index["index_end"] + 1
        ].clone()  # why clone? to avoid in-place operation

        time_index = self.tensors["time_index"][
            index["index_start"] : index["index_end"] + 1
        ].clone()

        group_id = self.tensors["group_id"][
            index["index_start"] : index["index_end"] + 1
        ].clone()
        # group id should be the same for each sample
        assert (
            group_id[0] == group_id
        ).all(), "group id should be the same for each sample"
        assert len(target) == len(time_index) == len(group_id), "length mismatch"

        mask = self.tensors["mask"][idx].clone()

        assert len(mask) == self.seq_len, "mask length mismatch"

        return dict(
            target=target,
            time_index=time_index,
            group_id=group_id,
            mask=mask,
        )

    def __len__(self):
        return len(self.index)

    def to_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = cpu_count(),
        pin_memory: bool = False,
        **kwargs,
    ) -> DataLoader:
        # set proper number of workers
        if num_workers > cpu_count() or num_workers < 0:
            warnings.warn(
                f"num_workers should be in the range [0, {cpu_count()}], "
                f"got {num_workers}. Set num_workers to {cpu_count()}."
            )
            num_workers = cpu_count()
        else:
            num_workers = min(num_workers, cpu_count())  # avoid too many workers

        if hasattr(self, "_collate_fn"):
            collate_fn = self._collate_fn

        default_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        default_kwargs.update(kwargs)

        return DataLoader(
            self,
            **default_kwargs,
        )

    def _collate_fn(
        self, batches: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function to combine items into mini-batch for dataloader.

        ## Padding
        data from :py:meth:`~__getitem__`:
        - mask [0, 1, 0, 1, 0]
        - target [0.1, 0.3, 0.2]
        - time_index [1, 3, 5]
        - group_id [1, 1, 1]

        padding to the length of seq_len:
        - mask [0, 1, 0, 1, 0]
        - target [0.1, 0, 0.3, 0, 0.2]
        - time_index [1, 0, 3, 0, 5]
        - group_id [1, 0, 1, 0, 1]

        Args:
            List[Dict[str, torch.Tensor], torch.Tensor]: List of samples generated with
                :py:meth:`~__getitem__`.

        Returns:
            Dict[str, torch.Tensor]: minibatch
        """
        # lengths = torch.tensor([len(batch["target"]) for batch in batches])
        mask = torch.stack([batch["mask"] for batch in batches])

        # padding target
        targets = torch.zeros(
            len(batches), self.seq_len, dtype=batches[0]["target"].dtype
        )

        # padding time index
        time_index = torch.zeros(
            len(batches), self.seq_len, dtype=batches[0]["time_index"].dtype
        )

        # padding group id
        group_id = torch.zeros(
            len(batches), self.seq_len, dtype=batches[0]["group_id"].dtype
        )

        for i, batch in enumerate(batches):
            targets[i, mask[i] == 0] = batch["target"]
            time_index[i, mask[i] == 0] = batch["time_index"]
            group_id[i, mask[i] == 0] = batch["group_id"]

        return dict(
            target=targets.unsqueeze(-1),
            time_index=time_index,
            group_id=group_id,
            mask=mask,
        )
