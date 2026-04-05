"""Efficient data loading for language model training."""

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Simple dataset that serves chunks from a pre-tokenized tensor.

    For experimentation, you can create synthetic data:
        data = torch.randint(0, vocab_size, (total_tokens,))
        dataset = TextDataset(data, seq_len=1024)

    Or load real tokenized data:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        dataset = TextDataset(torch.tensor(tokens), seq_len=1024)
    """

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def create_dataloader(
    data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader from tokenized data."""
    dataset = TextDataset(data, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
