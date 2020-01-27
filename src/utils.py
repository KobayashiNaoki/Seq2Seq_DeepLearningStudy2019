import torch


def make_mask_by_lengths(sequence_lengths, max_length=-1):
    if max_length == -1:
        max_length = max(sequence_lengths)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return (sequence_lengths.unsqueeze(1) >= range_tensor).long()
