import torch
import torch.nn as nn


class PackedLSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p):
        super(PackedLSTM, self).__init__(
            input_size, hidden_size, num_layers=num_layers, bias=True,
            batch_first=True, dropout=dropout_p, bidirectional=True)
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bias = True
        self._bidirectional = True
        self._encode = super(PackedLSTM, self).forward

    def forward(self, inputs, lengths, is_sorted=False):
        if not is_sorted:
            inputs, lengths, restoration_indices = self.sorting(inputs, lengths)

        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths, batch_first=True)

        packed_outputs, (ht, ct) = self._encode(packed_inputs)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True)

        if not is_sorted:
            outputs = outputs.index_select(0, restoration_indices)
            ct = ct.index_select(1, restoration_indices)
            ht = ht.index_select(1, restoration_indices)

        return outputs, (ht, ct)

    def sorting(self, inputs, lengths):
        sorted_lengths, permutation_index = lengths.sort(0, descending=True)
        sorted_inputs = inputs.index_select(0, permutation_index)

        index_range = torch.arange(0, len(lengths), device=lengths.device)

        _, reverse_mapping = permutation_index.sort(0, descending=False)
        restoration_indices = index_range.index_select(0, reverse_mapping)
        return sorted_inputs, sorted_lengths, restoration_indices


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input_feed, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input_feed, (h_1, c_1)
