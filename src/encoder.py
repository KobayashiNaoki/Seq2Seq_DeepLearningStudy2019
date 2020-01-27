import torch.nn as nn
from lstm import PackedLSTM


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        pass

    def forward(self):
        pass


class LstmEncoder(Encoder):
    def __init__(self,
                 embedder,
                 hidden_dim,
                 num_layers,
                 dropout_p):
        super(LstmEncoder, self).__init__()
        self._embedder = embedder
        embed_dim = self._embedder.get_embed_dim()
        self._lstm = PackedLSTM(
            embed_dim, hidden_dim, num_layers, dropout_p)

    def forward(self,
                inputs,
                lengths):

        embeddings = self._embedder(inputs)
        outputs, (hidden, cell) = self._lstm(embeddings, lengths)
        # , is_sorted=True)

        return {
            "memory_bank": outputs,
            "hidden": hidden,
            "cell": cell,
        }
