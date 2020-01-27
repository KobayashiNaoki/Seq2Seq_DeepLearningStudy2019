import torch.nn as nn


class Embedder(nn.Module):
    def __init__(self, field, embed_dim, dropout_p):
        super(Embedder, self).__init__()
        self._field = field
        self._vocab = field.vocab
        self._embed_dim = embed_dim
        self._embed = nn.Embedding(len(self._vocab), self._embed_dim)
        self._dropout = nn.Dropout(dropout_p)

    def forward(self, inputs):
        embeddings = self._embed(inputs)
        return self._dropout(embeddings)

    def get_embed_dim(self):
        return self._embed_dim

    def get_vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return self._vocab

    def get_init_token_idx(self):
        init_token = self._field.init_token
        init_token_idx = self._vocab.stoi[init_token]
        return init_token_idx

    def get_eos_token_idx(self):
        eos_token = self._field.eos_token
        eos_token_idx = self._vocab.stoi[eos_token]
        return eos_token_idx

    def get_pad_token_idx(self):
        pad_token = self._field.pad_token
        pad_token_idx = self._vocab.stoi[pad_token]
        return pad_token_idx
