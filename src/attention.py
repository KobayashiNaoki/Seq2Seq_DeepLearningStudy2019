import torch.nn.functional as F
import torch
import torch.nn as nn


def build_attention(hidden_dim, attention_type):
    if attention_type == 'mlp':
        return MLPAttention(hidden_dim)
    elif attention_type == 'general':
        return GeneralAttention(hidden_dim)
    elif attention_type == 'dot':
        return DotAttention(hidden_dim)

    exit()


class AttentionBase(nn.Module):
    def __init__(self, dim):
        super(AttentionBase, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)

    def score(self, decoder_hidden, memory_bank):
        raise NotImplementedError

    def forward(self, decoder_hidden, memory_bank, memory_mask=None):
        if decoder_hidden.dim() == 2:
            one_step = True
            decoder_hidden = decoder_hidden.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = decoder_hidden.size()

        align = self.score(decoder_hidden, memory_bank)

        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(memory_mask == 0, -float('inf'))

        align_vectors = F.softmax(align.view(batch * target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        context_vector = torch.bmm(align_vectors, memory_bank)

        concat_c = torch.cat(
            [context_vector, decoder_hidden], 2).view(batch*target_l, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

        return attn_h, align_vectors


class MLPAttention(AttentionBase):
    def __init__(self, dim):
        super(MLPAttention, self).__init__(dim)
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.linear_query = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, 1, bias=False)

    def score(self, h_t, h_s):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        dim = self.dim
        wq = self.linear_query(h_t.view(-1, dim))
        wq = wq.view(tgt_batch, tgt_len, 1, dim)
        wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

        uh = self.linear_context(h_s.contiguous().view(-1, dim))
        uh = uh.view(src_batch, 1, src_len, dim)
        uh = uh.expand(src_batch, tgt_len, src_len, dim)

        wquh = torch.tanh(wq + uh)

        return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)


class GeneralAttention(AttentionBase):
    def __init__(self, dim):
        super(GeneralAttention, self).__init__(dim)
        self.dim = dim
        self.linear_in = nn.Linear(dim, dim, bias=False)

    def score(self, h_t, h_s):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s_)


class DotAttention(AttentionBase):
    def __init__(self, dim):
        super(DotAttention, self).__init__(dim)
        self.dim = dim

    def score(self, h_t, h_s):
        h_s_ = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s_)
