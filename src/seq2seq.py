import torch
import torch.nn as nn
import torch.cuda as cutorch
import utils
from embedder import Embedder
from encoder import LstmEncoder
from decoder import LstmDecoder
from attention import build_attention


def build_model(config, src_field, tgt_field):
    embed_dim = config.embed_dim
    hidden_dim = config.hidden_dim
    num_layers = config.num_layers
    dropout_p = config.dropout_p
    attention_type = config.attention
    beam_width = config.beam_width

    model = Seq2Seq(
        LstmEncoder(
            Embedder(
                src_field,
                embed_dim,
                dropout_p
            ),
            hidden_dim,
            num_layers,
            dropout_p),
        LstmDecoder(
            Embedder(
                tgt_field,
                embed_dim,
                dropout_p
            ),
            hidden_dim,
            num_layers,
            dropout_p,
            build_attention(
                hidden_dim,
                attention_type
            ),
            beam_width
        ),
    )
    return model


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder):
        super(Seq2Seq, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, source_tokens, target_tokens=None, beam=False):
        state = self.encode(source_tokens)
        state = self.bridge(state)
        if beam:
            output = self.beam_decode(state)
        else:
            output = self.decode(state, target_tokens)
        return output

    def encode(self, source_tokens):
        inputs = source_tokens[0]
        lengths = source_tokens[1]
        source_mask = utils.make_mask_by_lengths(lengths)
        outputs = self._encoder(inputs, lengths)
        return {
            "source_mask": source_mask,
            "encoder_memory_bank": outputs["memory_bank"],
            "encoder_hidden": outputs["hidden"],
            "encoder_cell": outputs["cell"]
        }

    def bridge(self, state):
        hidden = state["encoder_hidden"]
        LL, batch_size, hidden_dim = hidden.size()
        backward_hidden = hidden.view(2, -1, batch_size, hidden_dim)[1]
        state["encoder_hidden"] = backward_hidden

        memory_bank = state["encoder_memory_bank"]
        batch_size, L, _ = memory_bank.size()
        memory_bank = torch.sum(
            memory_bank.view(batch_size, L, 2, hidden_dim), dim=2)
        state["encoder_memory_bank"] = memory_bank

        return state

    def decode(self, state, target_tokens):
        state.update(self._decoder.init_state(state))
        decoder_outputs = self._decoder(state, target_tokens)
        return decoder_outputs

    def beam_decode(self, state):
        state.update(self._decoder.init_state(state))
        decoder_outputs = self._decoder._forward_beam_search(state)
        return decoder_outputs
