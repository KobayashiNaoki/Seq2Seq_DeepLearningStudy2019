import torch
import torch.nn as nn
import torch.nn.functional as F
from lstm import StackedLSTM
from beam_search import BeamSearch
import utils


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()


class LstmDecoder(Decoder):
    def __init__(self,
                 embedder,
                 hidden_dim,
                 num_layers,
                 dropout_p,
                 attention,
                 beam_width):
        super(LstmDecoder, self).__init__()
        self._embedder = embedder
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._attention = attention
        decoder_input_dim = self._embedder.get_embed_dim()
        decoder_input_dim += self._hidden_dim  # for input-feeding

        self._lstm = StackedLSTM(
            decoder_input_dim, self._hidden_dim, num_layers, dropout_p)

        self._output_projection_layer = nn.Linear(
            self._hidden_dim, self._embedder.get_vocab_size())

        self._start_index = self._embedder.get_init_token_idx()
        self._eos_index = self._embedder.get_eos_token_idx()
        self._pad_index = self._embedder.get_pad_token_idx()
        self._max_decoding_steps = 100

        self._beam_search = BeamSearch(self._eos_index,
                                       self._max_decoding_steps,
                                       beam_width)

    def init_state(self, state):
        hidden = state["encoder_hidden"]
        _, batch_size, _ = hidden.size()
        cell = hidden.new_zeros(self._num_layers, batch_size, self._hidden_dim)
        input_feed = hidden.new_zeros(batch_size, self._hidden_dim)
        return {
            "decoder_hidden": hidden,
            "decoder_cell": cell,
            "decoder_input_feed": input_feed
        }

    def forward(self, state, target_tokens=None):
        source_mask = state["source_mask"]
        batch_size, _ = source_mask.size()

        if target_tokens:
            targets = target_tokens[0]
            target_mask = utils.make_mask_by_lengths(target_tokens[1])
            _, target_sequence_length = targets.size()
            num_decoding_steps = target_sequence_length - 1  # ignore eos padding
        else:
            num_decoding_steps = self._max_decoding_steps

        last_predictions = source_mask.new_full(
            (batch_size,), fill_value=self._start_index)

        step_logits = []
        step_predictions = []
        for timestep in range(num_decoding_steps):
            if target_tokens is None:
                input_choices = last_predictions
            else:
                input_choices = targets[:, timestep]

            output_projections, state = self._forward_step(input_choices, state)
            step_logits.append(output_projections.unsqueeze(1))
            class_probabilities = F.softmax(output_projections, dim=-1)
            _, predicted_classes = torch.max(class_probabilities, 1)
            last_predictions = predicted_classes
            step_predictions.append(last_predictions.unsqueeze(1))

        predictions = torch.cat(step_predictions, 1)
        output_dict = {"predictions": predictions}

        if target_tokens:
            logits = torch.cat(step_logits, 1)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _get_loss(self, logits, targets, target_mask):
        relevant_targets = targets[:, 1:].contiguous()
        _, _, num_classes = logits.size()
        return F.cross_entropy(logits.view(-1, num_classes),
                               relevant_targets.view(-1),
                               ignore_index=self._pad_index)

    def _forward_step(self,
                      last_predictions,
                      state):
        encoder_memory_bank = state["encoder_memory_bank"]
        source_mask = state["source_mask"]
        decoder_hidden = state["decoder_hidden"]
        decoder_cell = state["decoder_cell"]

        last_predictions_embedding = self._embedder(last_predictions)

        input_feed = state["decoder_input_feed"]
        decoder_input = torch.cat(
            (input_feed, last_predictions_embedding), -1)

        decoder_output, (decoder_hidden, decoder_cell) = self._lstm(
            decoder_input, (decoder_hidden, decoder_cell)
        )
        state["decoder_hidden"] = decoder_hidden
        state["decoder_cell"] = decoder_cell

        decoder_output, _ = self._attention(
            decoder_output, encoder_memory_bank, source_mask)
        state["decoder_input_feed"] = decoder_output

        output_projections = self._output_projection_layer(decoder_output)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)
        return class_log_probabilities, state

    def _forward_beam_search(self, state):
        source_mask = state["source_mask"]
        batch_size, _ = source_mask.size()

        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index)

        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self._forward_step
        )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict
