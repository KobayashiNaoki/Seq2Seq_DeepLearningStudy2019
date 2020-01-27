import torch
from tqdm import tqdm


class Translator():
    def __init__(self, config, fields, model, test_iter):
        self._config = config
        self._fields = fields
        self._model = model
        self._test_iter = test_iter
        self._translation_file = config.translation_file

        # src_field = self._fields[0][1]  # SRC field object
        tgt_field = self._fields[1][1]  # SRC field object
        self._tgt_vocab = tgt_field.vocab
        self._eos_token_idx = self._tgt_vocab.stoi[tgt_field.eos_token]

    def translate(self):
        model = self._model
        translations = []
        model.eval()
        for batch in tqdm(self._test_iter):
            with torch.no_grad():
                output_dict = model(batch.source_tokens, beam=True)
            predictions = output_dict["predictions"]
            predictions = self.delabeling(predictions, beam=True)
            translations.extend(predictions)

        with open(self._translation_file, "w") as f:
            text = '\n'.join([' '.join(sentence) for sentence in translations])
            print(text, file=f)

        return

    def delabeling(self, batch_labels, beam):
        tgt_vocab = self._tgt_vocab
        batch_tokens = []
        for labels in batch_labels.tolist():
            tokens = []
            if beam:
                labels = labels[0]
            for label in labels:
                if label == self._eos_token_idx:
                    break
                token = tgt_vocab.itos[label]
                tokens.append(token)

            batch_tokens.append(tokens)

        return batch_tokens
