from torchtext import data


def seq2seq_fields():
    SRC = data.Field(batch_first=True, include_lengths=True)
    TGT = data.Field(batch_first=True, include_lengths=True, is_target=True,
                     init_token='<S>', eos_token='</S>')
    fields = [('source_tokens', SRC), ('target_tokens', TGT)]

    return fields
