import torch
import torch.optim as optim
from fields import seq2seq_fields
from data_loader import DatasetReader
from checkpointer import Checkpointer
from trainer import Trainer
from translator import Translator
from config import get_config
from seq2seq import build_model


def main():
    config = get_config()
    if config.train:
        train(config)
    elif config.translate:
        translate(config)

    return 0


def train(config):
    fields = seq2seq_fields()
    datareader = DatasetReader(config, fields)
    train_iter, valid_iter = datareader.prepare_for_train()
    src_field = fields[0][1]  # SRC field object
    tgt_field = fields[1][1]  # SRC field object

    model = build_model(config, src_field, tgt_field)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    trainer = Trainer(config, model, optimizer, scheduler,
                      train_iter, valid_iter, fields)
    trainer.run()

    test_iter = datareader.prepare_for_test()
    translator = Translator(config, fields, model, test_iter)
    translator.translate()
    return


def translate(config):

    model_path = config.model_path

    if config.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    model_state = Checkpointer.restore(model_path, device)

    fields = model_state["fields"]
    src_field = fields[0][1]  # SRC field object
    tgt_field = fields[1][1]  # SRC field object
    datareader = DatasetReader(config, fields)
    test_iter = datareader.prepare_for_test()

    model_config = model_state["config"]
    model_config.beam_width = config.beam_width
    model = build_model(model_config, src_field, tgt_field)
    model.to(device)
    model.load_state_dict(model_state["model"])

    translator = Translator(config, fields, model, test_iter)
    translator.translate()


if __name__ == '__main__':
    main()
