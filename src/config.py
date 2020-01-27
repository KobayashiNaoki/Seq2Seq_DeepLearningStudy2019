import argparse
from pathlib import Path


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument('--train', '-train',
                   action='store_true')
    p.add_argument('--translate', '-translate',
                   action='store_true')
    p.add_argument('--data-root', '-data_root',
                   default='data/', type=Path)
    p.add_argument('--train-file', '-train_file',
                   default='train.tsv')
    p.add_argument('--valid-file', '-valid_file',
                   default='valid.tsv')
    p.add_argument('--test-file', '-test_file',
                   default='test.tsv')

    p.add_argument('--serialization-dir', '-serialization_dir',
                   default='model')
    p.add_argument('--keep-all-serialized-models', '-keep_all_serialized_models',
                   action="store_true")
    p.add_argument('--log-file', '-log_file', default='training.log')
    p.add_argument('--model-path', '-model_path')
    p.add_argument('--translation-file', '-translation_file',
                   default='translate.txt')

    p.add_argument('--batch-size', '-batch_size',
                   default=128, type=int)
    p.add_argument('--epochs', '-epochs',
                   default=20, type=int)
    p.add_argument('--gpu', '-gpu',
                   default=0, type=int)
    p.add_argument('--lr', '-lr',
                   default='1.0', type=float)
    p.add_argument('--lr-decay', '-lr_decay',
                   default=0.7, type=float)
    p.add_argument('--start-decay-epoch', '-start_decay_epoch',
                   default=13, type=int)
    p.add_argument('--grad-norm', '-grad_norm',
                   default=5.0, type=float)

    # p.add_argument('--src-hidden-dim', '-src_hidden_dim',
    #                default=500, type=int)
    # p.add_argument('--tgt-hidden-dim', '-tgt_hidden_dim',
    #                default=1000, type=int)
    p.add_argument('--hidden-dim', '-tgt_hidden_dim',
                   default=1000, type=int)
    p.add_argument('--embed-dim', '-embed_dim',
                   default=1000, type=int)
    p.add_argument('--num-layers', '-num_layers',
                   default=2, type=int)
    p.add_argument('--dropout-p', '-dropout_p',
                   default=0.3, type=float)
    p.add_argument('--attention', '-attention',
                   default='general', choices=['mlp', 'general', 'dot'])

    p.add_argument('--beam-width', '-beam_width',
                   default=20, type=int)
    # p.add_argument('', '')
    # p.add_argument('', '')
    return p.parse_args()
