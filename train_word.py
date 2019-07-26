import os

import click

from model.img2seq_hand import Img2SeqModel
from model.utils.general import Config
from model.utils.word_data_generator import DataGenerator
from model.utils.lr_schedule import LRSchedule
from model.utils.text import Vocab


@click.command()
@click.option('--data', default="configs/data_hand.json",
              help='Path to data json config')
@click.option('--vocab', default="configs/vocab_hand.json",
              help='Path to vocab json config')
@click.option('--training', default="configs/training_hand.json",
              help='Path to training json config')
@click.option('--model', default="configs/model.json",
              help='Path to model json config')
@click.option('--output', default="results/hand/",
              help='Dir for results and model weights')
def main(data, vocab, training, model, output):
    # Load configs
    dir_output = output
    config = Config([data, vocab, training, model])
    config.save(dir_output)
    vocab = Vocab(config)

    # Load datasets
    train_set = DataGenerator(
        index_file=config.index_train,
        path_formulas=config.path_formulas_train,
        dir_images=config.dir_images_train,
        max_iter=config.max_iter,
        path_matching=config.path_matching_train,
        max_len=config.max_length_formula,
        form_prepro=vocab.form_prepro)
    val_set = DataGenerator(
        index_file=config.index_val,
        path_formulas=config.path_formulas_val,
        dir_images=config.dir_images_val,
        max_iter=config.max_iter,
        path_matching=config.path_matching_val,
        max_len=config.max_length_formula,
        form_prepro=vocab.form_prepro)

    # Define learning rate schedule
    n_batches_epoch = ((len(train_set) + config.batch_size - 1) //
                       config.batch_size)

    print len(train_set)
    print config.batch_size
    print n_batches_epoch

    lr_schedule = LRSchedule(lr_init=config.lr_init,
                             start_decay=config.start_decay * n_batches_epoch,
                             end_decay=config.end_decay * n_batches_epoch,
                             end_warm=config.end_warm * n_batches_epoch,
                             lr_warm=config.lr_warm,
                             lr_min=config.lr_min)

    transfer_model = config.transfer_model

    # Build model and train
    model = Img2SeqModel(config, dir_output, vocab)
    model.build_train(config)
    if transfer_model and os.path.isdir(transfer_model):
        model.restore_session(transfer_model)

    model.train(config, train_set, val_set, lr_schedule)


if __name__ == "__main__":
    main()
