import click

from model.utils.general import Config
from model.utils.word_data_generator import DataGenerator
from model.utils.text import build_vocab, write_vocab


@click.command()
@click.option('--data', default="configs/data_hand.json",
              help='Path to data json config')
@click.option('--vocab', default="configs/vocab_hand.json",
              help='Path to vocab json config')
def main(data, vocab):
    data_config = Config(data)

    # datasets
    train_set = DataGenerator(
        index_file=data_config.index_train,
        path_formulas=data_config.path_formulas_train,
        dir_images=data_config.dir_images_train,
        path_matching=data_config.path_matching_train, use_aug=True)
    test_set = DataGenerator(
        index_file=data_config.index_test,
        path_formulas=data_config.path_formulas_test,
        dir_images=data_config.dir_images_train,
        path_matching=data_config.path_matching_test)
    val_set = DataGenerator(
        index_file=data_config.index_val,
        path_formulas=data_config.path_formulas_val,
        dir_images=data_config.dir_images_train,
        path_matching=data_config.path_matching_val)

    # produce images and matching files
    # train_set.build(buckets=None, n_threads=1)
    test_set.build()
    val_set.build()
    train_set.build()
    # train_set.build(buckets=data_config.buckets)
    # test_set.build(buckets=data_config.buckets)
    # val_set.build(buckets=data_config.buckets)

    # p = Augmentor.Pipeline(data_config.dir_images_train)
    # p.zoom(probability=1, min_factor=1.1, max_factor=1.5)
    # # p.process()
    # augmented_images, labels = p.sample(3)
    # print labels

    # vocab
    vocab_config = Config(vocab)
    vocab = build_vocab([train_set], min_count=vocab_config.min_count_tok)
    write_vocab(vocab, vocab_config.path_vocab)


if __name__ == "__main__":
    main()
