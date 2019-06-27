import click
import tensorflow as tf
import tensorflow.contrib.slim as slim

from model.img2seq_hand import Img2SeqModel
from model.utils.general import Config
from model.utils.text import Vocab


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


@click.command()
@click.option('--results', default="results/hand/", help='Dir to results')
def main(results):
    dir_output = results

    config_vocab = Config(dir_output + "vocab.json")
    config_model = Config(dir_output + "model.json")
    vocab = Vocab(config_vocab)

    model = Img2SeqModel(config_model, dir_output, vocab)
    model.build_pred()

    model_summary()


if __name__ == "__main__":
    main()
