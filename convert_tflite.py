import os

import click

import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

from model.img2seq_hand import Img2SeqModel
from model.utils.general import Config
from model.utils.text import Vocab


@click.command()
@click.option('--results', default="results/hand/", help='Dir to results')
def main(results):
    # restore config and model
    dir_output = results
    saved_path = os.path.join(dir_output, 'saved')

    config_data = Config(dir_output + "data.json")
    config_vocab = Config(dir_output + "vocab.json")
    config_model = Config(dir_output + "model.json")

    vocab = Vocab(config_vocab)
    model = Img2SeqModel(config_model, dir_output, vocab)
    model.build_pred()
    model.restore_session(dir_output + "model.weights/")

    model.save_savedmodel(saved_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    main()
