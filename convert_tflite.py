import datetime
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
    weights_dir = os.path.join(dir_output, 'model.weights/')

    t = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    # saved_path = 'saved_' + t
    saved_path = 'saved' + t
    saved_path = os.path.join(dir_output, saved_path)

    config_data = Config(dir_output + "data.json")
    config_vocab = Config(dir_output + "vocab.json")
    config_model = Config(dir_output + "model.json")
    vocab = Vocab(config_vocab)

    if not os.path.isdir(saved_path):
        model = Img2SeqModel(config_model, dir_output, vocab)
        model.build_pred()
        model.restore_session(weights_dir)

        model.save_savedmodel(saved_path)

    # chkp.print_tensors_in_checkpoint_file(weights_dir, tensor_name='', all_tensors=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
    converter.target_ops = [
        # tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    main()
