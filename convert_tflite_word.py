import datetime
import os

import click
import numpy as np
import tensorflow as tf
from PIL import Image

from model.img2seq_hand import Img2SeqModel
from model.utils.general import Config
from model.utils.text import Vocab

@click.command()
@click.option('--results', default="results/hand_word/", help='Dir to results')
def main(results):
    tf.enable_resource_variables()
    # restore config and model
    dir_output = results
    weights_dir = os.path.join(dir_output, 'model.weights/')

    t = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    # saved_path = 'saved_' + t
    saved_path = 'saved_word'
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

    SAMPLE_DIR = 'tools/data/hand/raw_word'

    def representative_dataset_gen():
        num_calibration_steps = 10

        if not os.path.isdir(SAMPLE_DIR):
            print 'Failed to read representative_dataset'
            return

        for f in os.listdir(SAMPLE_DIR):
            img_path = os.path.join(SAMPLE_DIR, f)
            img = Image.open(img_path)
            img = img.resize((80, 100), Image.BILINEAR)
            img.show()
            img = np.array(img)
            yield [img]

            num_calibration_steps -= 1
            if num_calibration_steps == 0:
                break

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
    converter.target_ops = [
        # tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS]

    # Following has "Segmentation fault"
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()
    open("converted_model_word.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    main()
