import os

import numpy as np
import tensorflow as tf
from PIL import Image

from model.utils.general import Config
from model.utils.image import greyscale
from model.utils.text import Vocab


def truncate_end(list_of_ids, id_end):
    """Removes the end of the list starting from the first id_end token"""
    list_trunc = []
    for idx in list_of_ids:
        if idx == id_end:
            break
        else:
            list_trunc.append(idx)

    return list_trunc


def interactive_shell(sess):
    """Creates interactive shell to play with model
    """
    while True:
        try:
            # for python 2
            img_path = raw_input("input> ")
        except NameError:
            # for python 3
            img_path = input("input> ")

        if img_path == "exit":
            break

        if img_path[-3:] == "png":
            img = Image.open(img_path)
            # img = img.resize((80, 100))
            img = img.resize((80, 100), Image.BILINEAR)
            img.show()
            img = np.array(img)

            # crop_image(img_path, tmp_img)
            # pad_image(tmp_img, tmp_img, buckets=None)
            # downsample_image(tmp_img, tmp_img, 2)
            # img = imread(tmp_img)

        img = greyscale(img)
        ids_eval, = sess.run(['transpose_1:0'], feed_dict={'img:0': [img]})
        # ids_eval, = sess.run(['transpose_1:0'], feed_dict={'img:0': [img], 'dropout:0': 1})
        p = truncate_end(ids_eval[0], vocab.id_end)
        p = " ".join([vocab.id_to_tok[idx] for idx in p])

        print p


if __name__ == "__main__":
    # restore config and model
    # dir_output = "results/small/"
    dir_output = "results/hand_small_rnn_cnn/"
    config_vocab = Config(dir_output + "vocab.json")
    config_model = Config(dir_output + "model.json")
    vocab = Vocab(config_vocab)

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], os.path.join(dir_output, 'saved'))
        graph = tf.get_default_graph()

        # sess.run('myOutput:0', feed_dict={'myInput:0': None})
        interactive_shell(sess)
