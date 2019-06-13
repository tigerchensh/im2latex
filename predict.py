import os

import numpy
from PIL import Image
from scipy.misc import imread

from model.img2seq import Img2SeqModel
from model.utils.general import Config, run
from model.utils.image import greyscale, crop_image, pad_image, \
    downsample_image, TIMEOUT
from model.utils.text import Vocab


def interactive_shell(model):
    """Creates interactive shell to play with model
    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
Enter a path to a file
input> data/images_test/0.png""")

    while True:
        try:
            # for python 2
            img_path = raw_input("input> ")
        except NameError:
            # for python 3
            img_path = input("input> ")

        if img_path == "exit":
            break

        dir_output = "/tmp/"

        if img_path[-3:] == "png":
            img = Image.open(img_path)
            img = img.resize((80, 100), Image.BILINEAR)
            img.show()
            img = numpy.array(img)

            # crop_image(img_path, tmp_img)
            # pad_image(tmp_img, tmp_img, buckets=None)
            # downsample_image(tmp_img, tmp_img, 2)
            # img = imread(tmp_img)

        elif img_path[-3:] == "pdf":
            # call magick to convert the pdf into a png file
            buckets = [
                [240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100],
                [560, 80], [560, 100], [640, 80], [640, 100], [720, 80], [720, 100],
                [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200],
                [1000, 400], [1200, 200], [1600, 200], [1600, 1600]
            ]

            name = img_path.split('/')[-1].split('.')[0]
            run("magick convert -density {} -quality {} {} {}".format(200, 100,
                                                                      img_path, dir_output + "{}.png".format(name)),
                TIMEOUT)
            img_path = dir_output + "{}.png".format(name)
            crop_image(img_path, img_path)
            pad_image(img_path, img_path, buckets=buckets)
            downsample_image(img_path, img_path, 2)

            img = imread(img_path)

        img = greyscale(img)
        hyps = model.predict(img)

        model.logger.info(hyps[0])


if __name__ == "__main__":
    # restore config and model
    # dir_output = "results/small/"
    dir_output = "results/hand/"
    config_vocab = Config(dir_output + "vocab.json")
    config_model = Config(dir_output + "model.json")
    vocab = Vocab(config_vocab)

    model = Img2SeqModel(config_model, dir_output, vocab)
    model.build_pred()
    model.restore_session(dir_output + "model.weights/")

    interactive_shell(model)
