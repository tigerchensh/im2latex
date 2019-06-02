import os

import imgaug.augmenters as iaa
import imgaug as ia
import cv2

PATH = '/home/xiao/code/im2latex/data/small'


def main():
    imgs = []
    for f in os.listdir(PATH):
        if not f.endswith('png'):
            continue
        im = cv2.imread(os.path.join(PATH, f))
        imgs.append(im)

    seq = iaa.Sequential([
        iaa.Invert(p=1),
        iaa.Affine(rotate=(-5, 5), scale=(0.5, 1.1)),
        iaa.ElasticTransformation(alpha=10.0, sigma=5.0),
        # iaa.SaltAndPepper(p=0.2),
        iaa.Invert(p=1),
    ])
    seq.show_grid(imgs[:1], cols=1, rows=1)
    seq.show_grid(imgs[2:], cols=1, rows=1)
    # images_aug = seq.augment_images(imgs)
    # seq.show_grid(images_aug, cols=1, rows=1)


if __name__ == '__main__':
    main()
