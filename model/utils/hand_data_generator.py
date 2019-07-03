import os
from multiprocessing import BoundedSemaphore

import imgaug.augmenters as iaa
import numpy as np
import psutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .text import split_formula, load_formulas


class DataGeneratorFile(object):
    """Simple Generator of tuples (img_path, formula_id)"""

    def __init__(self, filename):
        """Inits Data Generator File

        Iterator that returns
            tuple (img_path, formula_id)

        Args:
            filename: (string of path to file)

        """
        self._filename = filename

    def __iter__(self):
        with open(self._filename) as f:
            for line in f:
                line = line.strip().split(' ')
                path_img, id_formula = line[0], line[1]
                yield path_img, id_formula


class DataGenerator(object):
    """Data Generator of tuple (image, formula)"""

    def __init__(self, index_file, path_formulas, dir_images, path_matching,
                 form_prepro=lambda s: split_formula(s), iter_mode="data",
                 use_aug=False,
                 img_prepro=lambda x: x, max_iter=None, max_len=None, ):
        """Initializes the DataGenerator

        Raw images should be under dir_images/raw
        Index file should be under dir_images
        Small train file will be generated under dir_images/small

        Args:
            path_formulas: (string) file of formulas.
            dir_images: (string) dir of images, contains jpg files.
            path_matching: (string) file of name_of_img, id_formula
            img_prepro: (lambda function) takes an array -> an array. Default,
                identity
            form_prepro: (lambda function) takes a string -> array of int32.
                Default, identity.
            max_iter: (int) maximum numbers of elements in the dataset
            max_len: (int) maximum length of a formula in the dataset
                if longer, not yielded.
            iter_mode: (string) "data", "full" to set the type returned by the
                generator
            bucket: (bool) decides if bucket the data by size of image
            bucket_size: (int)

        """
        self._index_file = os.path.join(dir_images, index_file)
        self._path_formulas = path_formulas
        self._path_matching = path_matching
        self._img_prepro = img_prepro
        self._form_prepro = form_prepro
        self._max_iter = max_iter
        self._max_len = max_len
        self._iter_mode = iter_mode

        self._raw_dir_images = os.path.join(dir_images, 'raw')
        self._dir_images = os.path.join(dir_images, 'small')
        if not os.path.exists(self._dir_images):
            os.mkdir(self._dir_images)

        self._length = None
        self._formulas = self._load_formulas(path_formulas)

        self._set_data_generator()
        self._use_aug = use_aug

    def _load_formulas(self, filename):
        """Loads txt file with formulas in a dict

        Args:
            filename: (string) path of formulas.

        Returns:
            dict: dict[idx] = one formula

        """
        formulas = {}
        if os.path.isfile(filename):
            formulas = load_formulas(filename)
        return formulas

    def _set_data_generator(self):
        """Sets iterable or generator of tuples (img_path, id of formula)"""
        self._data_generator = DataGeneratorFile(self._path_matching)

    def _get_raw_formula(self, formula_id):
        try:
            formula_raw = self._formulas[int(formula_id)]
        except KeyError:
            print("Tried to access id {} but only {} formulas".format(
                formula_id, len(self._formulas)))
            print("Possible fix: mismatch between matching file and formulas")
            raise KeyError

        return formula_raw

    def _process_instance(self, example):
        """From path and formula id, returns actual data

        Applies preprocessing to both image and formula

        Args:
            example: tuple (img_path, formula_ids)
                img_path: (string) path to image
                formula_id: (int) id of the formula

        Returns:
            img: depending on _img_prepro
            formula: depending on _form_prepro

        """
        img_path, formula_id = example

        # raw_img = Image.open(os.path.join(self._dir_images, img_path)).convert('L')
        # raw_img = raw_img.resize((80, 100), Image.BILINEAR)
        raw_img = Image.open(os.path.join(self._dir_images, img_path))
        img = np.array(raw_img)
        img = np.expand_dims(img, axis=2)

        img_shape = np.shape(img)
        area = img_shape[0] * img_shape[1]
        max_area = 400 * 160
        img = self._img_prepro(img)
        formula = self._form_prepro(self._get_raw_formula(formula_id))

        if self._iter_mode == "data":
            inst = (img, formula)
        elif self._iter_mode == "full":
            inst = (img, formula, img_path, formula_id)

        # filter on the formula length
        if self._max_len is not None and len(formula) > self._max_len:
            skip = True
        else:
            skip = False

        if area > max_area:
            skip = True

        return inst, skip

    def __iter__(self):
        """Iterator over Dataset

        Yields:
            tuple (img, formula)

        """
        n_iter = 0
        for example in self._data_generator:
            if self._max_iter is not None and n_iter >= self._max_iter:
                break
            result, skip = self._process_instance(example)
            if skip:
                continue
            n_iter += 1
            yield result

    def __len__(self):
        if self._length is None:
            print("First call to len(dataset) - may take a while.")
            counter = 0
            for _ in self:
                counter += 1
            self._length = counter
            print("- done.")

        return self._length

    def resize_and_save_img(self, raw_img, path):
        img = raw_img.resize((80, 100), Image.BILINEAR)
        # img = raw_img.resize((80, 100))
        img.save(os.path.join(self._dir_images, path))

    def build(self):
        #  read index file and generate path_matching and formulas files.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        formulas = []
        cpt_cnt = psutil.cpu_count()
        executor = BoundedExecutor(cpt_cnt, cpt_cnt)
        aug_times = 20
        with open(self._index_file) as f:
            for line in tqdm(f):
                def process(l):
                    l = l.strip()
                    if not l:
                        return
                    img_path, formula = l.split(',')
                    formula = ' '.join(list(formula))
                    # check img path
                    if not os.path.isfile(os.path.join(self._raw_dir_images, img_path)):
                        return

                    raw_img = Image.open(os.path.join(self._raw_dir_images, img_path)).convert('L')
                    self.resize_and_save_img(raw_img, img_path)
                    formulas.append((formula, img_path))

                    if self._use_aug:
                        seq = iaa.Sequential([
                            iaa.Invert(p=1),
                            sometimes(iaa.Affine(rotate=(-25, 25))),
                            sometimes(iaa.Affine(shear=(-25, 25))),
                            sometimes(iaa.Affine(scale=(0.5, 1.3))),
                            sometimes(iaa.ElasticTransformation(alpha=10.0, sigma=5.0)),
                            sometimes(
                                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})),
                            sometimes(iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                            iaa.Invert(p=1),
                        ])
                        img = np.array(raw_img)
                        # seq.show_grid(img, cols=1, rows=1)

                        for i in xrange(aug_times):
                            image_aug = seq.augment_image(img)
                            image_aug = Image.fromarray(image_aug)
                            path = 'aug_{}_{}'.format(i, img_path)
                            self.resize_and_save_img(image_aug, path)
                            formulas.append((formula, path))

                executor.submit(process, line)

        with open(self._path_formulas, 'w') as formula_file, open(self._path_matching, 'w') as matching_file:
            for idx, (formula, img_path) in enumerate(formulas):
                self._formulas[idx] = formula
                formula_file.write('{}\n'.format(formula))
                matching_file.write('{} {}\n'.format(img_path, idx))


class BoundedExecutor:
    """BoundedExecutor behaves as a ThreadPoolExecutor which will block on
    calls to submit() once the limit given as "bound" work items are queued for
    execution.
    :param bound: Integer - the maximum number of items in the work queue
    :param max_workers: Integer - the size of the thread pool
    """

    def __init__(self, bound, max_workers):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = BoundedSemaphore(bound + max_workers)

    """See concurrent.futures.Executor#submit"""

    def submit(self, fn, *args, **kwargs):
        self.semaphore.acquire()
        try:
            future = self.executor.submit(fn, *args, **kwargs)
        except:
            self.semaphore.release()
            raise
        else:
            future.add_done_callback(lambda x: self.semaphore.release())
            return future

    """See concurrent.futures.Executor#shutdown"""

    def shutdown(self, wait=True):
        self.executor.shutdown(wait)
