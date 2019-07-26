import os

import pymongo
import requests
from tqdm import tqdm

client = pymongo.MongoClient('mongodb://192.168.2.111', 27017)

db = client.littlelights
images = db.image

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_PATH = os.path.join(DIR_PATH, 'data/hand/')
DOWNLOAD_DIR = os.path.join(DIR_PATH, 'raw')
DOWNLOAD_Word_DIR = os.path.join(DIR_PATH, 'raw_word')
INDEX_FILE = os.path.join(DIR_PATH, 'index.txt')
INDEX_Word_FILE = os.path.join(DIR_PATH, 'index_word.txt')


def download(url, file_path):
    if os.path.exists(file_path):
        print 'Ignore file: %s' % file_path
        return
    print 'Downloading file: %s from url: %s' % (file_path, url)
    r = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(r.content)


def main():
    if not os.path.isdir(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    if not os.path.isdir(DOWNLOAD_Word_DIR):
        os.makedirs(DOWNLOAD_Word_DIR)

    img_list = images.find({"type": "handwritten"})
    print 'Downloading %d events...' % img_list.count()
    err_cnt = 0
    img_word_list = images.find({"type": "handwritten_word"})

    with open(INDEX_FILE, 'w') as f:
        for img in tqdm(img_list):
            url = img['img_storage_id']
            label = img['label']
            _id = img['_id']
            file_name = '{}.png'.format(_id)
            path = '{}/{}'.format(DOWNLOAD_DIR, file_name)
            download(url=url, file_path=path)
            f.write('{},{}'.format(file_name, label) + os.linesep)

    with open(INDEX_Word_FILE, 'w') as f:
        for img in tqdm(img_word_list):
            url = img['img_storage_id']
            label = img['label']
            _id = img['_id']
            file_name = '{}.png'.format(_id)
            path = '{}/{}'.format(DOWNLOAD_Word_DIR, file_name)
            download(url=url, file_path=path)
            f.write('{},{}'.format(file_name, label) + os.linesep)

    print 'Error: %d events...' % err_cnt


if __name__ == '__main__':
    main()
