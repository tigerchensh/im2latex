import os

import pymongo
import requests
from tqdm import tqdm

client = pymongo.MongoClient('localhost', 27017)

db = client.littlelights
images = db.image

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DOWNLOAD_DIR = os.path.join(DIR_PATH, 'data/hand')
INDEX_FILE = os.path.join(DOWNLOAD_DIR, 'index.txt')


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

    img_list = images.find({})
    print 'Downloading %d events...' % img_list.count()
    err_cnt = 0

    with open(INDEX_FILE, 'w') as f:
        for img in tqdm(img_list):
            url = img['img_storage_id']
            label = img['label']
            _id = img['_id']
            file_name = '{}.png'.format(_id)
            path = '{}/{}'.format(DOWNLOAD_DIR, file_name)
            download(url=url, file_path=path)
            f.write('{},{}'.format(file_name, label) + os.linesep)

    print 'Error: %d events...' % err_cnt


if __name__ == '__main__':
    main()
