import os
from tqdm import tqdm
from PIL import Image
import re


def main():
    #  read index file and generate path_matching and formulas files.
    raw_dir = '/home/littlelight/Downloads/word'
    file = open('index_word.txt', "r")
    file2 = open("index_new.txt", "w")
    for line in tqdm(file):
        list = line.split()
        img_index = list[0]
        sucess = list[1]
        formula = list[len(list) - 1]
        regex = re.compile('[@_!#$%^&*()<>?/\|}{~:+-0123456789;]')
        if (regex.search(formula) == None) and sucess == 'ok' and ',' not in formula and '.' not in formula \
                and '"' not in formula and "'" not in formula:
            par, sub, index1, index2 = img_index.split('-')
            real_path = os.path.join(raw_dir, par, par + '-' + sub, img_index + '.png')
            if os.path.isfile(real_path):
                raw_img = Image.open(real_path).convert('L')
                resize_and_save_img(raw_img, img_index + '.png')
                file2.write(img_index + '.png' + ',' + formula + '\n')
    file.close()
    file2.close()


def resize_and_save_img(raw_img, raw_dir):
    image_size = raw_img.size
    width = image_size[0]
    height = image_size[1]
    # bigside = width if width > height else height
    # bigside = bigside * 2
    bigwidth = width*2
    bigheight = height*4
    background = Image.new('RGBA', (bigwidth, bigheight), (255, 255, 255, 255))
    offset = (int(round(((bigwidth - width) / 2), 0)), int(round(((bigheight - height) / 2), 0)))
    background.paste(raw_img, offset)
    background = background.resize((80, 100), Image.BILINEAR)
    background.save(os.path.join('/home/littlelight/im2latex/tools/data/hand/raw_word', raw_dir))


if __name__ == '__main__':
    main()
