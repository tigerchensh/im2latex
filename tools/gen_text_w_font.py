import os
import random

from PIL import Image, ImageFont, ImageDraw

dir_path = os.path.dirname(os.path.realpath(__file__))

font_dir = os.path.join(dir_path, 'fonts')
zoom_factor = 2

fonts = []
path = os.path.join(font_dir, 'Almost Cartoon.ttf')
font = ImageFont.truetype(path, size=18 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'AckiPreschool.ttf')
font = ImageFont.truetype(path, size=18 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'BPchildfatty.ttf')
font = ImageFont.truetype(path, size=18 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Children.ttf')
font = ImageFont.truetype(path, size=16 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Children Sans.ttf')
font = ImageFont.truetype(path, size=16 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'ComingSoon.ttf')
font = ImageFont.truetype(path, size=18 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'doves.ttf')
font = ImageFont.truetype(path, size=18 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'elizajane.ttf')
font = ImageFont.truetype(path, size=12 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Gruenewald VA normal.ttf')
font = ImageFont.truetype(path, size=18 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'KatetheGreat.ttf')
font = ImageFont.truetype(path, size=14 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Kindergarden.ttf')
font = ImageFont.truetype(path, size=14 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'PizzaismyFAVORITE.ttf')
font = ImageFont.truetype(path, size=14 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Schoolbell.ttf')
font = ImageFont.truetype(path, size=18 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Anke Print.ttf')
font = ImageFont.truetype(path, size=18 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'blzee.ttf')
font = ImageFont.truetype(path, size=18 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'BrownBagLunch.ttf')
font = ImageFont.truetype(path, size=24 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'CATHSGBR.ttf')
font = ImageFont.truetype(path, size=20 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'dadha___.ttf')
font = ImageFont.truetype(path, size=20 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Domestic_Manners.ttf')
font = ImageFont.truetype(path, size=18 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'emizfont.ttf')
font = ImageFont.truetype(path, size=18 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Hurryup.ttf')
font = ImageFont.truetype(path, size=22 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'James Fajardo.ttf')
font = ImageFont.truetype(path, size=26 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'LadylikeBB.ttf')
font = ImageFont.truetype(path, size=25 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'PAINP___.ttf')
font = ImageFont.truetype(path, size=22 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Popsies.ttf')
font = ImageFont.truetype(path, size=20 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Qokijo.ttf')
font = ImageFont.truetype(path, size=20 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'rabiohead.ttf')
font = ImageFont.truetype(path, size=24 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'SANTO___.ttf')
font = ImageFont.truetype(path, size=24 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Snake.ttf')
font = ImageFont.truetype(path, size=24 * zoom_factor)
fonts.append(font)
path = os.path.join(font_dir, 'Suwa.ttf')
font = ImageFont.truetype(path, size=24 * zoom_factor)
fonts.append(font)

# TODO: careful
path = os.path.join(font_dir, 'Absinthe.ttf')
font = ImageFont.truetype(path, size=22 * zoom_factor)
fonts.append(font)
# TODO: careful
path = os.path.join(font_dir, 'Spitter.ttf')
font = ImageFont.truetype(path, size=36 * zoom_factor)
fonts.append(font)


def random_bg_color():
    f = random.randint
    # return f(200, 255), f(200, 255), f(200, 255)
    return 255, 255, 255


def random_fg_color():
    f = random.randint
    # return f(0, 100), f(0, 100), f(0, 100)
    return 0, 0, 0


def draw_img(text, path):
    image = Image.new("RGB", (100 * zoom_factor, 32 * zoom_factor), random_bg_color())
    draw = ImageDraw.Draw(image)
    font = random.choice(fonts)
    # arr = path.split('/')
    # path = '{}/{}_{}'.format('/'.join(arr[:-1]), font.getname()[0], arr[-1])
    x, y = random.randint(1, 5), random.randint(5, 40),
    x, y = 5, 5
    draw.text((x, y), text, fill=random_fg_color(), font=font)
    image.save(path)
