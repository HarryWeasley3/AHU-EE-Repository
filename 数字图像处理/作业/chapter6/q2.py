import os
from PIL import Image
import numpy as np


def read_image_loc(which_image):
    '''reads the location of input image'''
    image_loc = os.path.normpath(
        input(f'Enter the location of {which_image} image: '))
    return image_loc


def load_pixels(image_loc):
    '''opens the image and returns its pixels'''
    img = Image.open(image_loc)
    px = img.load()
    return px


def get_dimensions(image_loc):
    return Image.open(image_loc).size


def produce_output(fg_px, bg_px, output_loc, w, h, threshold):
    '''produce output image'''

    output = []
    for x in range(h):
        output.append([])
        for y in range(w):

            g_pixel = fg_px[y, x][1]

            if g_pixel >= threshold:
                output[x].append(bg_px[y, x])

            else:
                output[x].append(fg_px[y, x])

    array = np.array(output, dtype=np.uint8)
    new_image = Image.fromarray(array)
    new_image.save(output_loc)


def main():

    fg_image = read_image_loc('foreground')
    bg_image = read_image_loc('background')

    fg_px = load_pixels(fg_image)
    bg_px = load_pixels(bg_image)

    w, h = get_dimensions(fg_image)

    output_loc = os.path.dirname(fg_image) + '/output.png'
    # set threshold as per requirement
    threshold = 235

    produce_output(fg_px, bg_px, output_loc, w, h, threshold)
    print('Saved ' + output_loc)


main()
