import numpy as np
from PIL import Image
import tensorflow as tf

#
# load image 'filename' and return a tensorflow tensor and a dict image_info
# the tensor has the dimension [batch, height, width, (channel = 1 or 3). The pixel are coded in tf.float32 (0-1)
# the dict contains the following info: format (JPEG, PNG, TIFF, etc), mode (RGB, L, I, ...), the size (height, width)
# dataformat (tf.uint8, tf.uint16, tf.uint32) and channel (1 or 3)
#
def load_image(filename):

    # http://jlbicquelet.free.fr/scripts/python/pil/pil.php
    # https://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes

    Mode_Image = {
        '1':'(1-bit pixels,black and white, stored with one pixel perbyte)',
        'L':'(8-bit pixels, black and white)',
        'P':'(8-bit pixels, mapped to any other mode using a color palette)',
        'RGB':'(3x8-bit pixels, true color)',
        'RGBA':'(4x8-bit pixels, true color with transparency mask)',
        'CMYK':'(4x8-bit pixels, color separation)',
        'YCbCr':'(3x8-bit pixels, color video format)',
        'LAB':'(3x8-bit pixels, the L*a*b color space)',
        'HSV':'(3x8-bit pixels, Hue, Saturation, Value color space)',
        'I;16':'(16-bit signed integer pixels)',
        'I;32':'(16-bit signed integer pixels)',
        'F':'(32-bit floating point pixels)'
        }

    image_info = {
        'format': None,
        'mode': None,
        'dataformat': None,
        'size': None,
        'channel': None}

    raw_image = Image.open(filename)
    print("Mode of image: " + raw_image.mode + ": " + Mode_Image[raw_image.mode] + "\n"
          + "Format of Image: " + raw_image.format + "\n"
          + "Size of image: " + str(raw_image.size) + "\n")
    
    if not raw_image.mode in ('RGB','I;16','I;32','L'):
        print("Format d'image non supporté \n")
        exit()

    image_info['format'] = raw_image.format
    image_info['mode'] = raw_image.mode
    image_info['size'] = raw_image.size
    if raw_image.mode == 'RGB':
        image_info['dataformat'] = tf.uint8
        image_info['channel'] = 3
    elif raw_image.mode == 'L':
        image_info['dataformat'] = tf.uint8
        image_info['channel'] = 1
    elif raw_image.mode == 'I;16':
        image_info['dataformat'] = tf.uint16
        image_info['channel'] = 1
    else:
        image_info['dataformat'] = tf.uint32
        image_info['channel'] = 1

    np_array = np.array(raw_image) # https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html
    np_array_to_tf = tf.convert_to_tensor(np_array) # https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor

    converted = tf.image.convert_image_dtype(np_array_to_tf,tf.float32,saturate=True) # put all values from [0,MAX] to [0,1] over 32 bits
    # converted.set_shape((dim,dim,3))
    image_tf = tf.expand_dims(converted, 0)
    
    return image_tf,image_info

#
# save a numpy array (diw_h,dim_w,channel) into an image file as defined in image_info['format']
# after having converted the coding of each pixel.
#
def save_image(filename, image, image_info):
    
    # convertit le codage pixel (0 - 1) sur 32 bit en (0 - MAX) sur x bit
    converted_img = tf.image.convert_image_dtype(
        image,
        image_info['dataformat'],
        saturate=True)

    pil_image = Image.fromarray(converted_img.numpy())
    pil_image.save(filename)