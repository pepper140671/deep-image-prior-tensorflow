import tensorflow as tf
import numpy as np
import scipy.stats as st
import datetime as dt

# Name of image to upscale. Must be a dim_h x dim_v BMP.
input_folder = "Input/"
output_folder = "output/"
image_name = "M84_Lx52_ABE1_NL"
image_type = ".bmp"
dim_h = dim_w = 256

log = True # to log in text file all information
log_file = image_name + "_log.txt"

blur = False # valeur initiale True / to be reviewed if not used

# The number of down sampling and up sampling layers.
# These should be equal if the ouput and input images
# are to be equal.
down_layer_count = 5
channels_per_layer = [8, 16, 32, 64, 128] # size similar to layer_count 
up_layer_count = 5

channels_per_skip_layer = [0, 0, 0, 4, 4] # size similar to layer_count

nbre_iterations = 201 # valeur initiale 5001


def load_image(filename, dim_h,dim_w):
    with open(input_folder + image_name + image_type, 'rb') as f:
        # https://www.tensorflow.org/api_docs/python/tf/io/decode_image
        raw_image = tf.image.decode_bmp(f.read())

    converted = tf.image.convert_image_dtype(
        raw_image,
        tf.float32,
        saturate=True
    )
    
    resized = tf.image.resize_images(
        images = converted,
        size = [dim_h, dim_w]
    )

    resized.set_shape((dim,dim,3))

    blur = gblur(tf.expand_dims(resized, 0))

    return blur

def save_image(filename, image):
    converted_img = tf.image.convert_image_dtype(
        image,
        tf.uint8,
        saturate=True)

    encoded_img = tf.image.encode_png(converted_img)
    
    with open(filename, 'wb') as f:
        f.write(encoded_img.eval())

def down_layer(layer, channel):
    # https://stackoverflow.com/questions/50308951/understanding-input-output-tensors-from-tf-layers-conv2d
    layer = tf.contrib.layers.conv2d(
        inputs=layer,
        num_outputs=channel,
        kernel_size=3,
        stride=2,
        padding='SAME',
        activation_fn=tf.nn.leaky_relu) # remplacé None
    
#    layer = tf.contrib.layers.batch_norm(
#        inputs=layer,
#        activation_fn=tf.nn.leaky_relu)

    layer = tf.contrib.layers.conv2d(
        inputs=layer,
        num_outputs=channel,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.leaky_relu) # remplacé None
    
#    layer = tf.contrib.layers.batch_norm(
#        inputs=layer,
#        activation_fn=tf.nn.leaky_relu)
    
    return layer

def up_layer(layer, channel):
    layer = tf.contrib.layers.batch_norm(
        inputs=layer)
    
    layer = tf.contrib.layers.conv2d(
        inputs=layer,
        num_outputs=channel,
        kernel_size=3,
        padding='SAME',
        activation_fn=tf.nn.leaky_relu) # remplacé None
    
#    layer = tf.contrib.layers.batch_norm(
#        inputs=layer,
#        activation_fn=tf.nn.leaky_relu
   
    layer = tf.contrib.layers.conv2d(
        inputs=layer,
        num_outputs=channel,
        kernel_size=1, # à quoi ça sert
        padding='SAME',
        activation_fn=tf.nn.leaky_relu) # remplacé None
    
#    layer = tf.contrib.layers.batch_norm(
#        inputs=layer,
#        activation_fn=tf.nn.leaky_relu)
    
    height, width = layer.get_shape()[1:3]
    layer = tf.image.resize_images(
        images = layer,
        size = [height*2, width*2]
    )
    
    return layer

def skip(layer):
    conv_out = tf.contrib.layers.conv2d(
        inputs=layer,
        num_outputs=4,
        kernel_size=1,
        stride=1,
        padding='SAME',
        normalizer_fn = tf.contrib.layers.batch_norm,
        activation_fn=tf.nn.leaky_relu)

    return conv_out

# Code from https://stackoverflow.com/a/29731818
def gkern(kernlen=5, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return (tf.convert_to_tensor(kernel, dtype=tf.float32))

# Apply the gaussian kernel to each channel to give the image a
# gaussian blur.
def gblur(layer):
    gaus_filter = tf.expand_dims(tf.stack([gkern(),gkern(),gkern()], axis=2), axis=3)
    if blur:
        return tf.nn.depthwise_conv2d(layer, gaus_filter, strides=[1,1,1,1], padding='SAME')
    else:
        return layer


image = load_image(image_name, dim)

rand = tf.placeholder(shape=(1,dim,dim,32), dtype=tf.float32)

# TODO: test if 32 channels improves performance
out = tf.constant(np.random.uniform(0, 0.1, size=(1,dim,dim,32)), dtype=tf.float32) + rand

# Connect up all the downsampling layers.
skips = []
# with tf.device('/GPU:0:'): # check name of GPU and indent the following lines.
for i in range(down_layer_count):
    out = down_layer(out,channels_per_layer[i])
    # Keep a list of the skip layers, so they can be connected
    # to the upsampling layers.
    skips.append(skip(out))

    print("Shape after downsample layer " + str(i) + ":" + str(out.get_shape()))
    if log:
        f = open(output_folder + log_file,'w+') ##### déplacer ####
        f.write("Shape after downsample layer " + str(i) + ":" + str(out.get_shape()) + "\n")

# Connect up the upsampling layers, from smallest to largest.
skips.reverse()
for i in range(up_layer_count):
    if i == 0:
        # As specified in the paper, the first upsampling layers is connected to
        # the last downsampling layer through a skip layer.
        out = up_layer(skip(out),channels_per_layer[-1])

        print("Shape after upsample " + str(i) + ":" + str(out.get_shape()))
        if log:
            f.write("Shape after upsample " + str(i) + ":" + str(out.get_shape()) + "\n")

    else:
        # The output of the rest of the skip layers is concatenated onto
        # the input of each upsampling layer.
        # Note: It's not clear from the paper if concat is the right operator
        # but nothing else makes sense for the shape of the tensors.
        out = up_layer(tf.concat([out, skips[i]], axis=3),channels_per_layer[-i-1])
        
        print("Shape after upsample " + str(i) + ":" + str(out.get_shape()))
        if log:
            f.write("Shape after upsample " + str(i) + ":" + str(out.get_shape()) + "\n")

# Restore original image dimensions and channels
out = tf.contrib.layers.conv2d(
    inputs=out,
    num_outputs=3,
    kernel_size=1,
    stride=1,
    padding='SAME',
    activation_fn=tf.nn.sigmoid)
print("Output shape: " + str(out.get_shape()))
if log:
    f.write("Output shape: " + str(out.get_shape()) + "\n")

E = tf.losses.mean_squared_error(image, gblur(out))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(E)

# sess = tf.InteractiveSession()
# https://www.tensorflow.org/guide/using_gpu
sess = tf.Session() # config=tf.ConfigProto(log_device_placement=True
sess.run(tf.global_variables_initializer())

save_image(output_folder + "corrupt.png", tf.reshape(image, (dim,dim,3)))

t_initial = dt.datetime.now()
print("Démarrage calcul: ",t_initial)
if log:
    f.write("Démarrage calcul: " + str(t_initial) + "\n")

for i in range(nbre_iterations):
    print("STEP 1")
    new_rand = np.random.uniform(0, 1.0/30.0, size=(1,dim,dim,32))
    _, lossval = sess.run(
        [train_op, E],
        feed_dict = {rand: new_rand}
    )
    print("STEP 2")
    if i % 100 == 0:
        image_out = sess.run(out, feed_dict={rand: new_rand}).reshape(dim,dim,3)
        save_image(output_folder + "%d_%s" % (i, image_name) + ".png", image_out)
    
    t_new = dt.datetime.now()
    t_interval = t_new - t_initial
    t_initial = t_new
    print(i, lossval, t_interval)
    if log:
        f.write("itération lossval temps: " + str(i) + "\t" + str(lossval) + "\t" + str(t_interval) + "\t" + "\n")

f.close()
