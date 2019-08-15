import Utils.IO_img as IO
import tensorflow as tf
import numpy as np
import scipy.stats as st
import datetime as dt

# Name of image to upscale. Must be a dim_h x dim_w BMP.
input_folder = "Input/"
output_folder = "output/"
image_name = "pupper.png"
image_type = "" # initialised by IO_img.load_image()
dim_h = dim_w = 0 # initialised by IO_img.load_image()

log = True # to log in text file all information
log_file = image_name + "_log.txt"

blur = False # valeur initiale True / to be reviewed if not used

# The number of down sampling and up sampling layers.
# These should be equal if the ouput and input images
# are to be equal.
down_layer_count = up_layer_count = 5
channels_per_layer = [8, 16, 32, 64, 128] # size similar to layer_count
channels_per_skip_layer = [4, 4, 4, 4, 4] # size similar to layer_count

nbre_iterations = 201 # valeur initiale 5001

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
    if channel == 0 :
        return layer
    else:
        layer = tf.contrib.layers.conv2d(
            inputs=layer,
            num_outputs=channel,
            kernel_size=3,
            stride=2,
            padding='SAME',
            activation_fn=tf.nn.leaky_relu) # remplacé None
    
        layer = tf.contrib.layers.conv2d(
            inputs=layer,
            num_outputs=channel,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.leaky_relu) # remplacé None
    
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

def skip(layer,channel):
    conv_out = tf.contrib.layers.conv2d(
        inputs=layer,
        num_outputs=channel,
        kernel_size=1,
        stride=1,
        padding='SAME',
        normalizer_fn = tf.contrib.layers.batch_norm,
        activation_fn=tf.nn.leaky_relu)

    return conv_out

image, image_type, (dim_h,dim_w) = IO.load_image(input_folder + image_name)

rand = tf.placeholder(shape=(dim_h,dim_w,32), dtype=tf.float32)

# TODO: test if 32 channels improves performance
out = tf.constant(np.random.uniform(0, 0.1, size=(1,dim_h,dim_w,32)), dtype=tf.float32) + rand

# Connect up all the downsampling layers.
skips = []
# with tf.device('/GPU:0:'): # check name of GPU and indent the following lines.

if log:
    f = open(output_folder + log_file,'w+')

for i in range(down_layer_count):
    print("Shape before downsample layer " + str(i) + ":" + str(out.get_shape()))
    if log:
        f.write("Shape before downsample layer " + str(i) + ":" + str(out.get_shape()) + "\n")


    out = down_layer(out,channels_per_layer[i])
    # Keep a list of the skip layers, so they can be connected
    # to the upsampling layers.
    skips.append(skip(out,channels_per_skip_layer[i]))

    print("Shape after downsample layer " + str(i) + ":" + str(out.get_shape()))
    if log:
        f.write("Shape after downsample layer " + str(i) + ":" + str(out.get_shape()) + "\n")

# Connect up the upsampling layers, from smallest to largest.
skips.reverse()

for i in range(up_layer_count):
    print("Shape before upsample " + str(up_layer_count - i - 1) + ":" + str(out.get_shape()))
    if log:
        f.write("Shape before upsample " + str(up_layer_count - i - 1) + ":" + str(out.get_shape()) + "\n")

    if i == 0:
        # As specified in the paper, the first upsampling layers is connected to
        # the last downsampling layer through a skip layer.
        out = up_layer(out,channels_per_layer[-1])

        print("Shape after upsample " + str(up_layer_count - i - 1) + ":" + str(out.get_shape()))
        if log:
            f.write("Shape after upsample " + str(up_layer_count - i - 1) + ":" + str(out.get_shape()) + "\n")

    else:
        # The output of the rest of the skip layers is concatenated onto
        # the input of each upsampling layer.
        # Note: It's not clear from the paper if concat is the right operator
        # but nothing else makes sense for the shape of the tensors.
        # 
        out_concat = tf.concat([out, skips[i]], axis=3)
        print("Shape after concat " + str(up_layer_count - i - 1) + ":" + str(out_concat.get_shape()))
        if log:
            f.write("Shape after concat " + str(up_layer_count - i - 1) + ":" + str(out_concat.get_shape()) + "\n")


        out = up_layer(out_concat, channels_per_layer[-i-1])
        
        print("Shape after upsample " + str(up_layer_count - i - 1) + ":" + str(out.get_shape()))
        if log:
            f.write("Shape after upsample " + str(up_layer_count - i - 1) + ":" + str(out.get_shape()) + "\n")

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

sess = tf.InteractiveSession()
# https://www.tensorflow.org/guide/using_gpu
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)
sess.run(tf.global_variables_initializer())

save_image(output_folder + "corrupt.png", tf.reshape(image, (dim_h,dim_w,3)))

t_initial = dt.datetime.now()
print("Démarrage calcul: ",t_initial)
if log:
    f.write("Démarrage calcul: " + str(t_initial) + "\n")

for i in range(nbre_iterations):
    print("point 1")
    new_rand = np.random.uniform(0, 1.0/30.0, size=(1,dim_h,dim_w,32))
    print("point 2")

    _, lossval = sess.run(
        [train_op, E],
        feed_dict = {rand: new_rand}
    )
    
    print("point 2")
    if i % 100 == 0:
        image_out = sess.run(out, feed_dict={rand: new_rand}).reshape(dim_h,dim_w,3)
        save_image(output_folder + "%d_%s" % (i, image_name) + ".png", image_out)
    
    t_new = dt.datetime.now()
    t_interval = t_new - t_initial
    t_initial = t_new
    print(i, lossval, t_interval)
    if log:
        f.write("itération lossval temps: " + str(i) + "\t" + str(lossval) + "\t" + str(t_interval) + "\t" + "\n")

f.close()
