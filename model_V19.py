# -*- coding: utf-8 -*-
import tensorflow as tf

l2 = tf.keras.regularizers.l2

def fix_GL_network(input_shape=(128, 88, 1), weight_decay=0.00001, num_classes=86):

    h = inputs = tf.keras.Input(input_shape)

    crop_1 = tf.image.crop_to_bounding_box(h, 0, 0, 22, 88)
    crop_2 = tf.image.crop_to_bounding_box(h, 22, 0, 48, 88)
    crop_3 = tf.image.crop_to_bounding_box(h, 70, 0, 58, 88)

    ###########################################################################################
    crop_1 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_1)
    crop_1 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1)
    crop_1 = tf.keras.layers.BatchNormalization()(crop_1)
    crop_1 = tf.keras.layers.LeakyReLU()(crop_1)

    crop_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_1)

    crop_1 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1)
    crop_1 = tf.keras.layers.BatchNormalization()(crop_1)
    crop_1 = tf.keras.layers.LeakyReLU()(crop_1)

    crop_1_max = tf.keras.layers.GlobalMaxPool2D()(crop_1)
    crop_1_max = tf.keras.layers.Dense(64/16)(crop_1_max)
    crop_1_max = tf.keras.layers.ReLU()(crop_1_max)
    crop_1_max = tf.keras.layers.Dense(64)(crop_1_max)

    crop_1_avg = tf.keras.layers.GlobalAveragePooling2D()(crop_1)
    crop_1_avg = tf.keras.layers.Dense(64/16)(crop_1_avg)
    crop_1_avg = tf.keras.layers.ReLU()(crop_1_avg)
    crop_1_avg = tf.keras.layers.Dense(64)(crop_1_avg)

    crop_1_sum = crop_1_max + crop_1_avg
    crop_1_sum = tf.nn.sigmoid(crop_1_sum)
    crop_1_sum = tf.expand_dims(crop_1_sum, 1)
    crop_1_sum = tf.expand_dims(crop_1_sum, 1)
    crop_1_ = tf.math.multiply(crop_1, crop_1_sum)
    
    crop_1_multi = tf.concat([ tf.reduce_max(crop_1_, -1, keepdims=True), tf.reduce_mean(crop_1_, -1, keepdims=True) ], -1)
    crop_1_multi = tf.keras.layers.Conv2D(filters=1,
                                    kernel_size=(5,7),
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1_multi)
    crop_1_multi = tf.keras.layers.BatchNormalization()(crop_1_multi)
    crop_1 = tf.multiply(crop_1_multi, crop_1_)
    crop_1 = tf.nn.sigmoid(crop_1)       
    ###########################################################################################

    ###########################################################################################
    crop_2 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_2)
    crop_2 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2)
    crop_2 = tf.keras.layers.BatchNormalization()(crop_2)
    crop_2 = tf.keras.layers.LeakyReLU()(crop_2)

    crop_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_2)

    crop_2 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2)
    crop_2 = tf.keras.layers.BatchNormalization()(crop_2)
    crop_2 = tf.keras.layers.LeakyReLU()(crop_2)

    crop_2_max = tf.keras.layers.GlobalMaxPool2D()(crop_2)
    crop_2_max = tf.keras.layers.Dense(64/16)(crop_2_max)
    crop_2_max = tf.keras.layers.ReLU()(crop_2_max)
    crop_2_max = tf.keras.layers.Dense(64)(crop_2_max)

    crop_2_avg = tf.keras.layers.GlobalAveragePooling2D()(crop_2)
    crop_2_avg = tf.keras.layers.Dense(64/16)(crop_2_avg)
    crop_2_avg = tf.keras.layers.ReLU()(crop_2_avg)
    crop_2_avg = tf.keras.layers.Dense(64)(crop_2_avg)

    crop_2_sum = crop_2_max + crop_2_avg
    crop_2_sum = tf.nn.sigmoid(crop_2_sum)
    crop_2_sum = tf.expand_dims(crop_2_sum, 1)
    crop_2_sum = tf.expand_dims(crop_2_sum, 1)
    crop_2_ = tf.math.multiply(crop_2, crop_2_sum)

    crop_2_multi = tf.concat([ tf.reduce_max(crop_2_, -1, keepdims=True), tf.reduce_mean(crop_2_, -1, keepdims=True) ], -1)
    crop_2_multi = tf.keras.layers.Conv2D(filters=1,
                                    kernel_size=(5,7),
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2_multi)
    crop_2_multi = tf.keras.layers.BatchNormalization()(crop_2_multi)
    crop_2 = tf.multiply(crop_2_multi, crop_2_)
    crop_2 = tf.nn.sigmoid(crop_2)
    ###########################################################################################

    ###########################################################################################
    crop_3 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_3)
    crop_3 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3)
    crop_3 = tf.keras.layers.BatchNormalization()(crop_3)
    crop_3 = tf.keras.layers.LeakyReLU()(crop_3)

    crop_3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_3)

    crop_3 = tf.keras.layers.ZeroPadding2D((1, 3))(crop_3)
    crop_3 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3)
    crop_3 = tf.keras.layers.BatchNormalization()(crop_3)
    crop_3 = tf.keras.layers.LeakyReLU()(crop_3)

    crop_3_max = tf.keras.layers.GlobalMaxPool2D()(crop_3)
    crop_3_max = tf.keras.layers.Dense(64/16)(crop_3_max)
    crop_3_max = tf.keras.layers.ReLU()(crop_3_max)
    crop_3_max = tf.keras.layers.Dense(64)(crop_3_max)

    crop_3_avg = tf.keras.layers.GlobalAveragePooling2D()(crop_3)
    crop_3_avg = tf.keras.layers.Dense(64/16)(crop_3_avg)
    crop_3_avg = tf.keras.layers.ReLU()(crop_3_avg)
    crop_3_avg = tf.keras.layers.Dense(64)(crop_3_avg)

    crop_3_sum = crop_3_max + crop_3_avg
    crop_3_sum = tf.nn.sigmoid(crop_3_sum)
    crop_3_sum = tf.expand_dims(crop_3_sum, 1)
    crop_3_sum = tf.expand_dims(crop_3_sum, 1)
    crop_3_ = tf.math.multiply(crop_3, crop_3_sum)

    crop_3_multi = tf.concat([ tf.reduce_max(crop_3_, -1, keepdims=True), tf.reduce_mean(crop_3_, -1, keepdims=True) ], -1)
    crop_3_multi = tf.keras.layers.Conv2D(filters=1,
                                    kernel_size=(5,7),
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3_multi)
    crop_3_multi = tf.keras.layers.BatchNormalization()(crop_3_multi)
    crop_3 = tf.multiply(crop_3_multi, crop_3_)
    crop_3 = tf.nn.sigmoid(crop_3)
    ###########################################################################################
    
    crop = tf.concat([crop_1, crop_2, crop_3], 1)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h_max = tf.keras.layers.GlobalMaxPool2D()(h)
    h_max = tf.keras.layers.Dense(64/16)(h_max)
    h_max = tf.keras.layers.ReLU()(h_max)
    h_max = tf.keras.layers.Dense(64)(h_max)

    h_avg = tf.keras.layers.GlobalAveragePooling2D()(h)
    h_avg = tf.keras.layers.Dense(64/16)(h_avg)
    h_avg = tf.keras.layers.ReLU()(h_avg)
    h_avg = tf.keras.layers.Dense(64)(h_avg)

    h_sum = h_max + h_avg
    h_sum = tf.nn.sigmoid(h_sum)
    h_sum = tf.expand_dims(h_sum, 1)
    h_sum = tf.expand_dims(h_sum, 1)
    h_ = tf.math.multiply(h, h_sum)

    h_multi = tf.concat([ tf.reduce_max(h_, -1, keepdims=True), tf.reduce_mean(h_, -1, keepdims=True) ], -1)
    h_multi = tf.keras.layers.Conv2D(filters=1,
                                kernel_size=(5,7),
                                padding="same",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h_multi)
    h_multi = tf.keras.layers.BatchNormalization()(h_multi)
    h = tf.multiply(h_multi, h_)
    h = tf.nn.sigmoid(h)

    h = tf.concat([h, crop], -1)

    h = tf.keras.layers.ZeroPadding2D((2,2))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h_max = tf.keras.layers.GlobalMaxPool2D()(h)
    h_max = tf.keras.layers.Dense(128/16)(h_max)
    h_max = tf.keras.layers.ReLU()(h_max)
    h_max = tf.keras.layers.Dense(128)(h_max)

    h_avg = tf.keras.layers.GlobalAveragePooling2D()(h)
    h_avg = tf.keras.layers.Dense(128/16)(h_avg)
    h_avg = tf.keras.layers.ReLU()(h_avg)
    h_avg = tf.keras.layers.Dense(128)(h_avg)

    h_sum = h_max + h_avg
    h_sum = tf.nn.sigmoid(h_sum)
    h_sum = tf.expand_dims(h_sum, 1)
    h_sum = tf.expand_dims(h_sum, 1)
    h_ = tf.math.multiply(h, h_sum)

    h_multi = tf.concat([ tf.reduce_max(h_, -1, keepdims=True), tf.reduce_mean(h_, -1, keepdims=True) ], -1)
    h_multi = tf.keras.layers.Conv2D(filters=1,
                                kernel_size=7,
                                padding="same",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h_multi)
    h_multi = tf.keras.layers.BatchNormalization()(h_multi)
    h = tf.multiply(h_multi, h_)
    h = tf.nn.sigmoid(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.ZeroPadding2D((2,2))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h_max = tf.keras.layers.GlobalMaxPool2D()(h)
    h_max = tf.keras.layers.Dense(128/16)(h_max)
    h_max = tf.keras.layers.ReLU()(h_max)
    h_max = tf.keras.layers.Dense(128)(h_max)

    h_avg = tf.keras.layers.GlobalAveragePooling2D()(h)
    h_avg = tf.keras.layers.Dense(128/16)(h_avg)
    h_avg = tf.keras.layers.ReLU()(h_avg)
    h_avg = tf.keras.layers.Dense(128)(h_avg)

    h_sum = h_max + h_avg
    h_sum = tf.nn.sigmoid(h_sum)
    h_sum = tf.expand_dims(h_sum, 1)
    h_sum = tf.expand_dims(h_sum, 1)
    h_ = tf.math.multiply(h, h_sum)

    h_multi = tf.concat([ tf.reduce_max(h_, -1, keepdims=True), tf.reduce_mean(h_, -1, keepdims=True) ], -1)
    h_multi = tf.keras.layers.Conv2D(filters=1,
                                kernel_size=7,
                                padding="same",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h_multi)
    h_multi = tf.keras.layers.BatchNormalization()(h_multi)
    h = tf.multiply(h_multi, h_)
    h = tf.nn.sigmoid(h)

    h = tf.keras.layers.GlobalMaxPool2D()(h)

    h = tf.keras.layers.Dense(1024)(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.Dropout(0.5)(h)

    h = tf.keras.layers.Dense(1024)(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.Dropout(0.5)(h)

    h1 = tf.keras.layers.Dense(num_classes)(h)

    h = tf.keras.layers.Dense(90)(h)

    h = tf.keras.layers.Reshape((9, 10))(h)


    return tf.keras.Model(inputs=inputs, outputs=[h, h1])

model = fix_GL_network()
model.summary()