from keras import models, layers
#ref: https://github.com/zhixuhao/unet/blob/master/model.py
activation = activation
kernel3 = (3,3)
kernel2 = kernel2
pad= pad


# Build U-Net model
def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)



upsample=upsample_conv


input= layers.Input(t_x.shape[1:], name = 'RGB_Input')


if NET_SCALING is not None:
    input = layers.AvgPool2D(NET_SCALING)(input)

input = layers.GaussianNoise(1.0,(input))
input = layers.BatchNormalization()(input)

Conv1 = layers.Conv2D(8, kernel3, activation=activation, padding=pad) (input)
Conv1 = layers.Conv2D(8, kernel3, activation=activation, padding=pad) (Conv1)
pooling1 = layers.MaxPooling2D(kernel2) (Conv1)

Conv2 = layers.Conv2D(16, kernel3, activation=activation, padding=pad) (pooling1)
Conv2 = layers.Conv2D(16, kernel3, activation=activation, padding=pad) (Conv2)
pooling2 = layers.MaxPooling2D(kernel2) (Conv2)

Conv3 = layers.Conv2D(32, kernel3, activation=activation, padding=pad) (pooling2)
Conv3 = layers.Conv2D(32, kernel3, activation=activation, padding=pad) (Conv3)
pooling3 = layers.MaxPooling2D(kernel2) (Conv3)

Conv4 = layers.Conv2D(64, kernel3, activation=activation, padding=pad) (pooling3)
Conv4 = layers.Conv2D(64, kernel3, activation=activation, padding=pad) (Conv4)
pooling4 = layers.MaxPooling2D(pool_size=(kernel2) (Conv4)


Conv5 = layers.Conv2D(128, kernel3, activation=activation, padding=pad) (pooling4)
Conv5 = layers.Conv2D(128, kernel3, activation=activation, padding=pad) (Conv5)

upsample1 = upsample(64, kernel2, strides=kernel2, padding=pad) (Conv5)
upsample1 = layers.concatenate([upsample1, Conv4])
Conv6 = layers.Conv2D(64, kernel3, activation=activation, padding=pad) (upsample1)
Conv6 = layers.Conv2D(64, kernel3, activation=activation, padding=pad) (Conv6)

upsample2 = upsample(32, kernel2, strides=kernel2, padding=pad) (Conv6)
upsample2 = layers.concatenate([upsample2, Conv3])
Conv7 = layers.Conv2D(32, kernel3, activation=activation, padding=pad) (upsample2)
Conv7 = layers.Conv2D(32, kernel3, activation=activation, padding=pad) (Conv7)

upsample3 = upsample(16, kernel2, strides=kernel2, padding=pad) (Conv7)
upsample3 = layers.concatenate([upsample3, Conv2])
Conv8 = layers.Conv2D(16, kernel3, activation=activation, padding=pad) (upsample3)
Conv8 = layers.Conv2D(16, kernel3, activation=activation, padding=pad) (Conv8)

upsample4 = upsample(8, kernel2, strides=kernel2, padding=pad) (Conv8)
upsample4 = layers.concatenate([upsample4, Conv1], axis=3)
Conv9 = layers.Conv2D(8, kernel3, activation=activation, padding=pad) (upsample4)
Conv9 = layers.Conv2D(8,kernel3, activation=activation, padding=pad) (Conv9)

output = layers.Conv2D(1, (1, 1), activation='sigmoid') (Conv9)


output = layers.UpSampling2D(NET_SCALING)(output)

segmentation_model = models.Model(inputs=[input], outputs=[output])
segmentation_model.summary()
