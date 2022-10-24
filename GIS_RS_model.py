from tensorflow.keras.layers import *
from tensorflow.keras import layers,Model
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def GAM_convlstm_res(inputs,time_step,nfilter_lstm,nfilter_conv2D):
    inp_list = tf.split(inputs,num_or_size_splits=time_step,axis=-1)
    list_X_att = []
    for i in range(len(inp_list)):
        c = inp_list[i].shape[-1]
        x_channel_att = conv_GAM(inp_list[i], nfilter_lstm, rate=4)
        list_X_att.append(x_channel_att)
    lstm_inp = tf.stack(list_X_att, axis=1)
    lstm_out1 = ConvLSTM2D(filters=list_X_att[0].shape[-1], kernel_size=(3, 3), strides=(1, 1),padding='same',
                          return_sequences=True,data_format='channels_last', dilation_rate=(1, 1), dropout=0.25,
                          go_backwards=True, recurrent_dropout=0.2)(lstm_inp)
    lstm_out1 = layers.BatchNormalization()(lstm_out1)
    lstm_out1 = layers.Activation('relu')(lstm_out1)
    # lstm_out1 = layers.Dropout(0.2)(lstm_out1)
    lstm_out2 = ConvLSTM2D(filters=list_X_att[0].shape[-1], kernel_size=(3, 3), strides=(1, 1),padding='same',
                          return_sequences=True,data_format='channels_last', dilation_rate=(1, 1), dropout=0.25,
                          go_backwards=True, recurrent_dropout=0.2)(lstm_out1)
    lstm_out2 = layers.BatchNormalization()(lstm_out2)
    lstm_out2 = layers.Activation('relu')(lstm_out2)
    lstm_out_list = tf.unstack(lstm_out2, axis=1)
    lstm_out_conc = concatenate(lstm_out_list, axis=-1)
    if inputs.shape[-1] != lstm_out_conc.shape[-1]:
        inputs = Conv2D(filters=lstm_out_conc.shape[-1], kernel_size=3, strides=1, padding='same', dilation_rate=(1, 1))(inputs)  # same # 'valid'
        inputs = layers.BatchNormalization()(inputs)
        inputs = layers.Activation('relu')(inputs)
    res_output = layers.add([inputs,lstm_out_conc])
    res_output = tf.nn.relu(res_output)
    res_output_list = tf.split(res_output, num_or_size_splits=time_step, axis=-1)   ##
    lstm_out_per = []
    for i in range(len(res_output_list)):
        lstm_out_list_conv = Conv2D(filters=nfilter_conv2D, kernel_size=3, strides=1,padding='same',dilation_rate=(1, 1))(res_output_list[i])   #same # 'valid'
        lstm_out_list_conv = layers.BatchNormalization(momentum=0.95, epsilon=1e-5)(lstm_out_list_conv)
        lstm_out_list_conv = layers.Activation('relu')(lstm_out_list_conv)
        # lstm_out_list_conv = layers.Dropout(0.2)(lstm_out_list_conv)
        ######################
        lstm_out_list_conv = Conv2D(filters=nfilter_conv2D, kernel_size=3, strides=1, padding='same',
                                    dilation_rate=(1, 1))(lstm_out_list_conv)  # same # 'valid'
        lstm_out_list_conv = layers.BatchNormalization(momentum=0.95, epsilon=1e-5)(lstm_out_list_conv)
        lstm_out_list_conv = layers.Activation('relu')(lstm_out_list_conv)
        # lstm_out_list_conv = layers.Dropout(0.2)(lstm_out_list_conv)
        lstm_out_per.append(lstm_out_list_conv)
    lstm_out_layer = tf.concat(lstm_out_per,axis=-1)
    ########################################
    # res_output = layers.add([input_shape,lstm_out_layer])
    # res_output = tf.nn.relu(res_output)
    # return res_output
    return lstm_out_layer


def conv_GAM(inputs,nfilter_conv2D,rate=5):
    conv1 = Conv2D(filters=nfilter_conv2D, kernel_size=3, strides=1, padding='same', dilation_rate=(1, 1))(inputs)  # same # 'valid'
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    # conv1 = layers.Dropout(0.3)(conv1)

    conv2 = Conv2D(filters=nfilter_conv2D, kernel_size=3, strides=1, padding='same', dilation_rate=(1, 1))(conv1)  # same # 'valid'
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)

    x_GAM = GAM_chan_spa(conv2,rate=rate)
    return x_GAM

def GAM_chan_spa(x, rate=5):
    x_channel_att = GAM_channel(x, rate=rate)
    x_out = x * x_channel_att
    x_spatial_att = GAM_spatial(x_out, rate=rate)
    x = x_out * x_spatial_att

    return x

def GAM_channel(x,rate=5):
    in_channels = x.shape[-1]
    x = Conv2D(filters = int(in_channels / rate),kernel_size = 1,padding='same')(x)
    x = Conv2D(filters=int(in_channels), kernel_size=1, padding='same')(x)

    return x

def GAM_spatial(x,rate=5):
    in_channels = x.shape[-1]
    x = Conv2D(filters=int(in_channels / rate), kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=in_channels, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.nn.sigmoid(x)

    return x

class modelMy(object):
    @staticmethod
    def build(input_shape, time_step,nfilters_list,cls_num):
        ##############################
        inp_shp = Input(input_shape)
        convlstm_out1 = GAM_convlstm_res(inp_shp, time_step, nfilters_list[0], nfilters_list[0 + 1])
        convlstm_out2 = GAM_convlstm_res(convlstm_out1, time_step, nfilters_list[1], nfilters_list[1 + 1])
        convlstm_out3 = GAM_convlstm_res(convlstm_out2, time_step, nfilters_list[2], nfilters_list[2 + 1])
        flatten_out1 = Flatten()(convlstm_out1)
        flatten_out11 = tf.nn.l2_normalize(flatten_out1, axis=-1)
        flatten_out1 = layers.Dropout(0.5)(flatten_out1)
        ouput1 = Dense(cls_num, activation='softmax', name='softmax_oup1')(flatten_out1)
        flatten_out2 = Flatten()(convlstm_out2)
        flatten_out22 = tf.nn.l2_normalize(flatten_out2, axis=-1)
        flatten_out2 = layers.Dropout(0.5)(flatten_out2)
        ouput2 = Dense(cls_num, activation='softmax', name='softmax_oup2')(flatten_out2)
        flatten_out3 = Flatten()(convlstm_out3)
        flatten_out33 = tf.nn.l2_normalize(flatten_out3, axis=-1)
        flatten_out3 = layers.Dropout(0.5)(flatten_out3)
        ouput3 = Dense(cls_num, activation='softmax', name='softmax_oup3')(flatten_out3)
        ouput4 = tf.reduce_mean([ouput1, ouput2, ouput3], axis=0)
        # Dense_s2 = Dense(128, activation='relu', name='LSTMDense')(out_s2)
        ##############################
        model = Model(inputs=[inp_shp],outputs=[ouput1, ouput2, ouput3, ouput4, flatten_out11, flatten_out22, flatten_out33])
        # model = Model(inputs=[inp_shp], outputs=[ouput1, ouput2, ouput3, ouput4])
        # model = Model(inputs=[inp_shp], outputs=[ouput3])
        return model
    @staticmethod
    def model_build(input_shape, time_step,nfilters_list,cls_num):
        return modelMy.build(input_shape, time_step,nfilters_list,cls_num)



