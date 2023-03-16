from utils4models import *
from keras.layers import *
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.layers import AveragePooling2D, Concatenate, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, NASNetMobile
from tensorflow.keras import metrics as ms


lr = 0.001
m = 0.9
dr = 0.3


def custom_metrics(number_of_classes=2):
    metrics = ['accuracy',
               tfa.metrics.F1Score(num_classes=number_of_classes, threshold=0.5, average='weighted'),
               tfa.metrics.CohenKappa(num_classes=number_of_classes),
               tfa.metrics.MatthewsCorrelationCoefficient(num_classes=number_of_classes),
               ms.AUC(name='auc'),
               ]
    return metrics


def unet(input_shape=(64, 64, 3), number_of_classes=2, include_top=True):
    inputs = Input(input_shape)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    if include_top:
        # Classification block
        flatten = Flatten(name='flatten')(conv8)
        fc1 = Dense(64, activation='relu')(flatten)
        drop1 = Dropout(dr)(fc1)
        fc2 = Dense(32, activation='relu')(drop1)
        drop2 = Dropout(dr)(fc2)
        out = Dense(number_of_classes, activation='softmax')(drop2)

    else:
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        out = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=out, name='unet')
    print(model)
    opt = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=custom_metrics(number_of_classes))
    #print(model.summary())
    return model


def vgg16(input_shape=(64, 64, 3), number_of_classes=2, include_top=True, global_avg_cf=True):
    """An forked version of Keras's application VGG-16 with Dropout layers added between FCs. """
    img_input = Input(shape=input_shape)
    # Block 1
    conv1_1 = Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv1')(img_input)
    conv1_2 = Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv2')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv1')(pool1)
    conv2_2 = Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv2')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv1')(pool2)
    conv3_2 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv3')(conv3_2)

    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3_3)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv1')(pool3)
    conv4_2 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv3')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4_3)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv1')(pool4)
    conv5_2 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv3')(conv5_2)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5_3)

    if include_top:
        # Classification block
        flatten = Flatten(name='flatten')(pool5)
        fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
        drop1 = Dropout(dr)(fc1)
        fc2 = Dense(4096, activation='relu', name='fc2')(drop1)
        drop2 = Dropout(dr)(fc2)
        out = Dense(number_of_classes, activation='softmax', name='predictions')(drop2)
        
    elif global_avg_cf:
        final_conv = Conv2D(number_of_classes, kernel_size=(3, 3), padding='same', activation='relu')
        final_pool = GlobalAveragePooling2D()
        activation = Activation('softmax')
        
    else:
        if pooling == 'avg':
            out = GlobalAveragePooling2D(name='global_average_pooling2d')(pool5)
        elif pooling == 'max':
            out = GlobalMaxPooling2D()(pool5)
            
    model_name = "vgg16" + ("_global_avg_cf" if global_avg_cf==True else "")
  
    model = Model(inputs=img_input, outputs=out, name=model_name)
    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=custom_metrics(number_of_classes))
    #print(model.summary())
    return model

def vgg5_block(input_shape=(256, 256, 3), number_of_classes=2, include_top=True, global_avg_cf=True):
    
    model_name = "vgg5_block" + ("_global_avg_cf" if global_avg_cf==True else "")
    model = Sequential(name= model_name)
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    
    if include_top:
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(number_of_classes, activation='softmax'))
        
    elif global_avg_cf:
        model.add(Dropout(0.25))
        model.add(Conv2D(number_of_classes, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax'))
        
    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=custom_metrics(number_of_classes)
                 )
    return model


def vgg19(input_shape=(64, 64, 3), number_of_classes=2, include_top=True, global_avg_cf=True):
    """An forked version of Keras's application VGG-19 with Dropout layers added between FCs. """
    img_input = Input(shape=input_shape)
    # Block 1
    conv1_1 = Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv1')(img_input)
    conv1_2 = Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv2')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv1')(pool1)
    conv2_2 = Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv2')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv1')(pool2)
    conv3_2 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv3')(conv3_2)

    conv3_4 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv4')(conv3_3)

    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3_4)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv1')(pool3)
    conv4_2 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv3')(conv4_2)
    conv4_4 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv4')(conv4_3)

    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4_4)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv1')(pool4)
    conv5_2 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv3')(conv5_2)
    conv5_4 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv4')(conv5_3)

    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5_4)

    if include_top:
        # Classification block
        flatten = Flatten(name='flatten')(pool5)
        fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
        drop1 = Dropout(dr)(fc1)
        fc2 = Dense(4096, activation='relu', name='fc2')(drop1)
        drop2 = Dropout(dr)(fc2)
        out = Dense(number_of_classes, activation='softmax', name='predictions')(drop2)
            
    elif global_avg_cf:
        final_conv = Conv2D(number_of_classes, kernel_size=(3, 3), padding='same', activation='relu')
        final_pool = GlobalAveragePooling2D()
        activation = Activation('softmax')
        
    else:
        if pooling == 'avg':
            out = GlobalAveragePooling2D(name='global_average_pooling2d')(pool5)
        elif pooling == 'max':
            out = GlobalMaxPooling2D()(pool5)
            
    model_name = "vgg19" + ("_global_avg_cf" if global_avg_cf==True else "")
    model = Model(inputs=img_input, outputs=out, name=model_name)
    # Compile Model
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=custom_metrics(number_of_classes))
    #print(model.summary())
    return model


def resnet50(input_shape=(64, 64, 3), number_of_classes=2, include_top=True, global_avg_cf=True):
    # Input
    img_input = Input(input_shape)

    # Fork Resnet50 from Keras.
    base_model = ResNet50(weights=None, include_top=False, input_tensor=img_input)

    if include_top:
        # Classification block
        flatten = Flatten(name='flatten')(base_model.get_layer('conv4_block6_out').output)
        fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
        drop1 = Dropout(dr)(fc1)
        fc2 = Dense(4096, activation='relu', name='fc2')(drop1)
        drop2 = Dropout(dr)(fc2)
        out = Dense(number_of_classes, activation='softmax', name='predictions')(drop2)
    
    elif global_avg_cf:
        final_conv = Conv2D(number_of_classes, kernel_size=(3, 3), padding='same', activation='relu')
        final_pool = GlobalAveragePooling2D()
        activation = Activation('softmax')
    
    model_name = "resnet50" + ("_global_avg_cf" if global_avg_cf==True else "") 
    model = Model(inputs=img_input, outputs=out, name=model_name)

    # Compile model
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=custom_metrics(number_of_classes))
    #print(model.summary())
    return model


def deeplabv3plus(input_shape=(64, 64, 3), number_of_classes=2, include_top=True):
    """Forked from https://github.com/PreetKumarBAU/Chagas-Segmentation-Models """

    # Input
    img_input = Input(input_shape)

    # Base Model (Can this change?)
    base_model = ResNet50(weights=None, include_top=False, input_tensor=img_input)

    # ASPP Block
    img_features = base_model.get_layer('conv4_block6_out').output
    x_a = ASPP(img_features)
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)

    # Get low-level features
    x_b = base_model.get_layer('conv2_block2_out').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)

    x = Concatenate()([x_a, x_b])

    # Convolutional Block 1
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Convolutional Block 2
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)

    # Outputs
    x = Conv2D(2, kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1))(x)
    x = Conv2D(1, (1, 1), name='output_layer')(x)
    out = Activation('sigmoid')(x)

    if include_top:
        x = GlobalAveragePooling2D()(base_model.output)
        out = Dense(number_of_classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            out = GlobalAveragePooling2D(name='global_average_pooling2d')(base_model.output)
        elif pooling == 'max':
            out = GlobalMaxPooling2D()(base_model.output)

    model = Model(inputs=base_model.input, outputs=out, name='deeplabv3plus')
    # Compile model
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=custom_metrics(number_of_classes))
    #print(model.summary())
    return model


def nasnet_mobile(input_shape=(64, 64, 3), number_of_classes=2, include_top=True):
    """Forked from https://github.com/PreetKumarBAU/Chagas-Segmentation-Models """
    base_model = NASNetMobile(input_shape=input_shape,
                              include_top=False,
                              weights=None,
                              input_tensor=None,
                              pooling=None)

    if include_top:
        x = GlobalAveragePooling2D()(base_model.output)
        out = Dense(number_of_classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            out = GlobalAveragePooling2D(name='global_average_pooling2d')(base_model.output)
        elif pooling == 'max':
            out = GlobalMaxPooling2D()(base_model.output)

    model = Model(inputs=base_model.input, outputs=out, name='nasnetmobile')

    # Compile model
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=custom_metrics(number_of_classes))
    #print(model.summary())
    return model


def inception_resnetv2(input_shape=(64, 64, 3), number_of_classes=2, include_top=True):
    base_model = InceptionResNetV2(input_shape=input_shape,
                                   include_top=False,
                                   weights=None,
                                   input_tensor=None,
                                   pooling=None)
    if include_top:
        x = GlobalAveragePooling2D()(base_model.output)
        out = Dense(number_of_classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            out = GlobalAveragePooling2D(name='global_average_pooling2d')(base_model.output)
        elif pooling == 'max':
            out = GlobalMaxPooling2D()(base_model.output)

    model = Model(inputs=base_model.input, outputs=out, name='inception-resnetv2')

    # Compile model
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=custom_metrics(number_of_classes))
    #print(model.summary())
    return model


def double_unet(input_shape=(64, 64, 3), number_of_classes=2, include_top=True):
    """Forked from https://github.com/PreetKumarBAU/Chagas-Segmentation-Models """
    img_input = Input(input_shape)
    x, skip_1 = encoder1(img_input)
    x = ASPP_v2(x, 64)
    x = decoder1(x, skip_1)
    outputs1 = output_block(x)

    x = img_input * outputs1

    x, skip_2 = encoder2(x)
    x = ASPP_v2(x, 64)
    x = decoder2(x, skip_1, skip_2)
    outputs2 = output_block(x)
    outputs = Concatenate()([outputs1, outputs2])

    if include_top:
        # Classification block
        flatten = Flatten(name='flatten')(outputs)
        fc1 = Dense(64, activation='relu')(flatten)
        drop1 = Dropout(dr)(fc1)
        fc2 = Dense(32, activation='relu')(drop1)
        drop2 = Dropout(dr)(fc2)
        out = Dense(number_of_classes, activation='softmax')(drop2)

    model = Model(img_input, out, name='double_unet')
    # Compile model
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=custom_metrics(number_of_classes))
    #print(model.summary())
    return model


# Reference: https://github.com/christianversloot/machine-learning-articles/blob/main/reducing-trainable-parameters-with-a-dense-free-convnet-classifier.md

# Global average pooling can be beneficial to avoid overfit, in terms of reducing number of trainable parameters
def cnn_global_avg(input_shape=(64, 64, 3), number_of_classes=2, include_top=True):
    # Create the model
    model = Sequential(name='cnn_global_avg')
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(number_of_classes, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    # Compile the model
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=custom_metrics(number_of_classes))
    return model
    
def test_cnn_dense(input_shape=(64, 64, 3), number_of_classes=2, include_top=True):
    # Create the model
    model = Sequential(name='test_cnn_dense')
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(number_of_classes, activation='softmax'))

   # Compile the model
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=custom_metrics(number_of_classes))
    return model

