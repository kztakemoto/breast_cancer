# Training code for simple hold-out train-val split
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, concatenate
from keras.optimizers import SGD
from keras.models import Model

from utils.data_generator import do_inference_aug

from model.OctaveResNet import OctaveResNet50

# model configuration (Octave ResNet50)
base_model = OctaveResNet50(include_top=False, weights=None, input_shape=(96,96,3), initial_strides=False)
x = base_model.output
out1 = GlobalMaxPooling2D()(x)
out2 = GlobalAveragePooling2D()(x)
out = concatenate([out1,out2])
out = BatchNormalization(epsilon = 1e-5)(out)
out = Dropout(0.4)(out)
fc = Dense(512,activation = 'relu')(out)
fc = BatchNormalization(epsilon = 1e-5)(fc)
fc = Dropout(0.3)(fc)
fc = Dense(256,activation = 'relu')(fc)
fc = BatchNormalization(epsilon = 1e-5)(fc)
fc = Dropout(0.3)(fc)
X = Dense(2, activation='softmax')(fc)
model =  Model(inputs=base_model.input, outputs=X)
model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(0.002, momentum=0.9, nesterov=True), metrics=['accuracy'])
# load model weight
model.load_weights('weight/octresnet_one_cycle_model.h5')

# load image data
X_val = np.load('data/X_val.npy')
# normalization
aug = do_inference_aug()
X_val = [aug(image=x)['image'] for x in X_val]
X_val = np.asarray(X_val, dtype='float32') 
# load label data
y_val = np.load('data/y_val.npy')

# prediction on X_val
preds = np.array(model.predict(X_val))

# compute accuracy
acc = np.sum(np.argmax(preds, axis =1) == np.argmax(y_val, axis=1)) / y_val.shape[0]
print(" Accuracy [val]: {:.2f}".format(acc*100.0))

