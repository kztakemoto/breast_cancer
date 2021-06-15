# Training with simple hold-out train-val split
import pandas as pd
import numpy as np
import cv2
import os

from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.models import Model

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utils.one_cycle_policy_lr import OneCycleLR
from utils.data_generator import data_gen, do_train_augmentations, do_inference_aug, plot_confusion_matrix
from model.OctaveResNet import OctaveResNet50
#from model.ResNeXt_CBAM import *

### config ###
img_size = (96,96,3)
batch_size = 64
#epochs = 38
epochs = 1

### load image data ###
# reading the training data CSV file (may need to modify the path)
df = pd.read_csv("./histopathologic-cancer-detection/train_labels.csv")
df_train, df_val = train_test_split(df, test_size=0.1, stratify= df['label'], random_state=123)

print("Train data: " + str(len(df_train[df_train["label"] == 1]) + len(df_train[df_train["label"] == 0])))
print("True positive in train data: " +  str(len(df_train[df_train["label"] == 1])))
print("True negative in train data: " +  str(len(df_train[df_train["label"] == 0])))
print("Valid data: " + str(len(df_val[df_val["label"] == 1]) + len(df_val[df_val["label"] == 0])))
print("True positive in validation data: " +  str(len(df_val[df_val["label"] == 1])))
print("True negative in validation data: " +  str(len(df_val[df_val["label"] == 0])))

# Train List (may need to modify the path)
train_list = df_train['id'].tolist()
train_list = ['./histopathologic-cancer-detection/train/'+ name + ".tif" for name in train_list]

# Validation List (may need to modify the path)
val_list = df_val['id'].tolist()
val_list = ['./histopathologic-cancer-detection/train/'+ name + ".tif" for name in val_list]

# Dictionary mapping Image IDs to corresponding labels....used in data_generator.py
id_label_map = {k:v for k,v in zip(df.id.values, df.label.values)}

# model configuration
# can use different models such as ResNeXt50, Seresnet50 by replacing them.
base_model = OctaveResNet50(include_top=False, weights=None, input_shape=img_size, classes=2, initial_strides=False)
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
model = Model(inputs=base_model.input, outputs=X)

lr_manager = OneCycleLR(max_lr=0.02, end_percentage=0.1, scale_percentage=None,
                        maximum_momentum=0.9,minimum_momentum=0.8)

callbacks = [lr_manager,
           ModelCheckpoint(filepath='weight/octresnet_one_cycle_model.h5', monitor='val_loss',mode='min',verbose=1,save_best_only=True)]

model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(0.002, momentum=0.9, nesterov=True), metrics=['accuracy'])


### Training ####
history = model.fit_generator(data_gen(train_list, id_label_map, batch_size, 96, do_train_augmentations),
                              validation_data=data_gen(val_list, id_label_map, batch_size, 96, do_inference_aug),
                              epochs = epochs,
                              steps_per_epoch = (len(train_list) // batch_size) + 1,
                              validation_steps = (len(val_list) // batch_size) + 1,
                              callbacks=callbacks,
                              verbose = 1)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "valid"], loc="upper left")
plt.savefig('loss_performance.png')
plt.clf()
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='valid')
plt.title("model acc")
plt.ylabel("acc")
plt.xlabel("epoch")
plt.legend(["train", "valid"], loc="upper left")
plt.savefig('acc_performance.png')


### Predition ###
preds = model.predict_generator(data_gen(val_list, id_label_map, 1, 96, do_inference_aug), steps = len(val_list))
preds = np.array(preds)
prob_1 = preds[:,1:2].flatten()
y_preds = np.argmax(preds, axis=1)

#Taking 0.5 as threshold for classification
true = df_val['label'].values

# printing AUC_ROC score
print(roc_auc_score(true,prob_1))

# Plotting the AUC_ROC curve
fpr, tpr, threshold = roc_curve(true, prob_1)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'g', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig('octresnet50_auc_roc.png')

# printing and plotting confusion matrix
cm = confusion_matrix(true,y_preds)
plot_confusion_matrix(cm,['no_tumor_tissue', 'has_tumor_tissue'])

# classification report consist of precision, recall, f1-score
report = classification_report(true,y_preds,target_names=['no_tumor_tissue', 'has_tumor_tissue'])
print(report)
