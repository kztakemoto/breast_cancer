# compile Kaggle images (https://www.kaggle.com/c/histopathologic-cancer-detection) to npy format files
import pandas as pd
import numpy as np
import cv2

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold

from utils.data_generator import do_inference_aug

# reading the training data CSV file (need to modify the path)
df = pd.read_csv("../histopathologic-cancer-detection/train_labels.csv")
df_train, df_val = train_test_split(df, test_size=0.1, stratify= df['label'], random_state=123)

print("Train data: " + str(len(df_train[df_train["label"] == 1]) + len(df_train[df_train["label"] == 0])))
print("True positive in train data: " +  str(len(df_train[df_train["label"] == 1])))
print("True negative in train data: " +  str(len(df_train[df_train["label"] == 0])))
print("Valid data: " + str(len(df_val[df_val["label"] == 1]) + len(df_val[df_val["label"] == 0])))
print("True positive in validation data: " +  str(len(df_val[df_val["label"] == 1])))
print("True negative in validation data: " +  str(len(df_val[df_val["label"] == 0])))

# Train List (need to modify the path)
train_list = df_train['id'].tolist()
train_list = ['../histopathologic-cancer-detection/train/'+ name + ".tif" for name in train_list]

# Validation List (need to modify the path)
val_list = df_val['id'].tolist()
val_list = ['../histopathologic-cancer-detection/train/'+ name + ".tif" for name in val_list]

#### save validation data in npy format
# image data
X = [cv2.resize(cv2.imread(x), (96, 96)) for x in val_list]
np.save('data/X_val.npy', np.asarray(X, dtype='float32'))
# label data
y = df_val['label'].values
y = np.asarray(y)
y = y.reshape(len(y), 1)
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y)
y = np.asarray(y, dtype='float32')
print(y.shape)
np.save('data/y_val.npy', y)


### save train data in npy format
# image data
X = [cv2.resize(cv2.imread(x), (96, 96)) for x in train_list]
np.save('data/X_train.npy', np.asarray(X, dtype='float32'))
# label data
y = df_train['label'].values
y = np.asarray(y)
y = y.reshape(len(y), 1)
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y)
y = np.asarray(y, dtype='float32')
print(y.shape)
np.save('data/y_train.npy', y)

