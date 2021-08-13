import cv2 
import numpy as np   
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard,CSVLogger
img_width=2048
img_height=2048
defect_folder='C:/Users/Admin/Downloads/dataset_train/dataset_train/spray_defects'
normal_folder='C:/Users/Admin/Downloads/dataset_train/dataset_train/normal'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config = config)
tf.keras.backend.set_session(sess)
###---1---###
import tensorflow as tf
from tensorflow import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
###--1--###

def normalize(img):
    min = img.min()
    max = img.max()
    return 2.0 * (img - min) / (max - min) - 1.0

def norm_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            img=cv2.imread(os.path.join(folder,filename))[:2048,:2048,:3]
#            img = cv2.fastNlMeansDenoising(img,50,50,20,21) 
            img = normalize(img)
            if img is not None:
                images.append(img)
    return images

img_defects_norm=norm_images_from_folder(defect_folder)
img_clean_norm=norm_images_from_folder(normal_folder)
img_all=img_defects_norm+img_clean_norm
Y=[1]*len(img_defects_norm)+[0]*len(img_clean_norm)
#img_all=np.load("img_all.npy")
#Y=np.load("Y_all.npy")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(img_all, Y, 
                    test_size=0.2, stratify = Y, random_state=0)

y_train=np.array(y_train)
y_test=np.array(y_test)
X_train=np.array(X_train)
X_test=np.array(X_test)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
print(y_train[:10])

model = tf.keras.applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. 
for layer in model.layers:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dense(512, activation="relu")(x)
x=Dropout(0.2)(x)
x = Dense(128, activation="relu")(x)
x = Dense(56, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)
# creating the final model 
model_final = Model(model.input, predictions)

#run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
#runmeta = tf.compat.v1.RunMetadata()
model_final.compile(loss='binary_crossentropy',
                   optimizer='adam',
                    metrics=['accuracy'])
                   # options = run_opts,
                   # run_metadata=runmeta)

print(model_final.summary())

model_path = 'ResNet50.h5'
early_stopping = EarlyStopping(patience=5)
model_checkpoint = ModelCheckpoint(model_path,
                                   save_best_only=True, save_weights_only=True)

csv_logger = CSVLogger("ResNet50.csv", append=True)
#name='vgg19_denoise'
tensorboard = TensorBoard(log_dir='ResNet50_log')

def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 :
    return 1e-4

model_final.fit(X_train, y_train,validation_data=(X_test, y_test),
                     epochs=35,batch_size=3, shuffle=True,
                     verbose=2,
                     callbacks=[model_checkpoint, early_stopping,csv_logger,
                          tf.keras.callbacks.LearningRateScheduler(decay),tensorboard])


# report
pred=model_final.predict(X_test)
pred_Y=np.argmax(pred,axis=1)
print("pred:", pred_Y)
print("actual:", y_test)

