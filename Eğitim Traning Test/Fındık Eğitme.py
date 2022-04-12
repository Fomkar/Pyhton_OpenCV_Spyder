# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:34:43 2022

@author: Gitek_Micro
"""

import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import cv2
from random import shuffle
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


tf.compat.v1.disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_label(folder):
    #buradaki folderlar sınıfların adı olacak ve etiketler bunlara göre verilecek
    if folder == "kabuklu":
        return np.array([1,0,0])
    elif folder == "ic":
        return np.array([0,1,0])
    elif folder == "yarim":
        return np.array([0,0,1])
    

    
#train_or_test_folder_name = test ya da train ya da validation klasörünün sadece ismi
#dataset_type_name = open ya da close olan klasörün ismi
def dataset_load(train_or_test_folder_name, start_index=None, end_index=None):
    
    #burası train, test, validate klasörlerinin bir üstü 
    base_dir = 'C:\\Users\\Gitek_Micro\\Desktop\\6285_yarim_eklendi\\dataset_for_train\\'
    
    dataset_load = []
    new_dataset = []
    path_dataset = base_dir + train_or_test_folder_name
    i = 0
    for folder in tqdm(os.listdir(path_dataset)): 
        for image in tqdm(os.listdir(path_dataset+"/"+folder)):
            label_image = create_label(folder)  #folder değişkeninde klasör adından gelen sınıf ismi olur.
            path_image = os.path.join(path_dataset+'/'+folder, image)
            image_data = cv2.imdecode(np.fromfile(path_image, np.uint8), -1) # klasörden resim okuma fonksiyonu
            image_data = cv2.resize(image_data, (227, 227))     #resmi alexnet modeli için 227ye resize eder
            dataset_load.append([np.array(image_data), np.array(label_image)])
            i = i + 1
    new_dataset = dataset_load[start_index : end_index]
    shuffle(new_dataset)
    
    return new_dataset


#içeriye yazılan değer dataset_load fonksiyonu içerisindeki base_dir pathinin son kısmına eklenecek ve o klasör okunacak.
train = dataset_load('Train')
validate = dataset_load('Validate')
test = dataset_load('Test')

#okunan verileri image ve label olarak ayırıyoruz.
images_train = np.array([i[0] for i in train])
label_train = [i[1] for i in train]
images_test = np.array([i[0] for i in test])
label_test = [i[1] for i in test]
images_validate = np.array([i[0] for i in validate])
label_validate = [i[1] for i in validate]

#gray veya siyah beyaz görüntü için aşağısı açılacak
# images_train = images_train.reshape(images_train.shape[0], images_train.shape[1], images_train.shape[2], 1)
# images_test = images_test.reshape(images_test.shape[0], images_test.shape[1], images_test.shape[2], 1)
# images_validate = images_validate.reshape(images_validate.shape[0], images_validate.shape[1], images_validate.shape[2], 1)

# alexnet model
model = tf.keras.models.Sequential([
    
    #layer 1
    tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(3, strides=(2, 2)),
    
    #layer 2
    tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(3, strides=(2, 2)),
    
    #layer3
    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),

    #loyer 4
    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),

    #layer 5    
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(3, strides=(2, 2)),
    
    #flatten layer öncesi düzenleme
    tf.keras.layers.Flatten(),
    
    #flatten layer 1
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.8),
    
    #flatten layer 2
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dropout(0.8),
    
    tf.keras.layers.Dense(3, activation='softmax')
    
    ])

#burası early stop fonksiyonu, overfitting olmaması için kullanılabilir.
callback = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=20, restore_best_weights=True)] #min delta değişim olması gereken değer

#optimizer = SGD(lr=1e-4, decay=1e-06, momentum=0.9)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-06, decay=1e-05)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

start_train = datetime.now()

history = model.fit(np.array(images_train), np.array(label_train), batch_size=64, epochs=150, shuffle=True, validation_data=(np.array(images_validate), np.array(label_validate)))
          
                                                                                                    
                                                                                                  
end_train = datetime.now()

#burası eğitim sonrası doğruluk değerlerini grafik olarak gösterecek
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('model accuracy and loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train acc', 'validation_acc', 'val_loss', 'train_loss'], loc=0)
plt.show()

print("toplam eğitim süresi : {}".format(end_train-start_train))

# start_test = datetime.now()
# result = model.evaluate(np.array(images_test), np.array(label_test), verbose=0, batch_size=8)
# #model.save("AF_endüstri_data_model.h5")
# print("test loss : ", result[0])
# print("test accuracy : ", result[1])


#burası sadece confusion matrix oluşturmak için.
predict = model.predict(images_test, batch_size=8)
y_predict = (predict > 0.5)
labels=[[0,1], [1,0]]
cm = confusion_matrix(np.argmax(label_test, 1), np.argmax(y_predict, 1))
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title("confusion matrix classifier")
fig.colorbar(cax)
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True Labels")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# target_names = ["closed_AF", "open_AF"]
# print(classification_report(np.argmax(label_test, 1), np.argmax(y_predict, 1), target_names=target_names))

# end_test = datetime.now()
# print("Toplam Test süresi : {}".format(end_test-start_test))



#model.summary()

#%% burada tüm test verileri için doğru ve yanlış tahminlediklerini plot ile gösterir

from PIL import Image
false_images = []
false_predicts = []
false_images_labels = []
#start_test = datetime.now()
result = model.evaluate(np.array(images_test), np.array(label_test), verbose=0, batch_size=8)
print("test loss : ", result[0])
print("test accuracy : ", result[1])
predict_inception = model.predict(images_test, batch_size=8)

argmax_inception = np.argmax(predict_inception, 1)
argmax_label = np.argmax(label_test, 1)

#buradaki range içerisindeki sayı test klasöründeki toplam veri sayısı olmalı
for i in range(0,942):
    if argmax_inception[i] != argmax_label[i]:
        predict_label = predict_inception[i]
        false_images.append(images_test[i])
        false_images_labels.append(label_test[i])
        false_predicts.append(predict_label)
        
# image = Image.fromarray(false_images[0], )
# image.show()
# false_images[4] = cv2.cvtColor(false_images[4], cv2.COLOR_BGR2RGB)
# plt.imshow(false_images[4], interpolation='nearest')
# plt.show()
# print("Gerçek etiketi : ", false_images_labels[4], " - tahmin edilen : ", false_predicts[4])

for i in range(0, len(false_images)):
    false_image = cv2.cvtColor(false_images[i], cv2.COLOR_BGR2RGB)
    plt.imshow(false_image, interpolation='nearest')
    plt.show()
    print("Gerçek etiketi : ", false_images_labels[i], " - tahmin edilen : ", false_predicts[i])
