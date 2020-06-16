from keras.models import Sequential # initialize neural network
from keras.layers import Convolution2D # making CNN to deal with image for video we use 3d
from keras.layers import MaxPooling2D # for proceed poooling step
from keras.layers import Flatten #convert pool to feature 
from keras.layers import Dense #create and connect nn

import time
start=time.time()#to check the time taken of the process
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator#data augmentation

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/shalo/Desktop/ML stuffs/datasets/corona/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:/Users/shalo/Desktop/ML stuffs/datasets/corona/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
print(training_set.class_indices)#give the indices of all classes


from keras.callbacks import EarlyStopping, ModelCheckpoint

                     
checkpoint = ModelCheckpoint("maskNOmask_classifier_checkpoint.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)
'''
earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement
                          min_delta = 0, #Abs value and is the min change required before we stop
                          patience = 3, #Number of epochs we wait before stopping 
                          verbose = 3, #quality of data
                          restore_best_weights = True) #keeps the best weigths once stopped 
'''
# we put our call backs into a callback list
#callbacks = [earlystop,checkpoint]
callbacks = [checkpoint]#we can use both early stopping as well as modelcheckpoint method 
classifier.fit_generator(training_set,
                         samples_per_epoch = 552,#all images in train dataset
                         nb_epoch = 50,
                         callbacks=callbacks,
                         validation_data = test_set,
                         nb_val_samples = 276)#all images in test dataset
#prediction

#classifier.save('maskNOmask_classifier1.h5')#saving the model

stop=time.time()

print("time taken ", (stop-start)," s")

import matplotlib.pyplot as plt

# from IPython.display import Inline

#visualising the performance
plt.plot(classifier.history.history['accuracy'])
plt.plot(classifier.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(classifier.history.history['loss'])
plt.plot(classifier.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



#testing evaluating the model
from tensorflow.keras.models import load_model
classifier = load_model('C:/Users/shalo/Desktop/ML stuffs/projects/corona/maskNOmask_classifier_checkpoint95%.h5')

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

img1 = image.load_img('C:/Users/shalo/Desktop/ML stuffs/projects/corona/normal3.jpg', target_size=(64, 64))
img = image.img_to_array(img1)
img = img/255
# create a batch of size 1 [N,H,W,C]
img = np.expand_dims(img, axis=0)
prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
print(prediction)
#use hit and trail method by using images of diffrent classes and get the probability
if(prediction[:,:]>0.65):
    value ='Not wearing mask :%1.2f'%(prediction[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
else:
    value ='Wearing mask :%1.2f'%(1.0-prediction[0,0])
    plt.text(20, 62,value,color='blue',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))

plt.imshow(img1)
plt.show()