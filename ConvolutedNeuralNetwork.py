from keras.models import Sequential
# first step of CNN, which is to build convolution layer(feature detection). We are using 2D since images are 2D unlike videos, which are 3D
#(third dimension be time)
from keras.layers import Convolution2D
# for step 2, which is pooling stepthat will add our pooling layers
from keras.layers import MaxPooling2D
# Step3--> converts all the pooled feature maps that we created using maxpooling into one large feature vector which becomes input of our fully
# connected layer
from keras.layers import Flatten
# Add fully connected layers
from keras.layers import Dense

# Initialise the CNN
classifier = Sequential()

# Step 1 --> Convolution::: Build convolutional layer which has all the feature maps
# add() method is used to add both normal and convolutional layer of several nodes
# we wont use Dense function as it builds fully connected layer
# 1st parameter --> number of feature extractors
# 2nd parameter --> number of rows in feature extractor
# 3rd parameter --> number of columns in feature extractor
# 4th parameter --> dimension of image on which data will be trained..64X64 pixels....3->RGB image.for gray image,its value will be one
# 5th parameter --> we are using activation function to make sure that there is not any negative values in our feature map to have non-linearity
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation="relu"))

# Step 2 --> Pooling
# We use pooling to reduce number of nodes that we will get after flattening step... We dont lose spatial structure of the model as we do max pooling
# thus, the performance is not affected much and we reduce time complexity considerably
# (2,2) --> dimension of pool over which we will take maximum of entries in feature map
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 --> Flattening
# If we directly flatten the image without convolution and pooling, we will lose information about spatial structure of image as we will get information
# only about one particular pixel at a time
# The nodes created after flattening is for one particular image and as we train across images, the nodes value will differ corresponding to 
# different images
classifier.add(Flatten())

# Step 4 --> Full Connection
# output_dim = no. of nodes.... this value comes after experimenting... we need to make sure that the value chosen is neither very high nor very
# low as higher will make it compute intensive and lower value wont make it a good model 
classifier.add(Dense(units=128,activation="relu"))
# adding output layer
classifier.add(Dense(units=1,activation="sigmoid")) 

# Compiling the CNN
# loss function is "binary_crossentropy" since it is logarithmic function which we use in logarithmic classifier and also because we have binary 
# output. If we had more than two outputs, we will use categorical entropy
# last parameter is performance metrics and we will choose most common one, which is """accuracy""" 
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

# Image Augmentation --> It augments our training images. We need a lot of data to find generic pattern in pixels for dogs and cats. 
# Image augmentation will create more images by flipping,rotating and doing other operations on images. Transformations of images are 
# random transformations, so the model will never find the two same image across the batches. So, it reduces overfitting
from keras.preprocessing.image import ImageDataGenerator
# rescale --> same as feature scaling so that every value of pixel is in range [0,1]
# other three parameters are just used to form various type of images to augment the number of training images
#train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
train_datagen = ImageDataGenerator(rescale = 1./255)

# fitting the training set according to CNN model
# 1st parameter --> path to training data
# 2nd parameter --> size of the image which will be input for CNN model
# 3rd parameter --> number of observations after which weights will be adjusted
# 4th parameter --> "binary" since we have only two dependent class - cats and dogs
training_set = train_datagen.flow_from_directory("dataset/training_set", target_size = (64,64), batch_size = 32, class_mode = "binary")
test_set = test_datagen.flow_from_directory("dataset/test_set", target_size = (64,64), batch_size = 32, class_mode = "binary")

# fitting the dataset in CNN model
# 1st parameter --> training set where CNN model will be applied
# 2nd parameter --> no. of images in training set 
# 3rd parameter --> no. of epochs
# 4th parameter --> test set on which validation of CNN model will be done
# 5th parameter --> no. of images in test set
classifier.fit_generator(training_set, steps_per_epoch = 8000//32, epochs = 25, validation_data = test_set, validation_steps = 2000//32)
   