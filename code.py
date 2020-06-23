import tensorflow as tf

print(tf.__version__)

dataset = tf.keras.datasets.fashion_mnist

DESIRED_ACCURACY = 0.99

class myCallBack(tf.model.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc')>=DESIRED_ACCURACY:
            print("\nReached "+ str(DESIRED_ACCURACY)+" so cancelling training.\n")
            self.model.stop_training = True
            

callbacks = myCallBack()    #instantiating the class myCallBack

(training_data,training_labels), (test_data,test_labels) = dataset.load()   #loading the data from our data set

#Reshape the data so it's in the form of a 4D list of dimensions 60000*28*28*1 (60000 images, width = 28, height= 28, colour info = 1(greyscale))
training_data = training_data.reshape(60000,28,28,1)
training_data = training_data/255.0

#Reshape the test data
test_data = test_data.reshape(10000,28,28,1)
test_data = test_data/255.0

#defining a model
model = tf.keras.model.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten()
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Summary provides us an overview of how the layers are manipulating the image and its parameters
model.summary()
#fitting the model with callbacks
model.fit(training_data, training_labels, epochs = 15, callbacks = [callbacks])

test_loss, test_acc = model.evaluate(test_data,test_labels)
print("The accuracy of the model on test data is: " + str(test_acc))
