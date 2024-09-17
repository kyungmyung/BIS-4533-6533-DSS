import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# image loading package
from PIL import Image

mnist_data = tf.keras.datasets.mnist
(x_train, y_train) , (x_test, y_test) = mnist_data.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

def image_check(n):
    plt.imshow(n, cmap=plt.cm.binary)
    plt.show()

image_check(x_train[1])

# NN model creation
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(x_train, y_train, epochs=10)


# Image change
def load_image(image_path):
    img = Image.open(image_path).convert('L')  # make it to black color
    img = img.resize((28, 28))  # 28x28
    img = np.array(img)
    img = 255 - img  # MNIST white background and black font
    img = img / 255.0  # normalization of the digit

    return img

# image load
image_path = r'C:\Users\kyungmyung\Downloads\pic.jpeg'  # image path
my_image = load_image(image_path)

# Prediction using deep learning model!
my_image_transfered = my_image[np.newaxis, ..., np.newaxis]  # Dimension (1, 28, 28, 1)
predicted_label = np.argmax(model.predict(my_image_transfered))
print(predicted_label)
#image_check(my_image)

# Add your personal handwriting data to the Mnist Data set
# add the data
new_image = my_image.reshape(1, 28, 28)
x_train = np.concatenate([x_train, new_image], axis=0)  # x_train add

# y_train add label 0
new_label = np.array([9])  # add label 9 to the y_train data set
y_train = np.concatenate([y_train, new_label], axis=0)  # y_train add

# NN model creation
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(x_train, y_train, epochs=10)

# image load
image_path = r'C:\Users\kyungmyung\Downloads\pic.jpeg'  # image path
my_image = load_image(image_path)
# Prediction using deep learning model!
my_image_transfered = my_image[np.newaxis, ..., np.newaxis]  # Dimension (1, 28, 28, 1)
predicted_label = np.argmax(model.predict(my_image_transfered))
print(predicted_label)























# Minist 28x28 pixels and image coloar = black so. channel is 1
# dimensions will be 28 x 28 x 1
# Batch size == 1
# np.newaxis is add one more dimension and ... is the automatic dimension corrections.
single_image = x_test[0][np.newaxis, ...]

# 모델로 예측
prediction = model.predict(single_image)

# 예측된 값에서 가장 높은 확률의 클래스 추출
predicted_label = np.argmax(prediction)


# Check accuracy and loss..
val_loss,val_acc = model.evaluate(x_test,y_test)

print("loss-> ",val_loss,"\nacc-> ",val_acc)


predictions = model.predict([x_test])
print('label -> ', y_test[0])
print('prediction -> ', np.argmax(predictions[0]))

image_check(x_test[2])

model.save('my_first_classification_model.h5')

model_restored = tf.keras.models.load_model('my_first_classification_model.h5')

# NN model creation
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(x_train, y_train, epochs=10)

# image load
image_path = r'C:\Users\kyungmyung\Downloads\pic.jpeg'  # image path
my_image = load_image(image_path)
# Prediction using deep learning model!
my_image_transfered = my_image[np.newaxis, ..., np.newaxis]  # Dimension (1, 28, 28, 1)
predicted_label = np.argmax(model.predict(my_image_transfered))
print(predicted_label)


