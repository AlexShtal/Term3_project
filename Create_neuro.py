from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras import utils

# Импорт датасета для обучения
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)

x_train = x_train / 255

y_train = utils.to_categorical(y_train, 10)

# Создание и обучение модели
model = Sequential([Dense(800, input_dim=784, activation="relu"),
                    Dense(10, activation="softmax")])

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())

history = model.fit(x_train, y_train,
                    batch_size=200,
                    epochs=100,
                    validation_split=0.2,
                    verbose=1)

# Сохранение модели
model.save("mnist_dense.h5")
