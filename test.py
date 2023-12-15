import keras.models
from keras.datasets import mnist
from keras import utils
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Оценка качества прогнозов

# Загрузка тестового датасета и подготовка его к использованию
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 784) / 255

y_test = utils.to_categorical(y_test, 10)

# Загрузка сохраненной модели
model = keras.models.load_model("mnist_dense.h5")

# Вычисление точности
scores = model.evaluate(x_test, y_test, verbose=0)

# Получение предсказаний для тестовых данных
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Вывод случайной выборки из предсказаний
fig = plt.figure(figsize=(12, 8))
fig.suptitle("Accuracy: " + str(round(scores[1] * 100, 4)) + "%")
grid = fig.add_gridspec(5, 10, left=0.025, bottom=0.075, right=0.975, top=0.915)

for i in range(0, 50):
    choose = randint(0, 10000)
    ax = fig.add_subplot(grid[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.imshow(x_test[choose].reshape(28, 28), cmap=plt.cm.binary)

    prediction = y_pred[choose]
    if prediction != np.argmax(y_test[choose]):
        ax.xaxis.label.set_color('red')
    else:
        ax.xaxis.label.set_color('green')
    ax.set_xlabel("prediction=" + str(prediction) + '\nactual=' + str(np.argmax(y_test[choose])))

# Получение метрик
precision, recall, f1_score, _ = precision_recall_fscore_support(np.argmax(y_test, axis=1), y_pred, average=None)

# Вывод метрик

# Confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', fontsize=12, color='black')
ax.set_ylabel('True outputs', fontsize=12, color='black')
ax.xaxis.set(ticks=range(10))
ax.yaxis.set(ticks=range(10))
ax.set_xticklabels(range(10))
ax.set_yticklabels(range(10))
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
ax.set_title('Confusion Matrix')
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cax)
cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")

# Вывод precision, recall и f1
fig, ax = plt.subplots(figsize=(8, 6))
ax.grid(True)
ax.set_title("Metrics per class")
ax.set_xlabel("Class")
ax.set_ylabel("Score")
ax.set_xticks(range(10))

ax.plot(range(10), precision, "-o", label="Precision")
ax.plot(range(10), recall, "-o", label="Recall")
ax.plot(range(10), f1_score, "-o", label="f1")

ax.legend(loc="lower left")
plt.show()
