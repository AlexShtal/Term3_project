from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab
import numpy as np

from default_image import default_img

model = load_model('mnist_dense.h5')


def predict_digit(img):
    # изменение рзмера изобржений на 28x28
    img = img.resize((28, 28))
    # конвертируем rgb в grayscale
    img = img.convert('L')
    img = np.array(img)
    img = img - default_img
    # print(img.reshape(1, 784)[0])
    if all(a == 0 for a in img.reshape(1, 784)[0]):
        return -1, None
    # изменение размерности для поддержки модели ввода и нормализации
    img = img.reshape(1, 784)
    img = abs(img / 255.0)
    # предсказание цифры
    res = model.predict(img)[0]
    return np.argmax(res), max(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        self.title("Определитель цифр")
        # Создание элементов
        self.canvas = tk.Canvas(self, width=304, height=304, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Ожидание\nввода...", font=("Times new roman", 64))
        self.classify_btn = tk.Button(self, text="Распознать", font=("Times new roman", 18),
                                      command=self.classify_handwriting, background="lightgreen", width=20, height=2)
        self.button_clear = tk.Button(self, text="Очистить", font=("Times new roman", 18), command=self.clear_all,
                                      background="lightgreen", width=20, height=2)

        # Сетка окна
        self.canvas.grid(row=0, column=0, pady=15, sticky=W, padx=15)
        self.label.grid(row=0, column=1, pady=15, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2, padx=2)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text="Ожидание\nввода...")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)  # получаем координату холста
        im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im)
        if digit == -1:
            self.label.configure(text="Ошибка:\nпустое поле\nввода")
        else:
            self.label.configure(text="Результат: " + str(digit) + '\nВероятность: ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 20
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


app = App()
app.mainloop()
