"""
Лабораторная работа №3
вариант 13

Реализовать модель сети Хопфилда с непрерывным состоянием и дискретным временем в асинхронном режиме.

Выполнил
Войткус С.А. гр.121703


"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def load_img(
    path, side
):  # Load and process images into a binary array, where pixels are represented as 1 or -1
    img = Image.open(path)
    img = img.resize((side, side))
    img = img.convert("1")
    img = 2 * np.array(img, int) - 1
    return img.flatten()


def show_array(img_array):  # Visualize images(array)
    side = int(np.sqrt(img_array.shape[0]))
    img_array = img_array.reshape((side, side))
    plt.figure(figsize=(3, 3))
    plt.imshow(img_array)
    plt.axis("off")
    plt.show()


def modify_img(n, img):
    # Случайным образом выбрать половину индексов для изменения
    indices_to_modify = np.random.choice(
        n, size=n // 2, replace=False
    )  # Без повторений
    for i in indices_to_modify:
        img[i] = -1  # Изменить выбранные пиксели на -1
    return img


def delta_projection_update(w, img, learning_rate=0.1):
    outer_product = np.outer(img, img)
    np.fill_diagonal(outer_product, 0)  # Убираем самовеса
    w += learning_rate * outer_product


def train_network(imgs, side, learning_rate=0.1, max_error=0.01):
    n = side * side
    w = np.random.uniform(
        -0.01, 0.01, (n, n)
    )  # Равномерное распределение от -0.01 до 0.01

    epoch = 1
    while True:
        total_error = 0

        for img in imgs:
            predicted_state = np.tanh(np.dot(w, img))
            error = np.mean((img - predicted_state) ** 2)  # Среднеквадратичная ошибка
            total_error += error
            
            delta_projection_update(w, img, learning_rate)

        print(f"{epoch} epoch")
        epoch+=1

        total_error /= len(imgs)  # Средняя ошибка по всем изображениям
        print(f"Total error: {total_error}")

        # Проверка на достижение максимальной ошибки
        if total_error < max_error:
            print("Достигнута максимально допустимая ошибка.")
            break

    return w


def save_weights(w, filename):
    np.save(filename, w)  # Сохранение весов в файл


def load_weights(filename):
    if os.path.exists(filename):
        return np.load(filename)  # Загрузка весов из файла
    else:
        return None


def reconstructed_image(n, w, state, iterations=100):
    previous_state = np.copy(state)  # Инициализация предыдущего состояния
    for iteration in range(iterations):
        for i in range(n):
            sum = np.dot(w[i], state)
            state[i] = np.tanh(sum)

        # Проверка на выход
        if np.array_equal(previous_state, state):
            print("Состояние стабилизировалось.")
            break
        print(f"{iteration +1} iteration")
        previous_state = np.copy(state)  # Обновление предыдущего состояния

    return state


# Параметры
side = 64
n = side * side
weights_file = "weights.npy"  # Имя файла для хранения весов

# Загрузка изображений
imgs = [load_img(f"{i}.jpg", side) for i in range(1, 4)]


# Проверка на наличие файла с весами
w = load_weights(weights_file)
if w is None:
    # Если файл не найден, обучаем сеть и сохраняем веса
    w = train_network(imgs, side, learning_rate=0.08, max_error=3.033e-1)
    save_weights(w, weights_file)
else:
    print("Weights loaded from file.")

# Инициализация состояния
# state = np.random.choice([-1, 1], size=n)  # случайные пиксели
state = modify_img(n, load_img("1.jpg", side))
#state =  load_img("4.jpg", side)
print("init state:")
show_array(state)

# Реконструкция
state = reconstructed_image(n, w, state)
print(state)
print("reconstructed image:")
show_array(state)
