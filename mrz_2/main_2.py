import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from multiprocessing import Pool


def load_img(path, side):
    img = Image.open(path)
    img = img.resize((side, side))
    img = img.convert("L")  # Грейскейл
    img = np.array(img, float) / 255.0  # Нормализация
    return img.flatten()


def show_array(img_array):  # Visualize images(array)
    side = int(np.sqrt(img_array.shape[0]))
    img_array = img_array.reshape((side, side))
    plt.figure(figsize=(3, 3))
    plt.imshow(img_array)
    plt.axis("off")
    plt.show()


def modify_img(
    n, img
):  # Introduce noise or modifications to an image, for testing the network’s ability to reconstruct it
    # Make 50% of image negative
    for i in range(n):
        if i > n / 2 - 1:
            img[i] = -1
    return img


def delta_projection_update(w, img, learning_rate=0.1):
    outer_product = np.outer(img, img)
    np.fill_diagonal(outer_product, 0)
    return w + learning_rate * outer_product

def parallel_update(args):
    w, img, learning_rate = args
    return delta_projection_update(w, img, learning_rate)

def delta_projection_update_batch(w, imgs, learning_rate=0.1):
    with Pool() as pool:
        updates = pool.map(parallel_update, [(w, img, learning_rate) for img in imgs])
    for update in updates:
        w += update

def train_network(imgs, side, learning_rate=0.1, epochs=100):
    n = side * side
    w = np.random.uniform(-0.01, 0.01, (n, n))

    for epoch in range(epochs):
        delta_projection_update_batch(w, imgs, learning_rate)
        print(f"{epoch + 1} epoch completed.")

    return w


def save_weights(w, filename):
    np.save(filename, w)  # Сохранение весов в файл


def load_weights(filename):
    if os.path.exists(filename):
        return np.load(filename)  # Загрузка весов из файла
    else:
        return None


def reconstructed_image(n, w, state, iterations=5):
    previous_state = np.copy(state)  # Инициализация предыдущего состояния
    for _ in range(iterations):
        for i in range(n):
            sum = np.dot(w[i], state)
            state[i] = np.tanh(sum)

        # Проверка на выход
        if np.array_equal(previous_state, state):
            print("Состояние стабилизировалось.")
            break
        previous_state = np.copy(state)  # Обновление предыдущего состояния
    return state


if __name__ == '__main__':
    # Параметры
    side = 100
    n = side * side
    weights_file = "weights.npy"  # Имя файла для хранения весов

    # Загрузка изображений
    imgs = [load_img(f"{i}.jpg", side) for i in range(1, 4)]

    # Проверка на наличие файла с весами
    w = load_weights(weights_file)
    if w is None:
        # Если файл не найден, обучаем сеть и сохраняем веса
        w = train_network(imgs, side, learning_rate=0.1, epochs=20)
        save_weights(w, weights_file)
    else:
        print("Weights loaded from file.")

    # Инициализация состояния
    state = np.random.choice([-1, 1], size=n)  # случайные пиксели
    state = modify_img(n, load_img("2.jpeg", side))
    print("init state:")
    show_array(state)

    # Реконструкция
    state = reconstructed_image(n, w, state)
    print("reconstructed image:")
    show_array(state)
