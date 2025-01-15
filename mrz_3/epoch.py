import random
import numpy as np
import math


# Функции активации
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


# Последовательности для прогнозирования
def factorial(n):
    if n < 0:
        return None
    return 1 if n == 0 or n == 1 else n * factorial(n - 1)


def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fib(n - 1) + fib(n - 2)


def arithmetic_prog(n, a=1, d=2):
    return a + n * d


def geometric_prog(n, a=1, r=2):
    return a * (r**n)


def pow_fun(n):
    return 1.125**n


def fact_fun(n):
    return math.log(factorial(n)) / math.log(10)


# Выбор последовательностей
sequences = {
    "Fibonacci": lambda n: fib(n),
    "Factorial log_10": lambda n: fact_fun(n),
    "Arithmetic progression": lambda n: arithmetic_prog(n),
    "Geometric progression": lambda n: geometric_prog(n),
    "Power function": lambda n: pow_fun(n),
}


def initialize_weights(input_size, output_size):
    return np.random.uniform(-0.1, 0.1, (input_size, output_size))


# Функция для расчета MAPE
def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# Функция для расчета MSE
def calculate_mse(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean((actual - predicted) ** 2)


# Пользовательская настройка
if __name__ == "__main__":
    # Параметры сети
    print("Enter the network parameters:")
    window_size = int(input("Sliding window size (p): "))
    hidden_layer_size = int(input("Hidden layer size: "))
    context_layer_size = int(input("Context layer size: "))
    output_layer_size = int(
        input("Number of effector neurons (output size): ")
    )  # Запрос количества эффекторных нейронов
    leaky_relu_alpha = float(input("Leaky ReLU alpha: "))
    reset_context = input("Reset context neurons at each epoch? (Y/N): ").lower() == "y"

    # Проверка последовательностей
    print("Available sequences:")
    for i, seq_name in enumerate(sequences.keys()):
        print(f"{i + 1}) {seq_name}")

    choice = int(input("Select sequence (1-5): "))
    choice = max(1, min(choice, len(sequences)))
    selected_sequence_name = list(sequences.keys())[choice - 1]
    sequence_func = sequences[selected_sequence_name]

    print(f"Selected sequence: {selected_sequence_name}")

    # Генерация последовательности
    k = 20
    current_sequence = [sequence_func(i) for i in range(k)]
    if len(current_sequence) < 10:
        raise ValueError("The sequence must contain at least 10 elements.")

    # Подготовка данных
    train_matrix = np.array(
        [
            current_sequence[i : i + window_size]
            for i in range(
                len(current_sequence) - window_size - (output_layer_size - 1)
            )
        ]
    )
    train_etalons = np.array(
        [
            current_sequence[i + window_size : i + window_size + output_layer_size]
            for i in range(
                len(current_sequence) - window_size - (output_layer_size - 1)
            )
        ]
    )

    # Инициализация параметров
    b_first = np.zeros((1, hidden_layer_size))
    b_second = np.zeros((1, output_layer_size))
    W_first = initialize_weights(window_size, hidden_layer_size)
    W_second = initialize_weights(hidden_layer_size, output_layer_size)
    context_weights = initialize_weights(context_layer_size, hidden_layer_size)

    learning_rate = 0.00001  # Уменьшенный шаг обучения
    max_gradient_norm = 1.0  # Ограничение градиента
    epochs = int(input("Enter the number of epochs for training: "))
    eps = 1e-8

    # Тренировка
    epoch = 0
    while epoch < epochs:
        epoch += 1
        error = 0
        previous_context = np.zeros((1, context_layer_size))

        for input_sequence, target_sequence in zip(train_matrix, train_etalons):
            input_sequence = input_sequence.reshape(1, -1)

            hidden = leaky_relu(
                input_sequence @ W_first + previous_context @ context_weights + b_first,
                leaky_relu_alpha,
            )
            output = hidden @ W_second + b_second

            delta = output - target_sequence.reshape(1, -1)
            error += np.sum(delta**2)

            # Обратное распространение ошибки
            grad_output = delta
            grad_hidden = grad_output @ W_second.T * (hidden > 0).astype(float)

            # Ограничение градиентов
            grad_output_norm = np.linalg.norm(grad_output)
            grad_hidden_norm = np.linalg.norm(grad_hidden)
            if grad_output_norm > max_gradient_norm:
                grad_output *= max_gradient_norm / grad_output_norm
            if grad_hidden_norm > max_gradient_norm:
                grad_hidden *= max_gradient_norm / grad_hidden_norm

            # Обновление весов
            W_second -= learning_rate * hidden.T @ grad_output
            b_second -= learning_rate * grad_output.sum(axis=0)

            W_first -= learning_rate * input_sequence.T @ grad_hidden
            b_first -= learning_rate * grad_hidden.sum(axis=0)

            context_weights -= learning_rate * previous_context.T @ grad_hidden

            previous_context = (
                hidden if not reset_context else np.zeros_like(previous_context)
            )

        # Вычисление MSE на текущем шаге
        mse = error / len(train_matrix)
        print(f"Epoch {epoch}, MSE: {mse:.4f}")

    # Тестирование
    while True:
        start = random.randint(0, k - window_size - output_layer_size)
        test_sequence = current_sequence[start : start + window_size]
        expected_output = current_sequence[
            start + window_size : start + window_size + output_layer_size
        ]

        test_input = np.array(test_sequence).reshape(1, -1)
        hidden = leaky_relu(
            test_input @ W_first + previous_context @ context_weights + b_first,
            leaky_relu_alpha,
        )
        output = hidden @ W_second + b_second

        print("Test input:", test_sequence)
        print("Expected output:", expected_output)
        print("Predicted output:", output.flatten())

        mape = calculate_mape(expected_output, output.flatten())
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

        if input("Do you want to test again? (Y/N): ").lower() == "n":
            break
