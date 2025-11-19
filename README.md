# BaumanEDUcnnFrame

BaumanEDUcnnFrame — это образовательный проект, написанный с нуля на C++17, который демонстрирует ключевые концепции глубокого обучения. Он включает в себя библиотеку для создания сверточных нейронных сетей (CNN) и агента для обучения с подкреплением (Reinforcement Learning) на примере классической игры "Змейка".

## Возможности

- **Библиотека CNN (`edunet`)**:
  - **Тензорная структура**: базовый класс `Tensor` для многомерных вычислений.
  - **Модульная архитектура**: легко комбинируемые слои, включая `Conv2DLayer`, `MaxPooling2DLayer`, `DenseLayer`, `FlattenLayer`, `DropoutLayer`.
  - **Функции активации**: `ReLULayer`, `SoftmaxLayer`, `SigmoidLayer`.
  - **Модель `Sequential`**: простой и интуитивно понятный способ создания нейросетей путем последовательного добавления слоев.
  - **Оптимизаторы**: реализованы `Adam` и `SGD` для эффективного обновления весов.
  - **Функции потерь**: `CrossEntropyLoss` (для классификации) и `MeanSquaredError` (для регрессии).
  - **Тренировочный цикл**: класс `Trainer` для удобного обучения и валидации моделей.
  - **Загрузчик данных**: `MNISTDataLoader` для работы с набором данных MNIST.
  - **Сохранение/Загрузка**: возможность сохранять обученные модели и загружать их для дальнейшего использования.

- **Агент для "Змейки" (Reinforcement Learning)**:
  - **DQN Агент**: реализация алгоритма Deep Q-Network (`DQNAgent`) для принятия решений.
  - **Интерактивная среда**: классическая игра "Змейка" (`snake.hpp`), адаптированная для обучения агента и визуализации в терминале с помощью `ncurses`.
  - **Обучение и Визуализация**: режимы для обучения агента (сбор опыта и обновление модели) и для наблюдения за его игрой в реальном времени.

## Структура проекта

```
.
├── CMakeLists.txt          # Главный файл сборки CMake
├── README.md               # Документация проекта
├── edunet/                 # Исходный код библиотеки нейронной сети
│   ├── Tensor.h            # Базовый класс тензора
│   ├── Layer.h             # Абстрактный класс слоя
│   ├── Sequential.h        # Класс для последовательной модели
│   ├── Conv2DLayer.h       # Сверточный слой
│   ├── DenseLayer.h        # Полносвязный слой
│   ├── ...                 # Другие слои, оптимизаторы, функции потерь
│   └── DQNAgent.h          # Агент для обучения с подкреплением
└── demonstration model/    # Демонстрационное приложение
    ├── main.cpp            # Главное меню и логика запуска
    ├── mnist_app.cpp/.h    # Логика для обучения и тестирования на MNIST
    ├── snake.hpp           # Реализация игры "Змейка"
    └── mnist/              # Данные MNIST
```

## Начало работы

### Требования

- **CMake** (версия 3.10 или выше)
- **Компилятор C++17** (например, GCC или Clang)
- **Библиотека `ncurses`** (для визуализации "Змейки")

На Ubuntu/Debian `ncurses` можно установить командой:
```bash
sudo apt-get update && sudo apt-get install libncurses-dev
```

### Сборка и запуск

1.  **Клонируйте репозиторий:**
    ```bash
    git clone <URL-репозитория>
    cd BaumanEDUcnnFrame
    ```

2.  **Создайте директорию для сборки:**
    ```bash
    mkdir build && cd build
    ```

3.  **Сконфигурируйте проект с помощью CMake:**
    ```bash
    cmake ..
    ```

4.  **Скомпилируйте проект:**
    ```bash
    make
    ```

5.  **Запустите демонстрационное приложение:**
    ```bash
    ./cnn_demo
    ```

После запуска вы увидите меню, где можно выбрать обучение или тестирование модели для MNIST, а также обучение или визуализацию агента для игры в "Змейку".

## Документация библиотеки `edunet`

### `Tensor`
Основная структура данных для всех операций. Представляет собой многомерный массив.

- **Создание:** `Tensor({dim1, dim2, ...})`
- **Доступ к элементам:** `tensor.at({idx1, idx2, ...})` или `tensor.at(n, c, h, w)` для 4D-тензоров.

### `Sequential`
Контейнер для слоев, который позволяет строить модель последовательно.

```cpp
#include "Sequential.h"
#include "DenseLayer.h"

Sequential model;
model.add(std::make_unique<DenseLayer>(784, 128)); // Добавление полносвязного слоя
model.add(std::make_unique<ReLULayer>());         // Добавление функции активации
```

- `forward(input)`: выполняет прямое распространение сигнала через все слои.
- `backward(gradient)`: выполняет обратное распространение ошибки.
- `summary()`: выводит информацию о слоях модели.

### Слои (`Layer`)

Все слои наследуются от базового класса `Layer` и реализуют методы `forward` и `backward`.

- **`Conv2DLayer(in_channels, out_channels, kernel_size, stride, padding)`**: 2D сверточный слой.
- **`MaxPooling2DLayer(pool_size, stride)`**: слой 2D-субдискретизации (max pooling).
- **`DenseLayer(input_size, output_size)`**: полносвязный слой.
- **`FlattenLayer()`**: преобразует многомерный тензор в 1D-вектор.
- **`DropoutLayer(rate)`**: слой регуляризации для предотвращения переобучения.

### Функции активации

Представлены в виде слоев.

- **`ReLULayer()`**: функция активации ReLU.
- **`SoftmaxLayer()`**: функция Softmax для задач многоклассовой классификации.
- **`SigmoidLayer()`**: функция активации Sigmoid.

### Функции потерь (`Loss`)

- **`CrossEntropyLoss`**: для задач классификации.
- **`MeanSquaredError`**: для задач регрессии.

### Оптимизаторы (`Optimizer`)

- **`Adam(learning_rate, beta1, beta2, epsilon)`**: оптимизатор Adam.
- **`SGD(learning_rate)`**: стохастический градиентный спуск.

### `Trainer`
Класс, который инкапсулирует логику обучения и валидации модели.

```cpp
#include "Trainer.h"
#include "Optimizer.h"
#include "Loss.h"

// ... создание модели ...
CrossEntropyLoss loss_fn;
auto optimizer = std::make_unique<Adam>(0.001f);
Trainer trainer(model, std::move(optimizer), loss_fn);

// Запуск обучения
trainer.fit(X_train, y_train, X_val, y_val, epochs, batch_size);
```

## Примеры использования

### CNN для классификации MNIST

Пример создания, обучения и тестирования модели LeNet-подобной архитектуры на данных MNIST.

```cpp
// Из demonstration model/mnist_app.cpp

// 1. Создание модели
Sequential model;
model.add(std::make_unique<Conv2DLayer>(1, 6, 5));      // Вход: 1x28x28, Выход: 6x24x24
model.add(std::make_unique<ReLULayer>());
model.add(std::make_unique<MaxPooling2DLayer>(2));     // Выход: 6x12x12
model.add(std::make_unique<Conv2DLayer>(6, 16, 5));     // Выход: 16x8x8
model.add(std::make_unique<ReLULayer>());
model.add(std::make_unique<MaxPooling2DLayer>(2));     // Выход: 16x4x4
model.add(std::make_unique<FlattenLayer>());           // Выход: 256
model.add(std::make_unique<DenseLayer>(256, 120));
model.add(std::make_unique<ReLULayer>());
model.add(std::make_unique<DenseLayer>(120, 84));
model.add(std::make_unique<ReLULayer>());
model.add(std::make_unique<DenseLayer>(84, 10));
model.add(std::make_unique<SoftmaxLayer>());

// 2. Настройка обучения
CrossEntropyLoss loss_fn;
auto optimizer = std::make_unique<Adam>(0.001f);
Trainer trainer(model, std::move(optimizer), loss_fn);

// 3. Загрузка данных
auto X_train = MNISTDataLoader::load_images("path/to/train-images");
auto y_train = MNISTDataLoader::load_labels("path/to/train-labels");
// ... загрузка данных для валидации ...

// 4. Обучение
trainer.fit(X_train, y_train, X_val, y_val, /*epochs=*/5, /*batch_size=*/64);
```

### DQN Агент для "Змейки"

Пример обучения агента с подкреплением.

```cpp
// Из demonstration model/main.cpp

const int STATE_SIZE = 8;  // Размер вектора состояния игры
const int ACTION_SIZE = 3; // 3 действия: вперед, повернуть налево, повернуть направо

// 1. Инициализация игры и агента
SnakeGame game;
DQNAgent agent(STATE_SIZE, ACTION_SIZE);

// 2. Цикл обучения
for (int episode = 0; episode < num_episodes; ++episode) {
    game.reset();
    auto state = game.get_state<float>(); // Получаем начальное состояние

    while (!game.is_over()) {
        // 3. Агент выбирает действие
        int action = agent.act(state);

        // 4. Среда выполняет действие и возвращает результат
        game.update_direction(action);
        game.update_without_render();
        bool done = game.is_over();
        auto next_state = game.get_state<float>();
        
        // 5. Рассчитываем награду
        float reward = calculate_reward(...);

        // 6. Сохраняем переход (state, action, reward, next_state) в память
        agent.remember(state, action, reward, next_state, done);
        state = next_state;
    }

    // 7. Обучаем модель на случайной выборке из памяти
    agent.replay(/*batch_size=*/32);
}
```

## Сохранение и загрузка моделей

Модели, созданные с помощью `Sequential`, можно легко сохранять и загружать.

```cpp
// Сохранение модели
model.save_model("my_model.bin");

// Загрузка модели
Sequential loaded_model;
// Важно: архитектура должна быть создана перед загрузкой весов
loaded_model.add(...); 
// ... (воссоздать ту же архитектуру)
loaded_model.load_model("my_model.bin");
```
Агент DQN также поддерживает сохранение и загрузку весов своей внутренней нейросети:
```cpp
// Сохранение весов агента
agent.save("snake_agent_weights.bin");

// Загрузка весов агента
agent.load("snake_agent_weights.bin");
```