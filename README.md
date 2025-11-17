# BaumanEDUcnnFrame

BaumanEDUcnnFrame — это написанная с нуля на C++17 библиотека для создания и обучения сверточных нейронных сетей (CNN). Проект демонстрирует ключевые концепции глубокого обучения, включая различные типы слоев, функции потерь и оптимизаторы.

## Возможности

- **Тензорная библиотека**: базовая структура данных `Tensor` для многомерных массивов.
- **Модульная архитектура слоев**:
  - `Conv2DLayer` (2D свертка)
  - `MaxPooling2DLayer` (2D макс-пулинг)
  - `DenseLayer` (полносвязный слой)
  - `FlattenLayer` (выравнивание данных)
  - `ReLULayer`, `SigmoidLayer`, `SoftmaxLayer` (функции активации)
  - `DropoutLayer` (дропаут для регуляризации)
- **Модель `Sequential`**: простой способ создания нейросетей путем последовательного добавления слоев.
- **Оптимизаторы**:
  - `Adam`
  - `GeneticAlgorithmOptimizer` (оптимизация с использованием генетического алгоритма)
- **Функции потерь**:
  - `CrossEntropyLoss` (перекрестная энтропия)
- **Тренировочный цикл**: класс `Trainer` для удобного обучения моделей.
- **Загрузчик данных**: `DataLoader` для работы с набором данных MNIST.

## Структура проекта

- `learnCNN.cpp`: Главный файл для определения, обучения и тестирования модели CNN.
- `Tensor.h`: Реализация тензоров.
- `*.h`: Заголовочные файлы, определяющие различные слои, оптимизаторы и утилиты.
- `mnist/`: Каталог для хранения набора данных MNIST.
- `CMakeLists.txt`: Скрипт для сборки проекта с использованием CMake.

## Системные требования

- C++17-совместимый компилятор (GCC, Clang, MSVC)
- CMake (версия 3.10 или выше)
- Git

## Начало работы

### 1. Клонирование репозитория

```bash
git clone <URL-репозитория>
cd BaumanEDUcnnFrame
```

### 2. Загрузка набора данных MNIST

Для обучения модели требуется набор данных MNIST. Скрипт `learnCNN.cpp` ожидает найти файлы данных в каталоге `mnist/`.

1. Создайте каталог `mnist`:
   ```bash
   mkdir mnist
   ```
2. Загрузите файлы набора данных MNIST и поместите их в этот каталог:
   - `train-images-idx3-ubyte`
   - `train-labels-idx1-ubyte`
   - `t10k-images-idx3-ubyte`
   - `t10k-labels-idx1-ubyte`

   Вы можете скачать их с официального сайта [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).

### 3. Сборка проекта

Используйте CMake для сборки проекта.

```bash
# Создание каталога для сборки
cmake -B build

# Сборка проекта
cmake --build build
```

### 4. Запуск обучения

После успешной сборки исполнимый файл `learnCNN` будет находиться в каталоге `build`.

```bash
./build/learnCNN
```

## Пример: определение модели

Вот как можно определить модель CNN (архитектура LeNet-5) с помощью этой библиотеки:

```cpp
#include "Sequential.h"
#include "Conv2DLayer.h"
#include "ReLULayer.h"
#include "MaxPooling2DLayer.h"
#include "FlattenLayer.h"
#include "DenseLayer.h"
#include "SoftmaxLayer.h"

// ...

Sequential model;
// Вход: (N, 1, 28, 28)
model.add(std::make_unique<Conv2DLayer>(1, 6, 5)); // (N, 6, 24, 24)
model.add(std::make_unique<ReLULayer>());
model.add(std::make_unique<MaxPooling2DLayer>(2)); // (N, 6, 12, 12)

model.add(std::make_unique<Conv2DLayer>(6, 16, 5)); // (N, 16, 8, 8)
model.add(std::make_unique<ReLULayer>());
model.add(std::make_unique<MaxPooling2DLayer>(2)); // (N, 16, 4, 4)

model.add(std::make_unique<FlattenLayer>()); // (N, 16 * 4 * 4 = 256)
model.add(std::make_unique<DenseLayer>(256, 120));
model.add(std::make_unique<ReLULayer>());
model.add(std::make_unique<DenseLayer>(120, 84));
model.add(std::make_unique<ReLULayer>());
model.add(std::make_unique<DenseLayer>(84, 10));
model.add(std::make_unique<SoftmaxLayer>());

model.summary(); // Вывод архитектуры модели в консоль
```
