// === main.cpp (Версия для тестирования с ВИЗУАЛИЗАЦИЕЙ) ===

#include "Tensor.h"
#include "Sequential.h"
#include "DataLoader.h"
#include "Visualizer.h" // <-- ВКЛЮЧАЕМ НАШ НОВЫЙ ФАЙЛ
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

void test_mnist_model() {
    std::cout << "\n--- MNIST Model Testing Demo ---\n" << std::endl;

    // --- 1. Загрузка обученной модели ---
    std::string model_path = "mnist_cnn_model.bin";
    Sequential model;
    
    try {
        std::cout << "Loading pre-trained model from: " << model_path << std::endl;
        model.load_model(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        std::cerr << "Please make sure you have run the training script first to generate '" << model_path << "'." << std::endl;
        return;
    }

    // --- 2. Переключение модели в режим оценки ---
    model.eval();
    std::cout << "Model switched to evaluation mode." << std::endl;

    // --- 3. Загрузка тестовых данных ---
    std::cout << "\nLoading MNIST test dataset..." << std::endl;
    std::string test_images_path = "mnist/t10k-images-idx3-ubyte";
    std::string test_labels_path = "mnist/t10k-labels-idx1-ubyte";

    auto X_test = MNISTDataLoader::load_images(test_images_path);
    auto y_test = MNISTDataLoader::load_labels(test_labels_path);
    std::cout << "Test dataset loaded successfully.\n" << std::endl;

    // --- 4. Оценка модели и подсчет точности ---
    if (X_test.empty()) {
        std::cerr << "Test dataset is empty. Cannot perform evaluation." << std::endl;
        return;
    }

    float correct_predictions = 0;
    
    std::cout << "Running inference on the test set..." << std::endl;
    for (size_t i = 0; i < X_test.size(); ++i) {
        const auto& image = X_test[i];
        const auto& label = y_test[i];
        Tensor prediction = model.forward(image);

        int pred_idx = std::distance(prediction.data.begin(), std::max_element(prediction.data.begin(), prediction.data.end()));
        int true_idx = std::distance(label.data.begin(), std::max_element(label.data.begin(), label.data.end()));

        if (pred_idx == true_idx) {
            correct_predictions++;
        }
    }

    // --- 5. Вывод общих результатов ---
    float accuracy = (X_test.size() > 0) ? (correct_predictions / X_test.size()) : 0.0f;
    
    std::cout << "\n--- Test Results ---" << std::endl;
    std::cout << "Total Test Samples: " << X_test.size() << std::endl;
    std::cout << "Correct Predictions: " << static_cast<int>(correct_predictions) << std::endl;
    std::cout << "Accuracy: " << accuracy * 100.0f << "%" << std::endl;
    std::cout << "--------------------" << std::endl;

    // --- 6. Вывод нескольких примеров с ВИЗУАЛИЗАЦИЕЙ ---
    std::cout << "\n--- Visual Sample Predictions ---" << std::endl;
    for (size_t i = 0; i < 5; ++i) { // Выведем 5 примеров для наглядности
        const auto& image = X_test[i];
        const auto& label = y_test[i];
        
        // ВЫЗОВ ФУНКЦИИ ВИЗУАЛИЗАЦИИ
        Visualizer::print_image_ascii(image);
        
        Tensor prediction = model.forward(image);
        
        int pred_idx = std::distance(prediction.data.begin(), std::max_element(prediction.data.begin(), prediction.data.end()));
        int true_idx = std::distance(label.data.begin(), std::max_element(label.data.begin(), label.data.end()));

        std::cout << "==> Model Predicted: " << pred_idx << ", Actual Value: " << true_idx 
                  << (pred_idx == true_idx ? " (Correct)" : " (INCORRECT!)") << std::endl;
        std::cout << "--------------------------------------------------------\n" << std::endl;
    }
}

int main() {
    try {
        test_mnist_model();
    } catch (const std::exception& e) {
        std::cerr << "An unhandled error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}