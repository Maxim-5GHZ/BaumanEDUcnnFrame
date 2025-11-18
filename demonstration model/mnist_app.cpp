#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <algorithm>
#include <random>

#include "mnist_app.h"
#include "DataLoader.h"
#include "Sequential.h"
#include "Conv2DLayer.h"
#include "ReLULayer.h"
#include "MaxPooling2DLayer.h"
#include "FlattenLayer.h"
#include "DenseLayer.h"
#include "SoftmaxLayer.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Trainer.h"

// Helper function to print a 28x28 MNIST image tensor as ASCII art
void print_ascii_image(const Tensor& image) {
    if (image.shape.size() != 4 || image.shape[0] != 1 || image.shape[1] != 1 || image.shape[2] != 28 || image.shape[3] != 28) {
        // std::cerr << "Invalid image tensor shape for ASCII art printing." << std::endl;
        return;
    }
    const char* shades = " .:-=+*#%@";
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            float pixel_value = image.data[i * 28 + j];
            int shade_index = static_cast<int>(pixel_value * 9.99f);
            std::cout << shades[shade_index] << shades[shade_index];
        }
        std::cout << std::endl;
    }
}

void run_mnist_training(const std::string& mnist_path) {
    int epochs;
    std::cout << "Enter number of epochs to train for: ";
    std::cin >> epochs;
    if (epochs <= 0) {
        std::cout << "Invalid number of epochs." << std::endl;
        return;
    }

    std::cout << "\n--- MNIST Training ---\n" << std::endl;

    try {
        std::cout << "Loading MNIST dataset..." << std::endl;
        auto X_train_full = MNISTDataLoader::load_images(mnist_path + "train-images-idx3-ubyte");
        auto y_train_full = MNISTDataLoader::load_labels(mnist_path + "train-labels-idx1-ubyte");
        auto X_val = MNISTDataLoader::load_images(mnist_path + "t10k-images-idx3-ubyte");
        auto y_val = MNISTDataLoader::load_labels(mnist_path + "t10k-labels-idx1-ubyte");
        std::cout << "Dataset loaded successfully.\n" << std::endl;

        Sequential model;
        model.add(std::make_unique<Conv2DLayer>(1, 6, 5));
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<MaxPooling2DLayer>(2));
        model.add(std::make_unique<Conv2DLayer>(6, 16, 5));
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<MaxPooling2DLayer>(2));
        model.add(std::make_unique<FlattenLayer>());
        model.add(std::make_unique<DenseLayer>(256, 120));
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<DenseLayer>(120, 84));
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<DenseLayer>(84, 10));
        model.add(std::make_unique<SoftmaxLayer>());

        std::cout << "Model Architecture:" << std::endl;
        model.summary();
        std::cout << std::endl;

        CrossEntropyLoss loss_fn;
        auto optimizer = std::make_unique<Adam>(0.001f);
        Trainer trainer(model, std::move(optimizer), loss_fn);

        int batch_size = 64;

        std::cout << "Starting training for " << epochs << " epochs..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        trainer.fit(X_train_full, y_train_full, X_val, y_val, epochs, batch_size);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        std::cout << "\n--- Training Finished ---" << std::endl;
        std::cout << "Total training time: " << duration.count() << " seconds" << std::endl;

        model.save_model("weights/mnist_cnn_model.bin");

    } catch (const std::exception& e) {
        std::cerr << "An error occurred during MNIST training: " << e.what() << std::endl;
    }
}

void run_mnist_testing(const std::string& mnist_path) {
    std::cout << "\n--- MNIST Model Demonstration ---\n" << std::endl;
    const std::string model_path = "weights/mnist_cnn_model.bin";

    try {
        // Define the exact model architecture before loading weights
        Sequential model;
        model.add(std::make_unique<Conv2DLayer>(1, 6, 5));
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<MaxPooling2DLayer>(2));
        model.add(std::make_unique<Conv2DLayer>(6, 16, 5));
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<MaxPooling2DLayer>(2));
        model.add(std::make_unique<FlattenLayer>());
        model.add(std::make_unique<DenseLayer>(256, 120));
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<DenseLayer>(120, 84));
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<DenseLayer>(84, 10));
        model.add(std::make_unique<SoftmaxLayer>());
        
        model.load_model(model_path);
        std::cout << "Model loaded successfully from " << model_path << std::endl;

        std::cout << "Loading MNIST test dataset..." << std::endl;
        auto X_test = MNISTDataLoader::load_images(mnist_path + "t10k-images-idx3-ubyte");
        auto y_test = MNISTDataLoader::load_labels(mnist_path + "t10k-labels-idx1-ubyte");
        std::cout << "Dataset loaded successfully." << std::endl;

        model.eval(); // Set model to evaluation mode

        std::mt19937 rng(std::random_device{}());
        char user_choice = 'y';

        while (user_choice == 'y' || user_choice == 'Y') {
            std::uniform_int_distribution<int> dist(0, X_test.size() - 1);
            int random_index = dist(rng);

            Tensor& image_tensor = X_test[random_index];
            Tensor& label_tensor = y_test[random_index];

            std::cout << "\n------------------------------------------" << std::endl;
            std::cout << "Displaying random test image #" << random_index << std::endl;
            
            // Print the image as ASCII art
            print_ascii_image(image_tensor);

            // Get the true label
            auto true_label_it = std::max_element(label_tensor.data.begin(), label_tensor.data.end());
            int true_label = std::distance(label_tensor.data.begin(), true_label_it);

            // Get model prediction
            Tensor prediction = model.forward(image_tensor);
            auto pred_label_it = std::max_element(prediction.data.begin(), prediction.data.end());
            int predicted_label = std::distance(prediction.data.begin(), pred_label_it);

            std::cout << "\nActual Label:    " << true_label << std::endl;
            std::cout << "Predicted Label: " << predicted_label << std::endl;
            std::cout << "------------------------------------------\n" << std::endl;

            std::cout << "Show another random image? (y/n): ";
            std::cin >> user_choice;
        }

        // Finally, calculate and display the overall test accuracy
        std::cout << "\nCalculating overall accuracy on the test set..." << std::endl;
        float correct_predictions = 0;
        int total_predictions = 0;
        int batch_size = 64;

        for (size_t i = 0; i < X_test.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, X_test.size());
            size_t current_batch_size = end - i;

            std::vector<int> x_batch_shape = {(int)current_batch_size, X_test[0].shape[1], X_test[0].shape[2], X_test[0].shape[3]};
            std::vector<int> y_batch_shape = {(int)current_batch_size, y_test[0].shape[1]};
            Tensor X_batch(x_batch_shape);
            Tensor y_batch(y_batch_shape);

            std::vector<float> x_batch_data;
            std::vector<float> y_batch_data;
            x_batch_data.reserve(current_batch_size * X_test[0].data.size());
            y_batch_data.reserve(current_batch_size * y_test[0].data.size());

            for(size_t j = 0; j < current_batch_size; ++j) {
                x_batch_data.insert(x_batch_data.end(), X_test[i+j].data.begin(), X_test[i+j].data.end());
                y_batch_data.insert(y_batch_data.end(), y_test[i+j].data.begin(), y_test[i+j].data.end());
            }
            
            X_batch.data = x_batch_data;
            y_batch.data = y_batch_data;

            Tensor y_pred = model.forward(X_batch);

            for (size_t j = 0; j < current_batch_size; ++j) {
                auto pred_start = y_pred.data.begin() + j * y_pred.shape[1];
                auto pred_end = pred_start + y_pred.shape[1];
                auto true_start = y_batch.data.begin() + j * y_batch.shape[1];
                
                int pred_idx = std::distance(pred_start, std::max_element(pred_start, pred_end));
                int true_idx = std::distance(true_start, std::max_element(true_start, true_start + y_batch.shape[1]));

                if (pred_idx == true_idx) {
                    correct_predictions++;
                }
            }
            total_predictions += current_batch_size;
        }

        float accuracy = (total_predictions > 0) ? (correct_predictions / total_predictions) : 0.0f;
        std::cout << "\n--- Testing Finished ---" << std::endl;
        std::cout << "Overall Test Accuracy: " << std::fixed << std::setprecision(2) << accuracy * 100.0f << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred during MNIST testing: " << e.what() << std::endl;
    }
}
