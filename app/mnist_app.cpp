#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <algorithm>

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

        model.save_model("mnist_cnn_model.bin");

    } catch (const std::exception& e) {
        std::cerr << "An error occurred during MNIST training: " << e.what() << std::endl;
    }
}

void run_mnist_testing(const std::string& mnist_path) {
    std::cout << "\n--- MNIST Testing ---\n" << std::endl;
    const std::string model_path = "mnist_cnn_model.bin";

    try {
        Sequential model;
        // We need to build the model with the correct layer types but dummy parameters,
        // as load_model reconstructs the parameters from the file.
        model.add(std::make_unique<Conv2DLayer>());
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<MaxPooling2DLayer>());
        model.add(std::make_unique<Conv2DLayer>());
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<MaxPooling2DLayer>());
        model.add(std::make_unique<FlattenLayer>());
        model.add(std::make_unique<DenseLayer>());
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<DenseLayer>());
        model.add(std::make_unique<ReLULayer>());
        model.add(std::make_unique<DenseLayer>());
        model.add(std::make_unique<SoftmaxLayer>());
        
        model.load_model(model_path);
        std::cout << "Model loaded successfully from " << model_path << std::endl;

        std::cout << "Loading MNIST test dataset..." << std::endl;
        auto X_test = MNISTDataLoader::load_images(mnist_path + "t10k-images-idx3-ubyte");
        auto y_test = MNISTDataLoader::load_labels(mnist_path + "t10k-labels-idx1-ubyte");
        std::cout << "Dataset loaded successfully." << std::endl;

        model.eval(); // Set model to evaluation mode

        float correct_predictions = 0;
        int total_predictions = 0;
        int batch_size = 64;

        for (size_t i = 0; i < X_test.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, X_test.size());
            size_t current_batch_size = end - i;

            // Create batch tensors
            std::vector<float> x_batch_data;
            std::vector<float> y_batch_data;
            for(size_t j = 0; j < current_batch_size; ++j) {
                x_batch_data.insert(x_batch_data.end(), X_test[i+j].data.begin(), X_test[i+j].data.end());
                y_batch_data.insert(y_batch_data.end(), y_test[i+j].data.begin(), y_test[i+j].data.end());
            }
            
            Tensor X_batch({(int)current_batch_size, X_test[0].shape[1], X_test[0].shape[2], X_test[0].shape[3]});
            X_batch.data = x_batch_data;
            
            Tensor y_batch({(int)current_batch_size, y_test[0].shape[1]});
            y_batch.data = y_batch_data;

            Tensor y_pred = model.forward(X_batch);

            for (size_t j = 0; j < current_batch_size; ++j) {
                auto pred_start = y_pred.data.begin() + j * y_pred.shape[1];
                auto pred_end = pred_start + y_pred.shape[1];
                auto true_start = y_batch.data.begin() + j * y_batch.shape[1];
                auto true_end = true_start + y_batch.shape[1];

                int pred_idx = std::distance(pred_start, std::max_element(pred_start, pred_end));
                int true_idx = std::distance(true_start, std::max_element(true_start, true_end));

                if (pred_idx == true_idx) {
                    correct_predictions++;
                }
            }
            total_predictions += current_batch_size;
        }

        float accuracy = (total_predictions > 0) ? (correct_predictions / total_predictions) : 0.0f;
        std::cout << "\n--- Testing Finished ---" << std::endl;
        std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << accuracy * 100.0f << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred during MNIST testing: " << e.what() << std::endl;
    }
}
