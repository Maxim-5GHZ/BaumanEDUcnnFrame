#include "Tensor.h"
#include "Sequential.h"
#include "DenseLayer.h"
#include "ReLULayer.h"
#include "SoftmaxLayer.h"
#include "FlattenLayer.h"
#include "Conv2DLayer.h"
#include "MaxPooling2DLayer.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Trainer.h"
#include "DataLoader.h"
#include <iostream>
#include <vector>
#include <algorithm>

void train_mnist() {
    std::cout << "\n--- MNIST Training Demo ---\n" << std::endl;

    std::cout << "Loading MNIST dataset..." << std::endl;
    std::string train_images_path = "mnist/train-images-idx3-ubyte";
    std::string train_labels_path = "mnist/train-labels-idx1-ubyte";
    std::string val_images_path = "mnist/t10k-images-idx3-ubyte";
    std::string val_labels_path = "mnist/t10k-labels-idx1-ubyte";

    auto X_train_full = MNISTDataLoader::load_images(train_images_path);
    auto y_train_full = MNISTDataLoader::load_labels(train_labels_path);
    auto X_val = MNISTDataLoader::load_images(val_images_path);
    auto y_val = MNISTDataLoader::load_labels(val_labels_path);
    std::cout << "Dataset loaded successfully.\n" << std::endl;

    Sequential model;
    // Input: (N, 1, 28, 28)
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

    std::cout << "Model Architecture:" << std::endl;
    model.summary();
    std::cout << std::endl;

    CrossEntropyLoss loss_fn;
    auto optimizer = std::make_unique<Adam>(0.001f);
    Trainer trainer(model, std::move(optimizer), loss_fn);

    int epochs = 5;
    int batch_size = 64;

    std::cout << "Starting training..." << std::endl;
    trainer.fit(X_train_full, y_train_full, X_val, y_val, epochs, batch_size);
    std::cout << "\n--- Training Finished ---" << std::endl;

    try {
        model.save_model("mnist_cnn_model.bin");
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
    }
}

int main() {
    try {
        train_mnist();
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}