#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <deque>
#include <limits>

// Include all necessary headers from the project
#include "snake.hpp"
#include "DQNAgent.h"
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

// Forward declarations for functions that will be implemented
void run_mnist_training();
void run_mnist_testing();
void run_snake_training();
void run_snake_visualization();
void show_menu();

// Helper function for snake distance calculation
float distance(Position p1, Position p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

int main() {
    int choice = 0;
    while (choice != 5) {
        show_menu();
        std::cin >> choice;

        // Handle non-integer input
        if (std::cin.fail()) {
            std::cout << "Invalid input. Please enter a number." << std::endl;
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            choice = 0; // Reset choice
            continue;
        }

        switch (choice) {
            case 1:
                run_mnist_training();
                break;
            case 2:
                run_mnist_testing();
                break;
            case 3:
                run_snake_training();
                break;
            case 4:
                run_snake_visualization();
                break;
            case 5:
                std::cout << "Exiting..." << std::endl;
                break;
            default:
                std::cout << "Invalid choice. Please try again." << std::endl;
                break;
        }
        if (choice != 5) {
            std::cout << "\nPress Enter to continue...";
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin.get();
        }
    }
    return 0;
}

void show_menu() {
    // Clear console before showing menu
    // std::cout << "\033[2J\033[1;1H"; 
    std::cout << "\n--- CNN Suite Main Menu ---" << std::endl;
    std::cout << "1. Train MNIST Model\n";
    std::cout << "2. Test MNIST Model\n";
    std::cout << "3. Train Snake Agent\n";
    std::cout << "4. Visualize Snake Agent\n";
    std::cout << "5. Exit\n";
    std::cout << "Enter your choice: ";
}

void run_mnist_training() {
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
        auto X_train_full = MNISTDataLoader::load_images("mnist/train-images-idx3-ubyte");
        auto y_train_full = MNISTDataLoader::load_labels("mnist/train-labels-idx1-ubyte");
        auto X_val = MNISTDataLoader::load_images("mnist/t10k-images-idx3-ubyte");
        auto y_val = MNISTDataLoader::load_labels("mnist/t10k-labels-idx1-ubyte");
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

void run_mnist_testing() {
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
        auto X_test = MNISTDataLoader::load_images("mnist/t10k-images-idx3-ubyte");
        auto y_test = MNISTDataLoader::load_labels("mnist/t10k-labels-idx1-ubyte");
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

void run_snake_training() {
    int minutes;
    std::cout << "Enter number of minutes to train for: ";
    std::cin >> minutes;
    if (minutes <= 0) {
        std::cout << "Invalid duration." << std::endl;
        return;
    }
    
    const int BATCH_SIZE = 32;
    const int STATE_SIZE = 8;
    const int ACTION_SIZE = 3;

    SnakeConfig config;
    config.width = 10;
    config.height = 10;
    config.initial_length = 3;
    config.max_steps_without_food = 200;

    SnakeGame game(config);
    DQNAgent agent(STATE_SIZE, ACTION_SIZE);

    std::deque<int> recent_scores;
    const int scores_window = 100;

    auto training_start_time = std::chrono::high_resolution_clock::now();
    auto training_end_time = training_start_time + std::chrono::minutes(minutes);
    int episode = 0;

    std::cout << "\n--- Snake Training for " << minutes << " minute(s) ---\n" << std::endl;

    while (std::chrono::high_resolution_clock::now() < training_end_time) {
        episode++;
        game.reset();
        auto state = game.get_state<float>();
        int score_before_move = 0;

        for (int time = 0; time < 5000; ++time) {
            int action = agent.act(state);

            Position head_before = game.get_head_position();
            Position food_pos = game.returnFoodPlace();
            float dist_before = distance(head_before, food_pos);
            score_before_move = game.returnScore();

            game.update_direction(action);
            game.update_without_render();

            bool done = game.is_over();
            auto next_state = game.get_state<float>();
            int score_after_move = game.returnScore();

            float reward = 0.0f;
            if (done) {
                reward = -10.0f;
            } else if (score_after_move > score_before_move) {
                reward = 10.0f;
            } else {
                Position head_after = game.get_head_position();
                float dist_after = distance(head_after, food_pos);
                reward = (dist_after < dist_before) ? 0.1f : -0.2f;
            }

            agent.remember(state, action, reward, next_state, done);
            state = next_state;

            if (done) break;
        }

        agent.replay(BATCH_SIZE);

        recent_scores.push_back(game.returnScore());
        if (recent_scores.size() > scores_window) {
            recent_scores.pop_front();
        }
        double avg_score = std::accumulate(recent_scores.begin(), recent_scores.end(), 0.0) / recent_scores.size();

        std::cout << "Episode " << std::setw(5) << episode
                  << " | Score: " << std::setw(3) << game.returnScore()
                  << " | Avg Score: " << std::fixed << std::setprecision(2) << std::setw(5) << avg_score
                  << std::endl;

        if (episode % 5 == 0) {
            agent.update_target_model();
        }
    }

    std::cout << "\n--- Training Finished ---" << std::endl;
    agent.save("snake-dqn-final.bin");
}

void run_snake_visualization() {
    std::cout << "\n--- Snake Visualization ---\n" << std::endl;
    const std::string model_path = "snake-dqn-final.bin";
    
    const int STATE_SIZE = 8;
    const int ACTION_SIZE = 3;

    SnakeConfig config;
    config.width = 10;
    config.height = 10;
    config.initial_length = 3;

    SnakeGame game(config);
    DQNAgent agent(STATE_SIZE, ACTION_SIZE);

    try {
        agent.load(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        std::cerr << "Please train a model first (Option 3)." << std::endl;
        return;
    }

    agent.set_evaluation_mode(true);

    try {
        initscr();
        cbreak();
        noecho();
        keypad(stdscr, TRUE);
        curs_set(0);
        timeout(0);

        mvprintw(0, 0, "Starting visualization. Press 'q' to quit.");
        refresh();

        while (true) {
            game.reset();
            
            while (!game.is_over()) {
                auto state = game.get_state<float>();
                int action = agent.act(state);

                game.update_direction(action);
                game.update();
                game.draw();

                int ch = getch();
                if (ch == 'q') {
                    endwin();
                    return;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            mvprintw(config.height + 1, 0, "Game Over! Score: %d. Press 'q' to quit or any other key to restart.", game.returnScore());
            timeout(-1);
            int ch = getch();
            if (ch == 'q') break;
            timeout(0);
        }

        endwin();
    } catch (const std::exception& e) {
        endwin();
        std::cerr << "An error occurred during visualization: " << e.what() << std::endl;
    }
}
