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
#include <libgen.h> // For dirname
#include <unistd.h> // For readlink
#include <linux/limits.h> // For PATH_MAX

// Include all necessary headers from the project
#include "snake.hpp"
#include "DQNAgent.h"
#include "mnist_app.h" // Include the new header for MNIST functions

// Forward declarations for snake functions
void run_snake_training();
void run_snake_visualization();
void show_menu();

// Helper function for snake distance calculation
float distance(Position p1, Position p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

int main() {
    // Get the path to the executable
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count == -1) {
        std::cerr << "Could not get executable path." << std::endl;
        return 1;
    }
    std::string exe_path(result, count);
    std::string exe_dir = dirname(&exe_path[0]);
    std::string mnist_path = exe_dir + "/mnist/";


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
                run_mnist_training(mnist_path);
                break;
            case 2:
                run_mnist_testing(mnist_path);
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