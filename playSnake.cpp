
#include "snake.hpp"

#include "DQNAgent.h"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
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
        std::cout << "Model loaded successfully from " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        std::cerr << "Please ensure the model file '" << model_path << "' exists and is accessible." << std::endl;
        return 1;
    }

    // Set agent to evaluation mode (no random actions)
    agent.set_evaluation_mode(true);

    try {
        initscr();
        cbreak();
        noecho();
        keypad(stdscr, TRUE);
        curs_set(0);
        timeout(0); // Non-blocking getch

        std::cout << "Starting visualization. Press 'q' to quit." << std::endl;

        while (true) {
            game.reset();
            
            while (!game.is_over()) {
                auto state = game.get_state<float>();
                int action = agent.act(state);

                game.update_direction(action);
                game.update(); // Use the rendering update method
                game.draw();

                // Check for quit command
                int ch = getch();
                if (ch == 'q') {
                    endwin();
                    std::cout << "Visualization stopped by user." << std::endl;
                    return 0;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            mvprintw(config.height + 1, 0, "Game Over! Final Score: %d. Press any key to restart.", game.returnScore());
            timeout(-1); // Blocking getch
            int ch = getch();
            if (ch == 'q') break;
            timeout(0); // Back to non-blocking
        }

        endwin();
    } catch (const std::exception& e) {
        endwin();
        std::cerr << "An error occurred during visualization: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Visualization finished." << std::endl;
    return 0;
}
