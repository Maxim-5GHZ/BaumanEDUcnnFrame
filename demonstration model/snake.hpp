
#include <deque>
#include <random>
#include <stdexcept>
#include <ncurses.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#pragma once

struct Position {
    int x, y;
    Position(int col = 0, int row = 0) : x(col), y(row) {}
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }
};

struct SnakeConfig {
    int width = 30;
    int height = 30;
    int initial_length = 15;
    int max_steps = 1000000000;
    int max_steps_without_food = 300000;
    char head_char = '@';
    char body_char = 'O';
    char food_char = 'F';
    char wall_char = '#';
    int food_score = 1;
    std::function<void()> on_game_over = [](){};
    std::function<void(int)> on_score_change = [](int){};
};

class SnakeGame {
private:
    SnakeConfig config;
    Position food;
    std::deque<Position> snake;
    int direction; 
    int steps_without_food{0};
    bool game_over;
    int score;
    std::mt19937 gen;
    std::uniform_int_distribution<> dist_x;
    std::uniform_int_distribution<> dist_y;
    int step_count;

    void place_food() {
        int attempts = 0;
        const int max_attempts = (config.width - 2) * (config.height - 2);
        
        while (attempts++ < max_attempts) {
            food.x = dist_x(gen);
            food.y = dist_y(gen);
            
            bool on_snake = false;
            for (const auto& segment : snake) {
                if (food == segment) {
                    on_snake = true;
                    break;
                }
            }
            
            if (!on_snake) return;
        }
        
        game_over = true;
        config.on_game_over();
    }

    void process_input(int key) {
        switch (key) {
            case KEY_UP:
                if (direction != 2) direction = 0;
                break;
            case KEY_RIGHT:
                if (direction != 3) direction = 1;
                break;
            case KEY_DOWN:
                if (direction != 0) direction = 2;
                break;
            case KEY_LEFT:
                if (direction != 1) direction = 3;
                break;
            case 'q':
                game_over = true;
                break;
            case 'p':
                while (getch() != 'p') {}
                break;
        }
    }

    void init_ncurses() {
        initscr();
        if (!stdscr) {
            throw std::runtime_error("Failed to initialize ncurses");
        }
        cbreak();
        noecho();
        keypad(stdscr, TRUE);
        curs_set(0);
    }

public:
    void update_without_render() {
        steps_without_food++; 
        step_count++;
        if (step_count > config.max_steps) {
            game_over = true;
            config.on_game_over();
            return;
        }
        
        Position head = snake.front();
        switch (direction) {
            case 0: head.y--; break;
            case 1: head.x++; break;
            case 2: head.y++; break;
            case 3: head.x--; break;
        }
        
        if (head.x <= 0 || head.x >= config.width-1 || 
            head.y <= 0 || head.y >= config.height-1) {
            game_over = true;
            config.on_game_over();
            return;
        }
        
    for (auto it = snake.begin() + 1; it != snake.end(); ++it) {
           if (head == *it) {
         game_over = true;
       config.on_game_over();
            return;
         }
   }

        snake.push_front(head);
        if (steps_without_food >= config.max_steps_without_food) {
            game_over = true;
            config.on_game_over();
            return;
    }

        if (head == food) {
            steps_without_food = 0;
            score += config.food_score;
            config.on_score_change(score);
            place_food();
        } else {
            snake.pop_back();
        }
    }
    SnakeGame(const SnakeConfig& cfg = {}) 
        : config(cfg),
          food(0, 0), 
          direction(1), 
          game_over(false), 
          score(0),
          gen(std::random_device()()),
          dist_x(1, config.width-2),
          dist_y(1, config.height-2),
          step_count(0) {
        
        if (config.width < 5 || config.height < 5) {
            throw std::invalid_argument("Game area too small (minimum 5x5)");
        }

        int start_x = config.width / 2;
        int start_y = config.height / 2;
        for (int i = 0; i < config.initial_length; ++i) {
            snake.emplace_back(start_x - i, start_y);
        }
        place_food();
    }

    int returnScore() const {
        return score;
    }

    Position returnFoodPlace() const {
        return food;
    }

    void update_direction(int action) {
        if (action == 1) {
            direction = (direction + 1) % 4;
        } else if (action == 2) {
            direction = (direction + 3) % 4;
        }
    }

    template<typename T>
    std::vector<T> get_state() const {
        std::vector<T> state(8);
        Position head = snake.front();

        int dx_current, dy_current;
        switch (direction) {
            case 0: dx_current = 0; dy_current = -1; break;
            case 1: dx_current = 1; dy_current = 0; break;
            case 2: dx_current = 0; dy_current = 1; break;
            case 3: dx_current = -1; dy_current = 0; break;
            default: dx_current = 0; dy_current = 0; break;
        }

        auto get_distance = [&](int dx, int dy) -> T {
            if (dx == 0 && dy == 0) return T(0);
            
            Position pos = head;
            int steps = 0;
            while (true) {
                pos.x += dx;
                pos.y += dy;
                steps++;
                
                if (pos.x < 0 || pos.x >= config.width || 
                    pos.y < 0 || pos.y >= config.height) {
                    break;
                }
                
                bool body_found = false;
                for (const auto& segment : snake) {
                    if (pos == segment) {
                        body_found = true;
                        break;
                    }
                }
                if (body_found) break;
            }
            
            return T(1) / T(steps);
        };

        state[0] = get_distance(dx_current, dy_current);
        state[1] = get_distance(dy_current, -dx_current);
        state[2] = get_distance(-dy_current, dx_current);

        state[3] = static_cast<T>(food.x - head.x) / (config.width - 2);
        state[4] = static_cast<T>(food.y - head.y) / (config.height - 2);
        
        state[5] = static_cast<T>(dx_current);
        state[6] = static_cast<T>(dy_current);
        
        state[7] = static_cast<T>(snake.size() - config.initial_length) / 
                  ((config.width-2)*(config.height-2) - config.initial_length);

        return state;
    }

    void draw() const {
        clear();
        
        for (int i = 0; i < config.width; i++) {
            mvaddch(0, i, config.wall_char);
            mvaddch(config.height-1, i, config.wall_char);
        }
        for (int i = 0; i < config.height; i++) {
            mvaddch(i, 0, config.wall_char);
            mvaddch(i, config.width-1, config.wall_char);
        }

        for (const auto& segment : snake) {
            mvaddch(segment.y, segment.x, config.body_char);
        }
       
        mvaddch(snake.front().y, snake.front().x, config.head_char);
        
        mvaddch(food.y, food.x, config.food_char);
        
        mvprintw(config.height, 0, "Score: %d | Steps: %d/%d", 
                score, step_count, config.max_steps);
        refresh();
    }

    void update() {
        Position head = snake.front();
        steps_without_food++; 
        step_count++;
        switch (direction) {
            case 0: head.y--; break;
            case 1: head.x++; break;
            case 2: head.y++; break;
            case 3: head.x--; break;
        }
        
        if (head.x <= 0 || head.x >= config.width-1 || 
            head.y <= 0 || head.y >= config.height-1) {
            game_over = true;
            config.on_game_over();
            return;
        }
        
        for (auto it = snake.begin() + 1; it != snake.end(); ++it) {
            if (head == *it) {
                game_over = true;
                config.on_game_over();
                return;
                }
        }

        snake.push_front(head);

        if (head == food) {
            score += config.food_score;
            config.on_score_change(score);
            place_food();
        } else {
            snake.pop_back();
        }

        if (steps_without_food >= config.max_steps_without_food) {
            game_over = true;
            config.on_game_over();
            return;
         }
    }

    void run() {
        try {
            init_ncurses();
            
            while (!game_over) {
                process_input(getch());
                update();
                draw();
            }
            
            endwin();
        } catch (const std::exception& e) {
            endwin();
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    bool is_over() const {
        return game_over;
    }

    Position get_head_position() const {
        return snake.front();
    }

    void reset() {
        snake.clear();
        int start_x = config.width / 2;
        int start_y = config.height / 2;
        for (int i = 0; i < config.initial_length; ++i) {
            snake.emplace_back(start_x - i, start_y);
        }
        
        direction = 1;
        score = 0;
        steps_without_food = 0;
        step_count = 0;
        game_over = false;
        
        place_food();
        config.on_score_change(score);
    }
};

