#pragma once

#include "Sequential.h"
#include "DenseLayer.h"
#include "ReLULayer.h"
#include "Optimizer.h"
#include "Loss.h"
#include "Tensor.h"
#include <vector>
#include <deque>
#include <random>
#include <algorithm>

// A transition in the environment
struct Transition {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> next_state;
    bool done;
};

class DQNAgent {
public:
    DQNAgent(int state_size, int action_size);

    // Choose an action based on the current state using epsilon-greedy policy
    int act(const std::vector<float>& state);

    // Store a transition in the replay memory
    void remember(const std::vector<float>& state, int action, float reward, const std::vector<float>& next_state, bool done);

    // Train the model by replaying a batch of experiences
    void replay(int batch_size);

    // Update the target network weights
    void update_target_model();

    // Save the model weights
    void save(const std::string& path);

    // Load the model weights
    void load(const std::string& path);

    // Set the agent to evaluation or training mode
    void set_evaluation_mode(bool eval);

private:
    int state_size;
    int action_size;
    std::deque<Transition> memory;
    size_t memory_size = 10000;
    float gamma = 0.95f;    // discount rate
    float epsilon = 1.0f;   // exploration rate
    float epsilon_min = 0.01f;
    float epsilon_decay = 0.995f;
    float learning_rate = 0.001f;

    Sequential model;
    Sequential target_model;
    std::unique_ptr<Adam> optimizer;
    MeanSquaredError loss_fn;

    std::mt19937 gen;

    Sequential build_model();
};
