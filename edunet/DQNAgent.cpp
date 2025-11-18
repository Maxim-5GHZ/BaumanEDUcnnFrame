#include "DQNAgent.h"
#include <iostream>
#include <fstream>
#include <algorithm> // For std::sample
#include <iterator>  // For std::back_inserter

// Helper to create a Tensor from a flat vector
Tensor vector_to_tensor(const std::vector<float>& vec, const std::vector<int>& shape) {
    Tensor t(shape);
    t.data = vec;
    return t;
}

DQNAgent::DQNAgent(int state_size, int action_size)
    : state_size(state_size),
      action_size(action_size),
      gen(std::random_device()()) {
    
    model = build_model();
    target_model = build_model();
    optimizer = std::make_unique<Adam>(learning_rate);

    update_target_model();
}

Sequential DQNAgent::build_model() {
    Sequential new_model;
    new_model.add(std::make_unique<DenseLayer>(state_size, 24));
    new_model.add(std::make_unique<ReLULayer>());
    new_model.add(std::make_unique<DenseLayer>(24, 24));
    new_model.add(std::make_unique<ReLULayer>());
    new_model.add(std::make_unique<DenseLayer>(24, action_size));
    return new_model;
}

void DQNAgent::update_target_model() {
    const std::string temp_path = "temp_agent_weights.bin";
    model.save_model(temp_path);
    target_model.load_model(temp_path);
}

void DQNAgent::remember(const std::vector<float>& state, int action, float reward, const std::vector<float>& next_state, bool done) {
    if (memory.size() >= memory_size) {
        memory.pop_front();
    }
    memory.push_back({state, action, reward, next_state, done});
}

int DQNAgent::act(const std::vector<float>& state) {
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    if (dis(gen) <= epsilon) {
        std::uniform_int_distribution<> distrib(0, action_size - 1);
        return distrib(gen);
    }

    Tensor state_tensor = vector_to_tensor(state, {1, state_size});
    Tensor q_values_tensor = model.forward(state_tensor);
    
    const auto& q_values = q_values_tensor.data;
    return std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end()));
}

void DQNAgent::replay(int batch_size) {
    if (memory.size() < batch_size) {
        return;
    }

    std::vector<Transition> minibatch;
    std::sample(memory.begin(), memory.end(), std::back_inserter(minibatch),
                batch_size, gen);

    std::vector<float> states_flat;
    std::vector<float> targets_flat;

    for (const auto& transition : minibatch) {
        // Get current Q values from the main model
        Tensor state_tensor = vector_to_tensor(transition.state, {1, state_size});
        Tensor current_q_tensor = model.forward(state_tensor);
        std::vector<float> target_q = current_q_tensor.data;

        float target_val;
        if (transition.done) {
            target_val = transition.reward;
        } else {
            // Get next Q values from the target model
            Tensor next_state_tensor = vector_to_tensor(transition.next_state, {1, state_size});
            Tensor next_q_tensor = target_model.forward(next_state_tensor);
            const auto& next_q_values = next_q_tensor.data;
            target_val = transition.reward + gamma * (*std::max_element(next_q_values.begin(), next_q_values.end()));
        }

        // Update the Q value for the action that was taken
        target_q[transition.action] = target_val;

        // Add to the batch data
        states_flat.insert(states_flat.end(), transition.state.begin(), transition.state.end());
        targets_flat.insert(targets_flat.end(), target_q.begin(), target_q.end());
    }

    // Create batch tensors
    Tensor states_batch = vector_to_tensor(states_flat, {batch_size, state_size});
    Tensor targets_batch = vector_to_tensor(targets_flat, {batch_size, action_size});

    // Perform a single training step
    Tensor y_pred = model.forward(states_batch);
    Tensor loss_grad = loss_fn.derivative(y_pred, targets_batch);
    model.backward(loss_grad);
    optimizer->step(model);

    // Decay epsilon
    if (epsilon > epsilon_min) {
        epsilon *= epsilon_decay;
    }
}

void DQNAgent::save(const std::string& path) {
    model.save_model(path);
}

void DQNAgent::load(const std::string& path) {
    model.load_model(path);
    update_target_model();
}

void DQNAgent::set_evaluation_mode(bool eval) {
    if (eval) {
        epsilon = 0.0f;
    } else {
        epsilon = 1.0f; // Reset to default exploration rate for training
    }
}