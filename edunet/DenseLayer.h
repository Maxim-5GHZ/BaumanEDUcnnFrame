#pragma once
#include "Layer.h"
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>

class DenseLayer : public Layer {
public:
    Tensor weights;
    Tensor bias;
    Tensor grad_weights;
    Tensor grad_bias;
    
private:
    Tensor last_input;
    int input_size;
    int output_size;
    
public:
    DenseLayer(int input_size, int output_size) 
        : input_size(input_size), output_size(output_size) {
        
        weights = Tensor({input_size, output_size});
        bias = Tensor({1, output_size});
        grad_weights = Tensor({input_size, output_size});
        grad_bias = Tensor({1, output_size});
        
        initialize_xavier();
    }
    
    DenseLayer() : input_size(0), output_size(0) {}
    
    Tensor forward(const Tensor& input) override {
        last_input = input;
        
        if (input.shape.size() != 2) {
            throw std::runtime_error("DenseLayer expects 2D input");
        }
        if (input.shape[1] != input_size) {
            throw std::runtime_error("Input size mismatch in DenseLayer");
        }
        
        Tensor output = Tensor::dot(input, weights);
        
        for (int i = 0; i < output.shape[0]; ++i) {
            for (int j = 0; j < output.shape[1]; ++j) {
                output.at(i, j) += bias.data[j];
            }
        }
        
        return output;
    }
    
    Tensor backward(const Tensor& output_gradient) override {
        if (output_gradient.shape.size() != 2) {
            throw std::runtime_error("DenseLayer expects 2D output gradient");
        }
        
        int batch_size = output_gradient.shape[0];
        
        Tensor last_input_T = transpose(last_input);
        grad_weights = Tensor::dot(last_input_T, output_gradient);
        
        grad_bias = Tensor({1, output_size});
        for (int j = 0; j < output_size; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < batch_size; ++i) {
                sum += output_gradient.at(i, j);
            }
            grad_bias.data[j] = sum;
        }
        
        Tensor weights_T = transpose(weights);
        Tensor input_gradient = Tensor::dot(output_gradient, weights_T);
        
        return input_gradient;
    }
    
    std::unique_ptr<Layer> clone() const override {
        auto new_layer = std::make_unique<DenseLayer>(input_size, output_size);
        new_layer->weights = this->weights;
        new_layer->bias = this->bias;
        new_layer->input_size = this->input_size;
        new_layer->output_size = this->output_size;
        return new_layer;
    }
    
    void initialize_xavier() {
        // ИСПРАВЛЕНО: Правильная инициализация ГСЧ для случайных весов при каждом запуске
        std::random_device rd;
        std::mt19937 generator(rd());
        
        float range = std::sqrt(6.0f / (input_size + output_size));
        std::uniform_real_distribution<float> distribution(-range, range);
        
        for (auto& w : weights.data) {
            w = distribution(generator);
        }
        
        for (auto& b : bias.data) {
            b = 0.0f;
        }
    }
    
    void initialize_he() {
        // ИСПРАВЛЕНО: Правильная инициализация ГСЧ
        std::random_device rd;
        std::mt19937 generator(rd());

        float stddev = std::sqrt(2.0f / input_size);
        std::normal_distribution<float> distribution(0.0f, stddev);
        
        for (auto& w : weights.data) {
            w = distribution(generator);
        }
        
        for (auto& b : bias.data) {
            b = 0.0f;
        }
    }
    
    Tensor transpose(const Tensor& tensor) const {
        if (tensor.shape.size() != 2) {
            throw std::runtime_error("Transpose expects 2D tensor");
        }
        
        Tensor result({tensor.shape[1], tensor.shape[0]});
        for (int i = 0; i < tensor.shape[0]; ++i) {
            for (int j = 0; j < tensor.shape[1]; ++j) {
                result.at(j, i) = tensor.at(i, j);
            }
        }
        return result;
    }

    void save_weights(const std::string& filename) const override {
        std::ofstream meta_file(filename + "_dense.meta");
        meta_file << input_size << " " << output_size;
        meta_file.close();
        
        weights.save_to_file(filename + "_weights.bin");
        bias.save_to_file(filename + "_bias.bin");
    }
    
    void load_weights(const std::string& filename) override {
        std::ifstream meta_file(filename + "_dense.meta");
        if (!meta_file) {
            throw std::runtime_error("Cannot open meta file: " + filename + "_dense.meta");
        }
        meta_file >> input_size >> output_size;
        meta_file.close();
        
        weights.load_from_file(filename + "_weights.bin");
        bias.load_from_file(filename + "_bias.bin");
        
        grad_weights = Tensor(weights.shape);
        grad_bias = Tensor(bias.shape);
    }
    
    std::string get_layer_type() const override { return "DenseLayer"; }
    
    std::string get_weights_string() const override {
        std::stringstream ss;
        ss << "input_size:" << input_size << ";output_size:" << output_size << ";";
        ss << "weights:" << weights.to_string() << ";bias:" << bias.to_string();
        return ss.str();
    }
    
    void set_weights_from_string(const std::string& data) override {
        size_t input_pos = data.find("input_size:");
        size_t output_pos = data.find("output_size:");
        size_t weights_pos = data.find("weights:");
        size_t bias_pos = data.find("bias:");
        
        if (input_pos == std::string::npos || output_pos == std::string::npos ||
            weights_pos == std::string::npos || bias_pos == std::string::npos) {
            throw std::runtime_error("Invalid weights string format for DenseLayer");
        }
        
        input_size = std::stoi(data.substr(input_pos + 11, output_pos - input_pos - 12));
        output_size = std::stoi(data.substr(output_pos + 12, weights_pos - output_pos - 13));
        
        weights.from_string(data.substr(weights_pos + 8, bias_pos - weights_pos - 9));
        bias.from_string(data.substr(bias_pos + 5));
        
        grad_weights = Tensor(weights.shape);
        grad_bias = Tensor(bias.shape);
    }
};