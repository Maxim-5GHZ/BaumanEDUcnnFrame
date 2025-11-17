#pragma once
#include "Layer.h"
#include "DenseLayer.h"
#include "ReLULayer.h"
#include "SoftmaxLayer.h"
#include "SigmoidLayer.h"
#include "FlattenLayer.h"
#include "DropoutLayer.h"
#include "Conv2DLayer.h"
#include "MaxPooling2DLayer.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>

class Sequential {
public:
    std::vector<std::unique_ptr<Layer>> layers;
    
    Sequential() = default;
    
    Sequential(const Sequential& other) {
        for (const auto& layer : other.layers) {
            layers.push_back(layer->clone());
        }
    }
    
    Sequential& operator=(const Sequential& other) {
        if (this == &other) return *this;
        layers.clear();
        for (const auto& layer : other.layers) {
            layers.push_back(layer->clone());
        }
        return *this;
    }
    
    Sequential(Sequential&& other) noexcept = default;
    Sequential& operator=(Sequential&& other) noexcept = default;
    
    void add(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }
    
    Tensor forward(const Tensor& input) {
        Tensor current_output = input;
        for (const auto& layer : layers) {
            current_output = layer->forward(current_output);
        }
        return current_output;
    }
    
    void backward(const Tensor& initial_gradient) {
        Tensor current_gradient = initial_gradient;
        for (int i = layers.size() - 1; i >= 0; --i) {
            current_gradient = layers[i]->backward(current_gradient);
        }
    }
    
    void save_model(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        
        size_t num_layers = layers.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
        
        for (size_t i = 0; i < num_layers; ++i) {
            std::string layer_type = layers[i]->get_layer_type();
            size_t type_length = layer_type.length();
            file.write(reinterpret_cast<const char*>(&type_length), sizeof(type_length));
            file.write(layer_type.c_str(), type_length);
            
            std::string weights_data = layers[i]->get_weights_string();
            size_t data_length = weights_data.length();
            file.write(reinterpret_cast<const char*>(&data_length), sizeof(data_length));
            file.write(weights_data.c_str(), data_length);
        }
        
        file.close();
        std::cout << "Model saved to: " << filename << std::endl;
    }
    
    void load_model(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading: " + filename);
        }
        
        layers.clear();
        
        size_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
        
        for (size_t i = 0; i < num_layers; ++i) {
            size_t type_length;
            file.read(reinterpret_cast<char*>(&type_length), sizeof(type_length));
            std::string layer_type(type_length, ' ');
            file.read(&layer_type[0], type_length);
            
            size_t data_length;
            file.read(reinterpret_cast<char*>(&data_length), sizeof(data_length));
            std::string weights_data(data_length, ' ');
            file.read(&weights_data[0], data_length);
            
            std::unique_ptr<Layer> layer;
            if (layer_type == "DenseLayer") {
                layer = std::make_unique<DenseLayer>();
            } else if (layer_type == "ReLULayer") {
                layer = std::make_unique<ReLULayer>();
            } else if (layer_type == "SoftmaxLayer") {
                layer = std::make_unique<SoftmaxLayer>();
            } else if (layer_type == "SigmoidLayer") {
                layer = std::make_unique<SigmoidLayer>();
            } else if (layer_type == "FlattenLayer") {
                layer = std::make_unique<FlattenLayer>();
            } else if (layer_type == "DropoutLayer") {
                layer = std::make_unique<DropoutLayer>();
            } else if (layer_type == "Conv2DLayer") {
                layer = std::make_unique<Conv2DLayer>();
            } else if (layer_type == "MaxPooling2DLayer") {
                layer = std::make_unique<MaxPooling2DLayer>();
            } else {
                throw std::runtime_error("Unknown layer type: " + layer_type);
            }
            
            if (layer) {
                layer->set_weights_from_string(weights_data);
                layers.push_back(std::move(layer));
            }
        }
        
        file.close();
        std::cout << "Model loaded from: " << filename << std::endl;
    }
    
    void summary() const {
        std::cout << "Model Summary:" << std::endl;
        std::cout << "==============" << std::endl;
        for (size_t i = 0; i < layers.size(); ++i) {
            std::cout << "Layer " << i << ": " << layers[i]->get_layer_type() << std::endl;
        }
    }

    // УЛУЧШЕНО: Методы для переключения режима всей модели
    void train() {
        for (auto& layer : layers) {
            layer->train();
        }
    }

    void eval() {
        for (auto& layer : layers) {
            layer->eval();
        }
    }
};