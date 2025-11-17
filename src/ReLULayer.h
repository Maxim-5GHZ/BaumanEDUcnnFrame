#pragma once
#include "Layer.h"
#include <fstream>

class ReLULayer : public Layer {
private:
    Tensor last_input;
public:
    Tensor forward(const Tensor& input) override {
        last_input = input;
        Tensor output = input;
        for (auto& val : output.data) {
            if (val < 0) {
                val = 0;
            }
        }
        return output;
    }
    
    Tensor backward(const Tensor& output_gradient) override {
        Tensor input_gradient = output_gradient;
        for (size_t i = 0; i < last_input.data.size(); ++i) {
            if (last_input.data[i] <= 0) {
                input_gradient.data[i] = 0;
            }
        }
        return input_gradient;
    }
    
    std::unique_ptr<Layer> clone() const override {
        return std::make_unique<ReLULayer>(*this);
    }
    
    void save_weights(const std::string& filename) const override {
        std::ofstream file(filename + "_relu.meta", std::ios::binary);
        file << "ReLULayer";
        file.close();
    }
    
    void load_weights(const std::string& filename) override {}
    
    std::string get_layer_type() const override { return "ReLULayer"; }
    
    std::string get_weights_string() const override { return "ReLULayer_no_weights"; }
    
    void set_weights_from_string(const std::string& data) override {}
};