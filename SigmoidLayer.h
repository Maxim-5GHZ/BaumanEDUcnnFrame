#pragma once
#include "Layer.h"
#include <cmath>
#include <fstream>

class SigmoidLayer : public Layer {
private:
    Tensor last_output; 
public:
    Tensor forward(const Tensor& input) override {
        Tensor output = input;
        for (size_t i = 0; i < output.data.size(); ++i) {
            output.data[i] = 1.0f / (1.0f + exp(-input.data[i]));
        }
        last_output = output; 
        return output;
    }
    
    Tensor backward(const Tensor& output_gradient) override {
        Tensor input_gradient = output_gradient;
        for (size_t i = 0; i < last_output.data.size(); ++i) {
            float s = last_output.data[i];
            input_gradient.data[i] = output_gradient.data[i] * (s * (1.0f - s));
        }
        return input_gradient;
    }
    
    std::unique_ptr<Layer> clone() const override {
        return std::make_unique<SigmoidLayer>(*this);
    }
    
    void save_weights(const std::string& filename) const override {
        std::ofstream file(filename + "_sigmoid.meta", std::ios::binary);
        file << "SigmoidLayer";
        file.close();
    }
    
    void load_weights(const std::string& filename) override {}
    
    std::string get_layer_type() const override { return "SigmoidLayer"; }
    
    std::string get_weights_string() const override { return "SigmoidLayer_no_weights"; }
    
    void set_weights_from_string(const std::string& data) override {}
};