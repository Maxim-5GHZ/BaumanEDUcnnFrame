#pragma once
#include "Layer.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>

class SoftmaxLayer : public Layer {
private:
    Tensor last_output;
public:
    Tensor forward(const Tensor& input) override {
        assert(input.shape.size() == 2); 
        Tensor output({input.shape[0], input.shape[1]});
        for (int i = 0; i < input.shape[0]; ++i) {
            float max_val = input.at(i, 0);
            for (int j = 1; j < input.shape[1]; ++j) {
                if (input.at(i, j) > max_val) max_val = input.at(i, j);
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < input.shape[1]; ++j) {
                float val = exp(input.at(i, j) - max_val);
                output.at(i, j) = val;
                sum_exp += val;
            }
            for (int j = 0; j < input.shape[1]; ++j) {
                output.at(i, j) /= sum_exp;
            }
        }
        last_output = output;
        return output;
    }
    
    Tensor backward(const Tensor& output_gradient) override {
        assert(output_gradient.shape.size() == 2);
        Tensor input_gradient({output_gradient.shape[0], output_gradient.shape[1]});
        for (int n = 0; n < output_gradient.shape[0]; ++n) {
            float dot_product = 0.0f;
            for (int j = 0; j < output_gradient.shape[1]; ++j) {
                dot_product += output_gradient.at(n, j) * last_output.at(n, j);
            }
            for (int i = 0; i < output_gradient.shape[1]; ++i) {
                input_gradient.at(n, i) = last_output.at(n, i) * (output_gradient.at(n, i) - dot_product);
            }
        }
        return input_gradient;
    }
    
    std::unique_ptr<Layer> clone() const override {
        return std::make_unique<SoftmaxLayer>(*this);
    }
    
    void save_weights(const std::string& filename) const override {
        std::ofstream file(filename + "_softmax.meta", std::ios::binary);
        file << "SoftmaxLayer";
        file.close();
    }
    
    void load_weights(const std::string& filename) override {}
    
    std::string get_layer_type() const override { return "SoftmaxLayer"; }
    
    std::string get_weights_string() const override { return "SoftmaxLayer_no_weights"; }
    
    void set_weights_from_string(const std::string& data) override {}
};