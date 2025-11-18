#pragma once
#include "Layer.h"
#include <fstream>

class FlattenLayer : public Layer {
private:
    std::vector<int> last_input_shape;
public:
    Tensor forward(const Tensor& input) override {
        last_input_shape = input.shape;
        int batch_size = input.shape[0];
        int features = 1;
        for (size_t i = 1; i < input.shape.size(); ++i) {
            features *= input.shape[i];
        }
        Tensor output({batch_size, features});
        output.data = input.data; 
        return output;
    }
    
    Tensor backward(const Tensor& output_gradient) override {
        Tensor input_gradient(last_input_shape);
        input_gradient.data = output_gradient.data;
        return input_gradient;
    }
    
    std::unique_ptr<Layer> clone() const override {
        return std::make_unique<FlattenLayer>(*this);
    }
    
    void save_weights(const std::string& filename) const override {
        std::ofstream file(filename + "_flatten.meta", std::ios::binary);
        file << "FlattenLayer";
        file.close();
    }
    
    void load_weights(const std::string& filename) override {}
    
    std::string get_layer_type() const override { return "FlattenLayer"; }
    
    std::string get_weights_string() const override { return "FlattenLayer_no_weights"; }
    
    void set_weights_from_string(const std::string& data) override {}
};