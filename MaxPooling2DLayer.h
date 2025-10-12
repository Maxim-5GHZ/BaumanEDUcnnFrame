#pragma once
#include "Layer.h"
#include <vector>
#include <algorithm>
#include <limits>
#include <fstream>

class MaxPooling2DLayer : public Layer {
private:
    int pool_size;
    int stride;
    Tensor last_input;
    std::vector<int> max_indices; 
public:
    MaxPooling2DLayer(int p_size, int s = -1) : pool_size(p_size) {
        this->stride = (s == -1) ? pool_size : s;
    }
    MaxPooling2DLayer() = default;

    Tensor forward(const Tensor& input) override {
        last_input = input;
        int N=input.shape[0], C=input.shape[1], H_in=input.shape[2], W_in=input.shape[3];
        int H_out = (H_in - pool_size) / stride + 1;
        int W_out = (W_in - pool_size) / stride + 1;
        Tensor output({N, C, H_out, W_out});
        max_indices.assign(output.data.size(), -1);
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        int max_idx = -1;
                        for (int ph = 0; ph < pool_size; ++ph) {
                            for (int pw = 0; pw < pool_size; ++pw) {
                                int h_in_idx = h * stride + ph;
                                int w_in_idx = w * stride + pw;
                                int flat_idx = n*C*H_in*W_in + c*H_in*W_in + h_in_idx*W_in + w_in_idx;
                                if (input.data[flat_idx] > max_val) {
                                    max_val = input.data[flat_idx];
                                    max_idx = flat_idx;
                                }
                            }
                        }
                        int out_flat_idx = n*C*H_out*W_out + c*H_out*W_out + h*W_out + w;
                        output.data[out_flat_idx] = max_val;
                        max_indices[out_flat_idx] = max_idx;
                    }
                }
            }
        }
        return output;
    }
    Tensor backward(const Tensor& output_gradient) override {
        Tensor input_gradient(last_input.shape);
        for (size_t i = 0; i < max_indices.size(); ++i) {
            int input_idx = max_indices[i];
            if(input_idx != -1) input_gradient.data[input_idx] += output_gradient.data[i];
        }
        return input_gradient;
    }

    std::unique_ptr<Layer> clone() const override { return std::make_unique<MaxPooling2DLayer>(*this); }
    
    void save_weights(const std::string& filename) const override {
        std::ofstream file(filename + "_maxpool.meta");
        file << pool_size << " " << stride;
        file.close();
    }
    
    void load_weights(const std::string& filename) override {
        std::ifstream file(filename + "_maxpool.meta");
        if (!file) throw std::runtime_error("Cannot open meta file: " + filename + "_maxpool.meta");
        file >> pool_size >> stride;
        file.close();
    }
    
    std::string get_layer_type() const override { return "MaxPooling2DLayer"; }
    
    std::string get_weights_string() const override {
        return "pool_size:" + std::to_string(pool_size) + ";stride:" + std::to_string(stride);
    }
    
    void set_weights_from_string(const std::string& data) override {
        size_t pool_pos = data.find("pool_size:") + 10;
        size_t stride_pos = data.find("stride:") + 7;
        pool_size = std::stoi(data.substr(pool_pos, data.find(";") - pool_pos));
        stride = std::stoi(data.substr(stride_pos));
    }
};