#pragma once
#include "Layer.h"
#include <random>
#include <fstream>
#include <sstream>

class Conv2DLayer : public Layer {
public:
    Tensor kernels; 
    Tensor biases;  
    Tensor grad_kernels;
    Tensor grad_biases;
private:
    int in_channels, out_channels;
    int kernel_size, stride, padding;
    Tensor last_input;
public:
    Conv2DLayer(int input_channels, int output_channels, int k_size, int s = 1, int p = 0)
        : in_channels(input_channels), out_channels(output_channels), 
          kernel_size(k_size), stride(s), padding(p) {
        kernels = Tensor({out_channels, in_channels, kernel_size, kernel_size});
        biases = Tensor({out_channels});
        grad_kernels = Tensor(kernels.shape);
        grad_biases = Tensor(biases.shape);
        
        // Xavier initialization
        // ИСПРАВЛЕНО: Правильная инициализация ГСЧ
        std::random_device rd;
        std::mt19937 generator(rd());
        float range = std::sqrt(6.0f / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size));
        std::uniform_real_distribution<float> distribution(-range, range);
        for (auto& w : kernels.data) w = distribution(generator);
    }
    Conv2DLayer() = default;

    Tensor forward(const Tensor& input) override {
        last_input = input;
        int N = input.shape[0], C_in = input.shape[1], H_in = input.shape[2], W_in = input.shape[3];
        int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
        int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;
        Tensor output({N, out_channels, H_out, W_out});
        for (int n = 0; n < N; ++n) {
            for (int c_out = 0; c_out < out_channels; ++c_out) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        float sum = biases.data[c_out];
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int h_in_idx = h * stride + kh - padding;
                                    int w_in_idx = w * stride + kw - padding;
                                    if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                        sum += input.at(n, c_in, h_in_idx, w_in_idx) * kernels.at(c_out, c_in, kh, kw);
                                    }
                                }
                            }
                        }
                        output.at(n, c_out, h, w) = sum;
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor& output_gradient) override {
        int N = last_input.shape[0], H_in = last_input.shape[2], W_in = last_input.shape[3];
        int H_out = output_gradient.shape[2], W_out = output_gradient.shape[3];
        Tensor input_gradient(last_input.shape);
        grad_kernels = Tensor(kernels.shape);
        grad_biases = Tensor(biases.shape);
        for (int n = 0; n < N; ++n) {
            for (int c_out = 0; c_out < out_channels; ++c_out) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        float grad_out_val = output_gradient.at(n, c_out, h, w);
                        grad_biases.data[c_out] += grad_out_val;
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int h_in_idx = h * stride + kh - padding;
                                    int w_in_idx = w * stride + kw - padding;
                                    if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                        grad_kernels.at(c_out, c_in, kh, kw) += last_input.at(n, c_in, h_in_idx, w_in_idx) * grad_out_val;
                                        input_gradient.at(n, c_in, h_in_idx, w_in_idx) += kernels.at(c_out, c_in, kh, kw) * grad_out_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return input_gradient;
    }

    std::unique_ptr<Layer> clone() const override { return std::make_unique<Conv2DLayer>(*this); }
    
    void save_weights(const std::string& filename) const override {
        std::ofstream meta_file(filename + "_conv2d.meta");
        meta_file << in_channels << " " << out_channels << " " << kernel_size << " " << stride << " " << padding;
        meta_file.close();
        kernels.save_to_file(filename + "_kernels.bin");
        biases.save_to_file(filename + "_biases.bin");
    }
    
    void load_weights(const std::string& filename) override {
        std::ifstream meta_file(filename + "_conv2d.meta");
        if (!meta_file) throw std::runtime_error("Cannot open meta file: " + filename + "_conv2d.meta");
        meta_file >> in_channels >> out_channels >> kernel_size >> stride >> padding;
        meta_file.close();
        kernels.load_from_file(filename + "_kernels.bin");
        biases.load_from_file(filename + "_biases.bin");
    }
    
    std::string get_layer_type() const override { return "Conv2DLayer"; }
    
    std::string get_weights_string() const override {
        std::stringstream ss;
        ss << "in_channels:" << in_channels << ";out_channels:" << out_channels << ";kernel_size:" << kernel_size
           << ";stride:" << stride << ";padding:" << padding << ";";
        ss << "kernels:" << kernels.to_string() << ";biases:" << biases.to_string();
        return ss.str();
    }
    
    void set_weights_from_string(const std::string& data) override {
        auto get_val = [&](const std::string& key) {
            size_t pos = data.find(key + ":");
            size_t end_pos = data.find(";", pos);
            return data.substr(pos + key.length() + 1, end_pos - (pos + key.length() + 1));
        };
        in_channels = std::stoi(get_val("in_channels"));
        out_channels = std::stoi(get_val("out_channels"));
        kernel_size = std::stoi(get_val("kernel_size"));
        stride = std::stoi(get_val("stride"));
        padding = std::stoi(get_val("padding"));
        size_t kernels_pos = data.find("kernels:");
        size_t biases_pos = data.find("biases:");
        kernels.from_string(data.substr(kernels_pos + 8, biases_pos - kernels_pos - 9));
        biases.from_string(data.substr(biases_pos + 7));
    }
};