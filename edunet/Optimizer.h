#pragma once
#include "Sequential.h"
#include <unordered_map>
#include <memory>
#include <cmath>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(Sequential& model) = 0;
    virtual void reset() {}
};

class SGD : public Optimizer {
private:
    float learning_rate;
public:
    SGD(float lr = 0.01f) : learning_rate(lr) {}
    
    void step(Sequential& model) override {
        for (const auto& layer_ptr : model.layers) {
            if (DenseLayer* dense = dynamic_cast<DenseLayer*>(layer_ptr.get())) {
                for (size_t i = 0; i < dense->weights.data.size(); ++i) dense->weights.data[i] -= learning_rate * dense->grad_weights.data[i];
                for (size_t i = 0; i < dense->bias.data.size(); ++i) dense->bias.data[i] -= learning_rate * dense->grad_bias.data[i];
            } else if (Conv2DLayer* conv = dynamic_cast<Conv2DLayer*>(layer_ptr.get())) {
                for (size_t i = 0; i < conv->kernels.data.size(); ++i) conv->kernels.data[i] -= learning_rate * conv->grad_kernels.data[i];
                for (size_t i = 0; i < conv->biases.data.size(); ++i) conv->biases.data[i] -= learning_rate * conv->grad_biases.data[i];
            }
        }
    }
};

class Adam : public Optimizer {
private:
    float learning_rate, beta1, beta2, epsilon;
    int timestep;
    struct LayerMoments { Tensor m_weights, v_weights, m_bias, v_bias; };
    // ИСПРАВЛЕНО: Ключ теперь int (индекс слоя), а не Layer*, что гораздо безопаснее
    std::unordered_map<int, LayerMoments> moments;
    
public:
    Adam(float lr=0.001f, float b1=0.9f, float b2=0.999f, float eps=1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), timestep(0) {}
    
    void step(Sequential& model) override {
        timestep++;
        // ИСПРАВЛЕНО: Итерация по модели с использованием индекса
        for (size_t i = 0; i < model.layers.size(); ++i) {
            auto& layer_ptr = model.layers[i];
            Layer* layer = layer_ptr.get();
            DenseLayer* dense = dynamic_cast<DenseLayer*>(layer);
            Conv2DLayer* conv = dynamic_cast<Conv2DLayer*>(layer);
            
            if (!dense && !conv) continue;

            // ИСПРАВЛЕНО: Используем индекс 'i' в качестве ключа
            if (moments.find(i) == moments.end()) {
                 if (dense) {
                    moments[i] = {Tensor(dense->weights.shape), Tensor(dense->weights.shape), Tensor(dense->bias.shape), Tensor(dense->bias.shape)};
                } else if (conv) {
                    moments[i] = {Tensor(conv->kernels.shape), Tensor(conv->kernels.shape), Tensor(conv->biases.shape), Tensor(conv->biases.shape)};
                }
            }
            
            // ИСПРАВЛЕНО: Используем индекс 'i' для доступа к моментам
            auto& layer_moments = moments[i];
            
            if (dense) {
                update_parameters(dense->weights, dense->grad_weights, layer_moments.m_weights, layer_moments.v_weights);
                update_parameters(dense->bias, dense->grad_bias, layer_moments.m_bias, layer_moments.v_bias);
            } else if (conv) {
                update_parameters(conv->kernels, conv->grad_kernels, layer_moments.m_weights, layer_moments.v_weights);
                update_parameters(conv->biases, conv->grad_biases, layer_moments.m_bias, layer_moments.v_bias);
            }
        }
    }
    
    void reset() override { moments.clear(); timestep = 0; }
    
private:
    void update_parameters(Tensor& params, const Tensor& grads, Tensor& m, Tensor& v) {
        for (size_t i = 0; i < m.data.size(); ++i) m.data[i] = beta1 * m.data[i] + (1 - beta1) * grads.data[i];
        for (size_t i = 0; i < v.data.size(); ++i) v.data[i] = beta2 * v.data[i] + (1 - beta2) * grads.data[i] * grads.data[i];
        
        float m_hat_scale = 1.0f / (1 - std::pow(beta1, timestep));
        float v_hat_scale = 1.0f / (1 - std::pow(beta2, timestep));
        
        for (size_t i = 0; i < params.data.size(); ++i) {
            float m_hat = m.data[i] * m_hat_scale;
            float v_hat = v.data[i] * v_hat_scale;
            params.data[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
};