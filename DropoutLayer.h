// === DropoutLayer.h (ИСПРАВЛЕННАЯ ВЕРСИЯ) ===

#pragma once
#include "Layer.h"
#include <random>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>

class DropoutLayer : public Layer {
private:
    float rate;
    Tensor mask;
    bool is_training = true;
    std::default_random_engine generator;
public:
    DropoutLayer(float dropout_rate = 0.5) : rate(dropout_rate) {
        generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
    
    // ИСПРАВЛЕНО: Удален неоднозначный конструктор по умолчанию.
    // Конструктор выше с параметром по умолчанию (dropout_rate = 0.5)
    // уже служит конструктором по умолчанию.
    // DropoutLayer() = default;  <-- ЭТА СТРОКА УДАЛЕНА

    // УЛУЧШЕНО: Реализация виртуальных методов базового класса
    void train() override { is_training = true; }
    void eval() override { is_training = false; }

    Tensor forward(const Tensor& input) override {
        if (!is_training || rate == 0.0f) {
            return input;
        }

        mask = Tensor(input.shape);
        std::bernoulli_distribution distribution(1.0f - rate);
        
        float scale = (rate < 1.0f) ? (1.0f / (1.0f - rate)) : 0.0f;
        
        Tensor output = input;
        for (size_t i = 0; i < mask.data.size(); ++i) {
            if (distribution(generator)) {
                mask.data[i] = 1.0f;
                output.data[i] *= scale; 
            } else {
                mask.data[i] = 0.0f;
                output.data[i] = 0.0f;
            }
        }
        return output;
    }

    Tensor backward(const Tensor& output_gradient) override {
        if (!is_training || rate == 0.0f) {
            return output_gradient;
        }
        
        Tensor input_gradient = output_gradient;
        float scale = (rate < 1.0f) ? (1.0f / (1.0f - rate)) : 0.0f;
        
        for (size_t i = 0; i < input_gradient.data.size(); ++i) {
            input_gradient.data[i] *= mask.data[i] * scale;
        }
        return input_gradient;
    }
    
    std::unique_ptr<Layer> clone() const override { 
        return std::make_unique<DropoutLayer>(*this); 
    }

    void save_weights(const std::string& filename) const override {
        std::ofstream file(filename + "_dropout.meta");
        file << rate;
        file.close();
    }

    void load_weights(const std::string& filename) override {
        std::ifstream file(filename + "_dropout.meta");
        if (!file) {
            throw std::runtime_error("Cannot open meta file: " + filename + "_dropout.meta");
        }
        file >> rate;
        file.close();
    }

    std::string get_layer_type() const override { return "DropoutLayer"; }

    std::string get_weights_string() const override { return "rate:" + std::to_string(rate); }
    
    void set_weights_from_string(const std::string& data) override {
        size_t pos = data.find("rate:");
        if (pos != std::string::npos) {
            rate = std::stof(data.substr(pos + 5));
        }
    }
};