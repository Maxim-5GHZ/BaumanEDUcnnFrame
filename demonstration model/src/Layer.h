#pragma once
#include "Tensor.h"
#include <memory>
#include <string>

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& output_gradient) = 0;
    virtual std::unique_ptr<Layer> clone() const = 0;
    
    // УЛУЧШЕНО: Добавлены виртуальные методы для переключения режимов train/eval
    virtual void train() {}
    virtual void eval() {}

    // Методы для сериализации
    virtual void save_weights(const std::string& filename) const = 0;
    virtual void load_weights(const std::string& filename) = 0;
    virtual std::string get_layer_type() const = 0;
    virtual std::string get_weights_string() const = 0;
    virtual void set_weights_from_string(const std::string& data) = 0;
};