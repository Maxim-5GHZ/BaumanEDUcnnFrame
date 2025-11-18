#pragma once
#include "Tensor.h"
#include <cmath>
#include <algorithm>

class Loss {
public:
    virtual ~Loss() = default;
    virtual float calculate(const Tensor& y_pred, const Tensor& y_true) = 0;
    virtual Tensor derivative(const Tensor& y_pred, const Tensor& y_true) = 0;
};

class MeanSquaredError : public Loss {
public:
    float calculate(const Tensor& y_pred, const Tensor& y_true) override {
        float sum = 0.0f;
        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            float diff = y_pred.data[i] - y_true.data[i];
            sum += diff * diff;
        }
        return sum / y_pred.shape[0];
    }

    Tensor derivative(const Tensor& y_pred, const Tensor& y_true) override {
        Tensor grad = y_pred;
        float batch_size = y_pred.shape[0];
        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            grad.data[i] = 2.0f * (y_pred.data[i] - y_true.data[i]) / batch_size;
        }
        return grad;
    }
};

class CrossEntropyLoss : public Loss {
public:
    float calculate(const Tensor& y_pred, const Tensor& y_true) override {
        assert(y_pred.shape == y_true.shape);
        float loss = 0.0f;
        int batch_size = y_pred.shape[0];
        int num_classes = y_pred.shape[1];

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_classes; ++j) {
                if (y_true.at(i, j) > 0.99f) {
                    float pred_clipped = std::max(y_pred.at(i, j), 1e-9f);
                    loss += -log(pred_clipped);
                    break;
                }
            }
        }
        return loss / batch_size;
    }

    Tensor derivative(const Tensor& y_pred, const Tensor& y_true) override {
        assert(y_pred.shape == y_true.shape);
        Tensor grad(y_pred.shape);
        float batch_size = y_pred.shape[0];

        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            float pred_clipped = std::max(y_pred.data[i], 1e-9f);
            grad.data[i] = -y_true.data[i] / pred_clipped / batch_size;
        }
        return grad;
    }
};