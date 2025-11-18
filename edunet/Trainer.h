#pragma once
#include "Sequential.h"
#include "Optimizer.h"
#include "Loss.h"
#include <vector>
#include "Sequential.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>

class Trainer {
private:
    Sequential& model;
    std::unique_ptr<Optimizer> optimizer;
    Loss& loss_fn;
    
public:
    Trainer(Sequential& m, std::unique_ptr<Optimizer> opt, Loss& loss)
        : model(m), optimizer(std::move(opt)), loss_fn(loss) {}

    float train_batch(const Tensor& X_batch, const Tensor& y_batch) {
        Tensor y_pred = model.forward(X_batch);
        float loss = loss_fn.calculate(y_pred, y_batch);
        Tensor loss_grad = loss_fn.derivative(y_pred, y_batch);
        model.backward(loss_grad);
        optimizer->step(model);
        return loss;
    }

    void fit(const std::vector<Tensor>& X_train, const std::vector<Tensor>& y_train,
             const std::vector<Tensor>& X_val, const std::vector<Tensor>& y_val,
             int epochs, int batch_size) {
        
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            std::cout << "Epoch " << epoch << "/" << epochs << std::endl;
            
            // --- Training Phase ---
            model.train(); // УЛУЧШЕНО: Переключение модели в режим обучения
            float epoch_loss = train_epoch(X_train, y_train, batch_size);
            
            // --- Validation Phase ---
            model.eval(); // УЛУЧШЕНО: Переключение модели в режим оценки
            auto [val_loss, val_accuracy] = evaluate(X_val, y_val, batch_size);

            std::cout << " - Loss: " << epoch_loss 
                      << " - Val Loss: " << val_loss 
                      << " - Val Accuracy: " << val_accuracy * 100.0f << "%" << std::endl;
        }
    }

private:
    float train_epoch(const std::vector<Tensor>& X, const std::vector<Tensor>& y, int batch_size) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        std::vector<int> indices(X.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        // УЛУЧШЕНО: Создаем тензоры для батча один раз вне цикла для производительности
        const auto& first_x = X[0];
        const auto& first_y = y[0];
        std::vector<int> x_batch_shape = first_x.shape;
        x_batch_shape[0] = batch_size;
        std::vector<int> y_batch_shape = first_y.shape;
        y_batch_shape[0] = batch_size;
        
        Tensor X_batch(x_batch_shape);
        Tensor y_batch(y_batch_shape);

        for (size_t i = 0; i < X.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, X.size());
            size_t current_batch_size = end - i;

            // УЛУЧШЕНО: Если последний батч меньше, изменяем размер тензора
            if (current_batch_size != static_cast<size_t>(X_batch.shape[0])) {
                X_batch.shape[0] = current_batch_size;
                X_batch.data.resize(current_batch_size * first_x.data.size());
                y_batch.shape[0] = current_batch_size;
                y_batch.data.resize(current_batch_size * first_y.data.size());
            }

            // Копируем данные в существующие тензоры батча
            for (size_t j = 0; j < current_batch_size; ++j) {
                const auto& x_sample = X[indices[i + j]];
                const auto& y_sample = y[indices[i + j]];
                std::copy(x_sample.data.begin(), x_sample.data.end(), X_batch.data.begin() + j * x_sample.data.size());
                std::copy(y_sample.data.begin(), y_sample.data.end(), y_batch.data.begin() + j * y_sample.data.size());
            }

            float batch_loss = train_batch(X_batch, y_batch);
            total_loss += batch_loss;
            num_batches++;
            
            std::cout << "\r" << "  Batch " << num_batches << "/" << (X.size() + batch_size - 1) / batch_size
                      << " - Batch Loss: " << batch_loss << std::flush;
        }
        std::cout << std::endl;
        return total_loss / num_batches;
    }

    std::pair<float, float> evaluate(const std::vector<Tensor>& X, const std::vector<Tensor>& y, int batch_size) {
        float total_loss = 0.0f;
        float correct_predictions = 0.0f;
        int num_batches = 0;

        // УЛУЧШЕНО: Аналогичная оптимизация создания батчей для evaluate
        const auto& first_x = X[0];
        const auto& first_y = y[0];
        std::vector<int> x_batch_shape = first_x.shape;
        x_batch_shape[0] = batch_size;
        std::vector<int> y_batch_shape = first_y.shape;
        y_batch_shape[0] = batch_size;

        Tensor X_batch(x_batch_shape);
        Tensor y_batch(y_batch_shape);

        for (size_t i = 0; i < X.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, X.size());
            size_t current_batch_size = end - i;

            if (current_batch_size != static_cast<size_t>(X_batch.shape[0])) {
                X_batch.shape[0] = current_batch_size;
                X_batch.data.resize(current_batch_size * first_x.data.size());
                y_batch.shape[0] = current_batch_size;
                y_batch.data.resize(current_batch_size * first_y.data.size());
            }

            for (size_t j = 0; j < current_batch_size; ++j) {
                const auto& x_sample = X[i + j];
                const auto& y_sample = y[i + j];
                std::copy(x_sample.data.begin(), x_sample.data.end(), X_batch.data.begin() + j * x_sample.data.size());
                std::copy(y_sample.data.begin(), y_sample.data.end(), y_batch.data.begin() + j * y_sample.data.size());
            }

            Tensor y_pred = model.forward(X_batch);
            total_loss += loss_fn.calculate(y_pred, y_batch);
            num_batches++;

            for (size_t j = 0; j < current_batch_size; ++j) {
                auto pred_start = y_pred.data.begin() + j * y_pred.shape[1];
                auto pred_end = pred_start + y_pred.shape[1];
                auto true_start = y_batch.data.begin() + j * y_batch.shape[1];
                auto true_end = true_start + y_batch.shape[1];

                int pred_idx = std::distance(pred_start, std::max_element(pred_start, pred_end));
                int true_idx = std::distance(true_start, std::max_element(true_start, true_end));

                if (pred_idx == true_idx) {
                    correct_predictions++;
                }
            }
        }
        
        float avg_loss = (num_batches > 0) ? (total_loss / num_batches) : 0.0f;
        float accuracy = (X.size() > 0) ? (correct_predictions / X.size()) : 0.0f;
        return {avg_loss, accuracy};
    }
};