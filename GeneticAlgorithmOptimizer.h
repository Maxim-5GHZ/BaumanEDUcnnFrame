#pragma once
#include "Sequential.h"
#include "Loss.h"
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

struct Individual {
    Sequential model;
    float fitness = 0.0f;
    bool operator<(const Individual& other) const {
        return fitness > other.fitness; 
    }
};

class GeneticAlgorithmOptimizer {
public:
    int population_size;
    float mutation_rate;
    float mutation_strength;
    int elitism_count; 
    std::vector<Individual> population;
private:
    std::mt19937 generator;
public:
    GeneticAlgorithmOptimizer(int pop_size = 50, float mut_rate = 0.05f, float mut_strength = 0.1f, int elite_count = 2)
        : population_size(pop_size), mutation_rate(mut_rate), mutation_strength(mut_strength), elitism_count(elite_count) {
        std::random_device rd;
        generator.seed(rd());
    }
    Sequential train(const Sequential& model_template, const Tensor& X_train, const Tensor& y_train, Loss& loss_fn, int generations) {
        initialize_population(model_template);
        for (int gen = 0; gen < generations; ++gen) {
            calculate_fitness(X_train, y_train, loss_fn);
            std::sort(population.begin(), population.end());
            if (gen % 10 == 0) {
                std::cout << "Generation " << gen << ", Best Fitness: " << population[0].fitness 
                          << ", Loss: " << (1.0f / (population[0].fitness + 1e-9f)) << std::endl;
            }
            evolve_new_generation();
        }
        calculate_fitness(X_train, y_train, loss_fn);
        std::sort(population.begin(), population.end());
        std::cout << "\n--- Training Finished ---" << std::endl;
        std::cout << "Final Best Loss: " << (1.0f / (population[0].fitness + 1e-9f)) << std::endl;
        return population[0].model;
    }
private:
    void initialize_population(const Sequential& model_template) {
        population.clear();
        for (int i = 0; i < population_size; ++i) {
            Individual individual;
            individual.model = Sequential(model_template);
            population.push_back(std::move(individual));
        }
    }
    void calculate_fitness(const Tensor& X, const Tensor& y, Loss& loss_fn) {
        for (auto& individual : population) {
            Tensor y_pred = individual.model.forward(X);
            float loss = loss_fn.calculate(y_pred, y);
            individual.fitness = 1.0f / (loss + 1e-6f);
        }
    }
    void evolve_new_generation() {
        std::vector<Individual> next_generation;
        next_generation.reserve(population_size);
        for (int i = 0; i < elitism_count; ++i) {
            next_generation.push_back(population[i]);
        }
        while (next_generation.size() < static_cast<size_t>(population_size)) {
            Individual& parent1 = select_parent();
            Individual& parent2 = select_parent();
            Individual offspring = crossover(parent1, parent2);
            mutate(offspring);
            next_generation.push_back(std::move(offspring));
        }
        population = std::move(next_generation);
    }
    Individual& select_parent() {
        int index = std::uniform_int_distribution<int>(0, population_size / 2 - 1)(generator);
        return population[index];
    }
    Individual crossover(const Individual& parent1, const Individual& parent2) {
        Individual child;
        child.model = Sequential(parent1.model); 
        for (size_t i = 0; i < child.model.layers.size(); ++i) {
            if (DenseLayer* child_dense = dynamic_cast<DenseLayer*>(child.model.layers[i].get())) {
                const DenseLayer* p1_dense = dynamic_cast<const DenseLayer*>(parent1.model.layers[i].get());
                const DenseLayer* p2_dense = dynamic_cast<const DenseLayer*>(parent2.model.layers[i].get());
                if(p1_dense && p2_dense) {
                    for(size_t j = 0; j < child_dense->weights.data.size(); ++j) {
                        if (std::uniform_real_distribution<float>(0, 1)(generator) < 0.5f) {
                            child_dense->weights.data[j] = p1_dense->weights.data[j];
                        } else {
                            child_dense->weights.data[j] = p2_dense->weights.data[j];
                        }
                    }
                    for(size_t j = 0; j < child_dense->bias.data.size(); ++j) {
                         if (std::uniform_real_distribution<float>(0, 1)(generator) < 0.5f) {
                            child_dense->bias.data[j] = p1_dense->bias.data[j];
                        } else {
                            child_dense->bias.data[j] = p2_dense->bias.data[j];
                        }
                    }
                }
            }
        }
        return child;
    }
    void mutate(Individual& individual) {
        std::uniform_real_distribution<float> dist(-mutation_strength, mutation_strength);
        for (auto& layer : individual.model.layers) {
            if (DenseLayer* dense_layer = dynamic_cast<DenseLayer*>(layer.get())) {
                for (auto& weight : dense_layer->weights.data) {
                    if (std::uniform_real_distribution<float>(0, 1)(generator) < mutation_rate) {
                        weight += dist(generator);
                    }
                }
                for (auto& b : dense_layer->bias.data) {
                    if (std::uniform_real_distribution<float>(0, 1)(generator) < mutation_rate) {
                        b += dist(generator);
                    }
                }
            }
        }
    }
};