#pragma once
#include "Tensor.h"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace MNISTDataLoader {
    uint32_t reverse_int(uint32_t i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
        return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
    }

    std::vector<Tensor> load_images(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);

        uint32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
        file.read(reinterpret_cast<char*>(&magic_number), 4); magic_number = reverse_int(magic_number);
        if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file magic number.");

        file.read(reinterpret_cast<char*>(&num_images), 4); num_images = reverse_int(num_images);
        file.read(reinterpret_cast<char*>(&num_rows), 4); num_rows = reverse_int(num_rows);
        file.read(reinterpret_cast<char*>(&num_cols), 4); num_cols = reverse_int(num_cols);

        std::cout << "Loading " << num_images << " images of size " << num_rows << "x" << num_cols << std::endl;

        std::vector<Tensor> images;
        images.reserve(num_images);
        int image_size = num_rows * num_cols;
        std::vector<unsigned char> buffer(image_size);

        for (uint32_t i = 0; i < num_images; ++i) {
            file.read(reinterpret_cast<char*>(buffer.data()), image_size);
            Tensor image_tensor({1, 1, (int)num_rows, (int)num_cols});
            for (int j = 0; j < image_size; ++j) {
                image_tensor.data[j] = static_cast<float>(buffer[j]) / 255.0f;
            }
            images.push_back(image_tensor);
        }
        return images;
    }

    std::vector<Tensor> load_labels(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);
        
        uint32_t magic_number = 0, num_labels = 0;
        file.read(reinterpret_cast<char*>(&magic_number), 4); magic_number = reverse_int(magic_number);
        if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file magic number.");

        file.read(reinterpret_cast<char*>(&num_labels), 4); num_labels = reverse_int(num_labels);
        
        std::cout << "Loading " << num_labels << " labels." << std::endl;

        std::vector<Tensor> labels;
        labels.reserve(num_labels);
        for (uint32_t i = 0; i < num_labels; ++i) {
            unsigned char label = 0;
            file.read(reinterpret_cast<char*>(&label), 1);
            Tensor label_tensor({1, 10});
            if (label < 10) {
                label_tensor.data[label] = 1.0f;
            }
            labels.push_back(label_tensor);
        }
        return labels;
    }
};