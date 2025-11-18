#pragma once
#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <stdexcept>
#include <fstream>
#include <string>
#include <sstream>
#include <memory>

class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int> strides;

    Tensor() = default;
    Tensor(const std::vector<int>& s) : shape(s) {
        int total_size = 1;
        for (int dim : s) {
            total_size *= dim;
        }
        data.resize(total_size, 0.0f);
        calculate_strides();
    }

    void save_to_file(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for writing: " + filename);
        size_t shape_size = shape.size();
        file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
        file.write(reinterpret_cast<const char*>(shape.data()), shape_size * sizeof(int));
        size_t data_size = data.size();
        file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        file.write(reinterpret_cast<const char*>(data.data()), data_size * sizeof(float));
        file.close();
    }
    
    void load_from_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for reading: " + filename);
        size_t shape_size;
        file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
        shape.resize(shape_size);
        file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(int));
        size_t data_size;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        data.resize(data_size);
        file.read(reinterpret_cast<char*>(data.data()), data_size * sizeof(float));
        calculate_strides();
        file.close();
    }
    
    std::string to_string() const {
        std::stringstream ss;
        ss << "shape:[";
        for (size_t i = 0; i < shape.size(); ++i) ss << shape[i] << (i < shape.size() - 1 ? "," : "");
        ss << "] data:[";
        for (size_t i = 0; i < data.size(); ++i) ss << data[i] << (i < data.size() - 1 ? "," : "");
        ss << "]";
        return ss.str();
    }
    
    void from_string(const std::string& str) {
        shape.clear(); data.clear();
        size_t shape_start = str.find("shape:[") + 7;
        size_t shape_end = str.find("]", shape_start);
        size_t data_start = str.find("data:[") + 6;
        size_t data_end = str.find("]", data_start);
        std::string shape_str = str.substr(shape_start, shape_end - shape_start);
        std::stringstream ss_shape(shape_str);
        std::string item;
        while (std::getline(ss_shape, item, ',')) if (!item.empty()) shape.push_back(std::stoi(item));
        std::string data_str = str.substr(data_start, data_end - data_start);
        std::stringstream ss_data(data_str);
        while (std::getline(ss_data, item, ',')) if (!item.empty()) data.push_back(std::stof(item));
        calculate_strides();
    }

private:
    void calculate_strides() {
        if (shape.empty()) { strides.clear(); return; }
        strides.resize(shape.size());
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

public:
    int getIndex(const std::vector<int>& indices) const {
        assert(indices.size() == shape.size());
        int index = 0;
        for (size_t i = 0; i < indices.size(); ++i) index += indices[i] * strides[i];
        return index;
    }
    float& at(const std::vector<int>& indices) { return data[getIndex(indices)]; }
    const float& at(const std::vector<int>& indices) const { return data[getIndex(indices)]; }
    float& at(int row, int col) { assert(shape.size() == 2); return data[row * strides[0] + col * strides[1]]; }
    const float& at(int row, int col) const { assert(shape.size() == 2); return data[row * strides[0] + col * strides[1]]; }
    float& at(int n, int c, int h, int w) { assert(shape.size() == 4); return data[n*strides[0]+c*strides[1]+h*strides[2]+w*strides[3]]; }
    const float& at(int n, int c, int h, int w) const { assert(shape.size() == 4); return data[n*strides[0]+c*strides[1]+h*strides[2]+w*strides[3]]; }
    
    static Tensor dot(const Tensor& a, const Tensor& b) {
        assert(a.shape.size() == 2 && b.shape.size() == 2 && a.shape[1] == b.shape[0]);
        Tensor result({a.shape[0], b.shape[1]});
        for (int i = 0; i < a.shape[0]; ++i) {
            for (int j = 0; j < b.shape[1]; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < a.shape[1]; ++k) sum += a.at(i, k) * b.at(k, j);
                result.at(i, j) = sum;
            }
        }
        return result;
    }
    
    void print() const {
        std::cout << "Tensor Shape: [";
        for(size_t i=0; i<shape.size(); ++i) std::cout << shape[i] << (i == shape.size()-1 ? "" : ", ");
        std::cout << "]" << std::endl;
        if (shape.size() == 2) {
            for (int i = 0; i < shape[0]; ++i) {
                for (int j = 0; j < shape[1]; ++j) std::cout << at(i, j) << " ";
                std::cout << std::endl;
            }
        } else {
            for(const auto& val : data) std::cout << val << " ";
            std::cout << std::endl;
        }
    }
};