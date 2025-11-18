// === Visualizer.h ===

#pragma once
#include "Tensor.h"
#include <iostream>
#include <string>

namespace Visualizer {

    /**
     * @brief Печатает тензор изображения в виде ASCII-арта в консоль.
     * 
     * @param image Тензор с формой (1, 1, H, W), где значения пикселей находятся в диапазоне [0.0, 1.0].
     */
    void print_image_ascii(const Tensor& image) {
        // Проверяем, что тензор имеет правильную форму для изображения
        assert(image.shape.size() == 4 && image.shape[0] == 1 && image.shape[1] == 1);

        int height = image.shape[2];
        int width = image.shape[3];

        // "Палитра" символов от самого светлого (пробел) к самому темному (@)
        const std::string ramp = " .:-=+*#%@";
        
        std::cout << "Image (" << height << "x" << width << "):" << std::endl;

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // Получаем значение пикселя (от 0.0 до 1.0)
                float pixel_value = image.at(0, 0, h, w);
                
                // Преобразуем значение пикселя в индекс символа в нашей палитре
                int ramp_index = static_cast<int>(pixel_value * (ramp.length() - 1));
                
                // Печатаем символ дважды, чтобы изображение не было слишком "сплюснутым" по вертикали
                std::cout << ramp[ramp_index] << ramp[ramp_index];
            }
            // Переход на новую строку после каждой строки пикселей
            std::cout << std::endl;
        }
    }

} // namespace Visualizer