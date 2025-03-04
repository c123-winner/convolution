#include <stdio.h>
#include <stdlib.h>  // 用于动态内存分配

// 卷积计算函数（CPU实现）
void convolution(float *input, float *output, float *kernel, 
                 int width, int height, int kernel_size) {
    int half_kernel = kernel_size / 2;

    // 遍历每个像素
    for (int ty = 0; ty < height; ++ty) {
        for (int tx = 0; tx < width; ++tx) {
            float sum = 0.0f;

            // 遍历卷积核
            for (int i = -half_kernel; i <= half_kernel; ++i) {
                for (int j = -half_kernel; j <= half_kernel; ++j) {
                    int x = tx + i;
                    int y = ty + j;

                    // 边界检查（处理图像边缘）
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        sum += input[y * width + x] * 
                               kernel[(i + half_kernel) * kernel_size + (j + half_kernel)];
                    }
                }
            }

            output[ty * width + tx] = sum;
        }
    }
}

int main() {
    // 定义图像和卷积核参数
    int width = 1024;      // 图像宽度
    int height = 1024;     // 图像高度
    int kernel_size = 3;   // 卷积核大小（3x3）
    int total_pixels = width * height;

    // 分配内存
    float *input = (float*)malloc(total_pixels * sizeof(float));
    float *output = (float*)malloc(total_pixels * sizeof(float));
    float *kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));

    // 初始化输入图像（模拟灰度数据）
    for (int i = 0; i < total_pixels; i++) {
        input[i] = (float)(i % 255);  // 0-254循环填充
    }

    // 初始化卷积核（均值滤波器）
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] = 1.0f / 9.0f;  // 3x3均值滤波
    }

    // 执行卷积计算
    convolution(input, output, kernel, width, height, kernel_size);

    // 验证结果：输出中心点数值
    int center_x = width / 2;
    int center_y = height / 2;
    printf("Convolution result at (%d, %d): %.2f\n", 
           center_x, center_y, output[center_y * width + center_x]);

    // 释放内存
    free(input);
    free(output);
    free(kernel);

    return 0;
}