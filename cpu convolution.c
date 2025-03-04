#include <stdio.h>    // 必须保留 # 符号
#include <stdlib.h>   // 动态内存管理头文件

// 卷积函数声明（必须在调用前定义或声明）
void convolution(float *input, float *output, float *kernel, 
                 int width, int height, int kernel_size);

int main() {
    // 明确定义所有变量（关键修复）
    int width = 1024;
    int height = 1024;       // 原代码中缺少的变量
    int kernel_size = 3;
    int total_pixels = width * height;  // 计算总像素数

    // 内存分配（确保变量已定义）
    float *input = (float*)malloc(total_pixels * sizeof(float));
    float *output = (float*)malloc(total_pixels * sizeof(float));
    float *kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));

    // 初始化输入数据
    for (int i = 0; i < total_pixels; i++) {
        input[i] = (float)(i % 255);
    }

    // 初始化卷积核（3x3均值滤波）
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] = 1.0f / 9.0f;
    }

    // 调用卷积函数（参数必须全部定义）
    convolution(input, output, kernel, width, height, kernel_size);

    // 输出结果
    int center_x = width / 2;
    int center_y = height / 2;
    printf("Result at (%d, %d): %.2f\n", 
           center_x, center_y, output[center_y * width + center_x]);

    // 释放内存
    free(input);
    free(output);
    free(kernel);

    return 0;
}

// 卷积函数实现（需放在main之后或提前声明）
void convolution(float *input, float *output, float *kernel, 
                 int width, int height, int kernel_size) {
    int half_kernel = kernel_size / 2;
    
    for (int ty = 0; ty < height; ++ty) {
        for (int tx = 0; tx < width; ++tx) {
            float sum = 0.0f;
            
            for (int i = -half_kernel; i <= half_kernel; ++i) {
                for (int j = -half_kernel; j <= half_kernel; ++j) {
                    int x = tx + j;
                    int y = ty + i;
                    
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
