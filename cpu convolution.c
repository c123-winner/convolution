/**
 * 文件：cpu_convolution.c
 * 功能：在CPU上实现图像卷积计算（支持任意奇数尺寸卷积核）
 * 作者：cwd
 * 日期：2024.06
 */

#include <stdio.h>    // 标准输入输出（如printf）
#include <stdlib.h>   // 动态内存管理（如malloc、free）

/**
 * 卷积计算函数（CPU实现）
 * 
 * @param input     输入图像数据（一维数组，按行存储）
 * @param output    输出图像数据（需提前分配内存）
 * @param kernel    卷积核数据（一维数组，按行存储）
 * @param width     输入图像的宽度（像素数）
 * @param height    输入图像的高度（像素数）
 * @param kernel_size 卷积核尺寸（必须为奇数，如3、5等）
 */
void convolution(float *input, float *output, float *kernel, 
                 int width, int height, int kernel_size) {
    int half_kernel = kernel_size / 2;  // 卷积核半径（例如3x3卷积核半径为1）

    // 遍历图像的每一个像素点（ty为纵坐标，tx为横坐标）
    for (int ty = 0; ty < height; ++ty) {
        for (int tx = 0; tx < width; ++tx) {
            float sum = 0.0f;  // 当前像素的卷积累加值

            // 遍历卷积核的每一个元素（i为纵向偏移，j为横向偏移）
            for (int i = -half_kernel; i <= half_kernel; ++i) {
                for (int j = -half_kernel; j <= half_kernel; ++j) {
                    // 计算当前卷积核元素对应的输入图像坐标
                    int x = tx + j;  // 注意：j对应横向偏移（列方向）
                    int y = ty + i;  // 注意：i对应纵向偏移（行方向）

                    // 边界检查：确保坐标在图像范围内
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        // 计算输入图像中的索引：行优先存储（y * width + x）
                        float input_value = input[y * width + x];
                        // 计算卷积核中的索引：行优先存储（(i + half_kernel) * kernel_size + (j + half_kernel)）
                        float kernel_value = kernel[(i + half_kernel) * kernel_size + (j + half_kernel)];
                        sum += input_value * kernel_value;
                    }
                }
            }

            // 将计算结果写入输出图像的对应位置
            output[ty * width + tx] = sum;
        }
    }
}

/**
 * 主函数：测试卷积算法的正确性
 */
int main() {
    // ------------ 参数定义 ------------
    int width = 1024;        // 图像宽度（像素）
    int height = 1024;       // 图像高度（像素）
    int kernel_size = 3;     // 卷积核尺寸（3x3）
    int total_pixels = width * height;  // 图像总像素数

    // ------------ 内存分配 ------------
    // 分配输入图像内存（大小为 width*height 个 float）
    float *input = (float*)malloc(total_pixels * sizeof(float));
    // 分配输出图像内存（与输入同尺寸）
    float *output = (float*)malloc(total_pixels * sizeof(float));
    // 分配卷积核内存（大小为 kernel_size*kernel_size 个 float）
    float *kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));

    // ------------ 数据初始化 ------------
    // 初始化输入图像：模拟灰度值（0-254循环）
    for (int i = 0; i < total_pixels; i++) {
        input[i] = (float)(i % 255);  // 假设灰度范围为0-254
    }

    // 初始化卷积核：3x3均值滤波器（每个元素为1/9）
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] = 1.0f / 9.0f;  // 3x3均值滤波核
    }

    // ------------ 执行卷积计算 ------------
    convolution(input, output, kernel, width, height, kernel_size);

    // ------------ 结果验证 ------------
    // 计算图像中心点坐标（假设图像尺寸为偶数）
    int center_x = width / 2;
    int center_y = height / 2;
    // 输出中心点的卷积结果值
    printf("卷积结果位于 (%d, %d): %.2f\n", 
           center_x, center_y, output[center_y * width + center_x]);

    // ------------ 释放内存 ------------
    free(input);    // 释放输入图像内存
    free(output);   // 释放输出图像内存
    free(kernel);   // 释放卷积核内存

    return 0;
}
