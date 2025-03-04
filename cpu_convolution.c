/**
 * 功能：在CPU上实现图像卷积计算（支持任意奇数尺寸卷积核）
 * 特性：包含输入验证、内存管理、边界处理
 */
#include <stdio.h>
#include <stdlib.h>

void convolution(float *input, float *output, float *kernel, 
                 int width, int height, int kernel_size) {
    // 参数验证
    if (kernel_size % 2 == 0) {
        fprintf(stderr, "错误：卷积核尺寸必须为奇数\n");
        return;
    }
    
    int half_kernel = kernel_size / 2;

    // 使用嵌套循环进行二维卷积计算
    for (int ty = 0; ty < height; ++ty) {
        for (int tx = 0; tx < width; ++tx) {
            float sum = 0.0f;
            
            // 遍历卷积核邻域
            for (int i = -half_kernel; i <= half_kernel; ++i) {
                for (int j = -half_kernel; j <= half_kernel; ++j) {
                    int x = tx + j;
                    int y = ty + i;
                    
                    // 边界反射处理（根据需求可改为零填充或边界复制）
                    if (x < 0) x = -x - 1;
                    else if (x >= width) x = 2*width - x - 1;
                    if (y < 0) y = -y - 1;
                    else if (y >= height) y = 2*height - y - 1;

                    int input_idx = y * width + x;
                    int kernel_idx = (i + half_kernel) * kernel_size + (j + half_kernel);
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
            output[ty * width + tx] = sum;
        }
    }
}

int main() {
    // 参数配置
    int width = 1024;           // 图像宽度
    int height = 1024;          // 图像高度
    int kernel_size = 3;        // 卷积核尺寸
    int total_pixels = width * height;

    // 内存分配（包含错误检查）
    float *input, *output, *kernel;
    if (!(input = (float*)malloc(total_pixels * sizeof(float))) ||
        !(output = (float*)malloc(total_pixels * sizeof(float))) ||
        !(kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float)))) {
        fprintf(stderr, "内存分配失败\n");
        exit(EXIT_FAILURE);
    }

    // 数据初始化
    printf("初始化数据...\n");
    for (int i = 0; i < total_pixels; i++) 
        input[i] = (float)(i % 255);  // 模拟0-254的灰度值

    float kernel_sum = 0.0f;
    for (int i = 0; i < kernel_size*kernel_size; i++) {
        kernel[i] = 1.0f / 9.0f;     // 均值滤波器
        kernel_sum += kernel[i];
    }
    printf("卷积核权重和: %.2f\n", kernel_sum);

    // 执行卷积运算
    printf("开始卷积计算...\n");
    convolution(input, output, kernel, width, height, kernel_size);

    // 验证中心点结果
    int center_x = width/2 - 1;
    int center_y = height/2 - 1;
    printf("中心点(%d, %d)卷积结果: %.2f\n", center_x, center_y, 
           output[center_y * width + center_x]);

    // 资源清理
    free(input);
    free(output);
    free(kernel);
    return 0;
}
