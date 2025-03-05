#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 常量内存声明（适合小尺寸卷积核）
__constant__ float c_kernel[9]; // 最大支持3x3卷积核

// CUDA错误检查宏
#define CHECK_CUDA(call) {                                         \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                             \
}

// CUDA卷积内核（使用反射边界处理）
__global__ void convolution_kernel(const float* input, float* output,
                                   int width, int height, int kernel_size) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tx >= width || ty >= height) return;

    int half_kernel = kernel_size / 2;
    float sum = 0.0f;

    for (int i = -half_kernel; i <= half_kernel; ++i) {
        for (int j = -half_kernel; j <= half_kernel; ++j) {
            int x = tx + j;
            int y = ty + i;
            
            // 反射边界处理
            x = abs(x); if (x >= width) x = 2*width - x - 1;
            y = abs(y); if (y >= height) y = 2*height - y - 1;

            int kernel_idx = (i + half_kernel) * kernel_size + (j + half_kernel);
            sum += input[y * width + x] * c_kernel[kernel_idx];
        }
    }
    output[ty * width + tx] = sum;
}

void gpu_convolution(float *h_input, float *h_output, float *h_kernel,
                     int width, int height, int kernel_size) {
    // 设备内存指针
    float *d_input, *d_output;
    size_t size = width * height * sizeof(float);

    // 1. 分配设备内存
    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));

    // 2. 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, h_kernel, 
                kernel_size*kernel_size*sizeof(float)));

    // 3. 配置执行参数
    dim3 block(16, 16); // 256 threads per block
    dim3 grid((width + block.x - 1)/block.x, 
              (height + block.y - 1)/block.y);

    // 4. 启动内核
    convolution_kernel<<<grid, block>>>(d_input, d_output, width, height, kernel_size);
    CHECK_CUDA(cudaGetLastError());

    // 5. 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // 6. 释放设备内存
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const int kernel_size = 3;
    const int total_pixels = width * height;

    // 主机内存分配（对齐分配提升传输效率）
    float *h_input, *h_output, *h_kernel;
    CHECK_CUDA(cudaMallocHost((void**)&h_input, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_output, total_pixels * sizeof(float)));
    h_kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));

    // 数据初始化
    printf("初始化数据...\n");
    for (int i = 0; i < total_pixels; ++i)
        h_input[i] = (float)(i % 255);

    // 高斯模糊核 (3x3)
    const float gaussian_kernel[9] = {
        1/16.0f, 2/16.0f, 1/16.0f,
        2/16.0f, 4/16.0f, 2/16.0f,
        1/16.0f, 2/16.0f, 1/16.0f
    };
    memcpy(h_kernel, gaussian_kernel, sizeof(gaussian_kernel));

    // 执行GPU卷积
    printf("启动CUDA卷积...\n");
    gpu_convolution(h_input, h_output, h_kernel, width, height, kernel_size);

    // 验证结果
    int center_idx = (height/2) * width + (width/2);
    printf("中心点结果: %.2f\n", h_output[center_idx]);

    // 释放资源
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));
    free(h_kernel);
    
    return 0;
}
