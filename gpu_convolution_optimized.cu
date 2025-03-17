#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// 常量内存仅用于3x3卷积核
__constant__ float c_kernel[9];

// 边界处理模式枚举
enum BorderMode { ZERO_PAD, MIRROR };

// 设备函数支持多种边界模式
__device__ float get_value(const float* input, int x, int y, int width, int height, BorderMode mode) {
    if (mode == ZERO_PAD) {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0.0f;
    } else { // MIRROR
        x = abs(x) % (2 * width);
        y = abs(y) % (2 * height);
        if (x >= width) x = 2 * width - x - 1;
        if (y >= height) y = 2 * height - y - 1;
    }
    return input[y * width + x];
}

// 卷积核（支持动态共享内存）
__global__ void convolution_kernel(
    const float* input, float* output, const float* d_kernel,
    int width, int height, int kernel_size, BorderMode mode
) {
    extern __shared__ float s_input[]; // 动态共享内存

    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int half_kernel = kernel_size / 2;
    const int shared_width = blockDim.x + 2 * half_kernel;

    // 共享内存索引计算
    const int s_x = threadIdx.x + half_kernel;
    const int s_y = threadIdx.y + half_kernel;
    const int shared_idx = s_y * shared_width + s_x;

    // 加载主区域
    s_input[shared_idx] = get_value(input, tx, ty, width, height, mode);

    // 边界填充
    if (threadIdx.x < half_kernel) {
        s_input[s_y * shared_width + (s_x - half_kernel)] = 
            get_value(input, tx - half_kernel, ty, width, height, mode);
        s_input[s_y * shared_width + (s_x + blockDim.x)] = 
            get_value(input, tx + blockDim.x, ty, width, height, mode);
    }
    if (threadIdx.y < half_kernel) {
        s_input[(s_y - half_kernel) * shared_width + s_x] = 
            get_value(input, tx, ty - half_kernel, width, height, mode);
        s_input[(s_y + blockDim.y) * shared_width + s_x] = 
            get_value(input, tx, ty + blockDim.y, width, height, mode);
    }
    __syncthreads();

    // 卷积计算
    float sum = 0.0f;
    for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
        for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
            const int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
            const float kernel_val = (kernel_size == 3) ? c_kernel[kernel_idx] : d_kernel[kernel_idx];
            sum += s_input[(s_y + ky) * shared_width + (s_x + kx)] * kernel_val;
        }
    }

    if (tx < width && ty < height) {
        output[ty * width + tx] = sum;
    }
}

// 导出C接口（供Python调用）
extern "C" {
void gpu_convolution(
    float* h_input, float* h_output, float* h_kernel,
    int width, int height, int kernel_size, 
    BorderMode mode, float* elapsed_ms
) {
    float *d_input, *d_output, *d_kernel = nullptr;
    size_t size = width * height * sizeof(float);
    size_t kernel_size_bytes = kernel_size * kernel_size * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // 处理卷积核
    if (kernel_size == 3) {
        CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size_bytes));
    } else {
        CHECK_CUDA(cudaMalloc(&d_kernel, kernel_size_bytes));
        CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice));
    }

    // 动态调整block大小
    dim3 block;
    if (width <= 32 || height <= 32) {
        block = dim3(8, 8);
    } else {
        block = dim3(16, 16);
    }
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 计算共享内存大小
    const int shared_mem_size = 
        (block.x + 2*(kernel_size/2)) * 
        (block.y + 2*(kernel_size/2)) * sizeof(float);

    // 计时事件
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    convolution_kernel<<<grid, block, shared_mem_size>>>(
        d_input, d_output, d_kernel, 
        width, height, kernel_size, mode
    );
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    if (elapsed_ms) *elapsed_ms = time_ms;

    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // 清理
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    if (kernel_size != 3 && d_kernel) CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}
}
