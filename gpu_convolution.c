#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__constant__ float c_kernel[9];  // 仅适用于 3x3 核
__device__ float get_value(const float *input, int x, int y, int width, int height) {
    x = max(0, min(x, width - 1));  
    y = max(0, min(y, height - 1)); 
    return input[y * width + x];
}

__global__ void convolution_kernel(const float *input, float *output,
                                   const float *d_kernel, int width, int height, int kernel_size) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= width || ty >= height) return;

    int half_kernel = kernel_size / 2;
    float sum = 0.0f;

    // **共享内存优化**（适用于中等尺寸卷积核）
    __shared__ float s_input[18][18];  
    int s_x = threadIdx.x + half_kernel;
    int s_y = threadIdx.y + half_kernel;

    // 加载当前线程块数据到共享内存
    s_input[s_y][s_x] = get_value(input, tx, ty, width, height);

    // 处理边界补充（额外填充 1 像素）
    if (threadIdx.x < half_kernel) {
        s_input[s_y][s_x - half_kernel] = get_value(input, tx - half_kernel, ty, width, height);
        s_input[s_y][s_x + blockDim.x] = get_value(input, tx + blockDim.x, ty, width, height);
    }
    if (threadIdx.y < half_kernel) {
        s_input[s_y - half_kernel][s_x] = get_value(input, tx, ty - half_kernel, width, height);
        s_input[s_y + blockDim.y][s_x] = get_value(input, tx, ty + blockDim.y, width, height);
    }
    __syncthreads();

    // 执行卷积计算
    for (int i = -half_kernel; i <= half_kernel; ++i) {
        for (int j = -half_kernel; j <= half_kernel; ++j) {
            int k_idx = (i + half_kernel) * kernel_size + (j + half_kernel);
            sum += s_input[s_y + i][s_x + j] * (kernel_size == 3 ? c_kernel[k_idx] : d_kernel[k_idx]);
        }
    }

    output[ty * width + tx] = sum;
}

void gpu_convolution(float *h_input, float *h_output, float *h_kernel,
                     int width, int height, int kernel_size) {
    float *d_input, *d_output, *d_kernel = nullptr;
    size_t size = width * height * sizeof(float);
    size_t kernel_size_bytes = kernel_size * kernel_size * sizeof(float);

    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    if (kernel_size == 3) {
        CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size_bytes));
    } else {
        CHECK_CUDA(cudaMalloc((void**)&d_kernel, kernel_size_bytes));
        CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice));
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    convolution_kernel<<<grid, block>>>(d_input, d_output, d_kernel, width, height, kernel_size);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    if (kernel_size > 3) CHECK_CUDA(cudaFree(d_kernel));
}

int main() {
    const int width = 1024, height = 1024, kernel_size = 3;
    const int total_pixels = width * height;

    float *h_input, *h_output, *h_kernel;
    CHECK_CUDA(cudaMallocHost((void**)&h_input, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_output, total_pixels * sizeof(float)));
    h_kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));

    for (int i = 0; i < total_pixels; ++i) h_input[i] = (float)(i % 255);
    
    const float gaussian_kernel[9] = {
        1/16.0f, 2/16.0f, 1/16.0f,
        2/16.0f, 4/16.0f, 2/16.0f,
        1/16.0f, 2/16.0f, 1/16.0f
    };
    memcpy(h_kernel, gaussian_kernel, sizeof(gaussian_kernel));

    gpu_convolution(h_input, h_output, h_kernel, width, height, kernel_size);

    printf("中心点结果: %.2f\n", h_output[(height / 2) * width + (width / 2)]);

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));
    free(h_kernel);
    
    return 0;
}
