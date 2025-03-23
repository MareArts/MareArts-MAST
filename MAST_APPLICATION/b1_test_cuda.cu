#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

void checkCudaError(cudaError_t error, const char *function) {
    if (error != cudaSuccess) {
        std::cerr << "Error in " << function << ": " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

int main() {
    int c;
    int *dev_c;

    // Get and print the CUDA Runtime version
    int runtimeVer = 0;
    cudaRuntimeGetVersion(&runtimeVer);
    std::cout << "CUDA Runtime Version: " << runtimeVer / 1000 << "." << (runtimeVer % 100) / 10 << std::endl;

    // Allocate memory on the GPU
    checkCudaError(cudaMalloc((void**)&dev_c, sizeof(int)), "cudaMalloc");

    // Launch the add kernel on the GPU
    add<<<1,1>>>(2, 7, dev_c);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy the result back to the host
    checkCudaError(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");

    std::cout << "2 + 7 = " << c << std::endl;

    // Free the memory allocated on the GPU
    cudaFree(dev_c);
    
    return 0;
}
