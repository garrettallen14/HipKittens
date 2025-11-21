#include <hip/hip_runtime.h>
#include <iostream>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d\n", (int)threadIdx.x);
}

int main() {
    std::cout << "Hello from CPU!" << std::endl;
    
    hipLaunchKernelGGL(hello_kernel, dim3(1), dim3(4), 0, 0);
    hipDeviceSynchronize();
    
    std::cout << "Kernel execution complete." << std::endl;
    return 0;
}
