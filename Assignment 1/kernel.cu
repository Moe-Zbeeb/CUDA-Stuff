
#include "common.h"
#include "timer.h"

__global__ void axpb_kernel(double* x, double* y, double a, double b, unsigned int M) {

    // TODO Progress  
    int IdxThread = blockDim.x * blockIdx.x + threadIdx.x ;   
    if( IdxThread < M ){
        y[IdxThread] = a*x[IdxThread] + b; 
    }
}

void axpb_gpu(double* x, double* y, double a, double b, unsigned int M) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);

    // TODO   done 
    double *x_d, *y_d; 
    cudaMalloc((void**) &x_d, M*sizeof(double));    
    cudaMalloc((void**) &y_d, M*sizeof(double)); 





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO  done
    cudaMemcpy(x_d, x, M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, M*sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO Done 
    const unsigned int numThreadsPerBlock = 1024;    
    const unsigned int numBlocks = (M + numThreadsPerBlock - 1 )/numThreadsPerBlock; 
    axpb_kernel <<< numBlocks, numThreadsPerBlock >>> (x_d, y_d, a, b, M);




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO done  
    cudaMemcpy(y, y_d, M*sizeof(double), cudaMemcpyDeviceToHost); 


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO 
    cudaFree(x_d);
    cudaFree(y_d);  



    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

