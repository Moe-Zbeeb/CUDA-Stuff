#include "timer.h"  

void vecadd_cpu(float* x, float* y, float* z,int N){
    for (int i=0; i<N; ++i){
            z[i] = x[i] + y[i]  
    }  
} 



__global__ void vecadd_kernel(x_d, y_d, z_d, N){
    int i = blockDim.x*blockIdx.x*threadIdx.x; 
    if(i<N){
    z_d[i] = x_d[i] + y_d[i]; 
    } 
 }


void vecadd_gpu(float* x, float* x, float* z){
 
    //Allcote memory on device     
    float *x_d, *y_d, *z_d;                
    cudaMalloc((void**) &x_d, N*&sizeof(float));
    cudaMalloc((void**) &y_d, N*&sizeof(float));
    cudaMalloc((void**) &x_d, N*&sizeof(float));


    //copy data from host to device  
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice)  ; 
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice)   ;
    

    //call a kernel    
    const unsigned int numThreadsPerBlock = 512    
    const unsigned int numBlocks = (N + numThreadsPerBlock -1)/numThreadsPerBlock;   

    vecadd_kernel <<<numThreadsPerBlock, numBlocks >>> (x_d, y_d, z_d, N); 

    //copy data back to host  
    cudaMemcpy(z_d, z, N*sizeof(float), cudaMemcpyHostToDevice); 


    //free the device memery    
    cudafree(x_d);
    cudafree(y_d);
    cudafree(z_d) ;

}


int main(int argc, char** argv) {
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int N = (argc > 1) ? (atoi(argv[1])) : (1 << 25);
    float* x = (float*) malloc(N * sizeof(float));
    float* y = (float*) malloc(N * sizeof(float));
    float* z = (float*) malloc(N * sizeof(float));
    for (unsigned int i = 0; i < N; ++i) {
        x[i] = rand();
        y[i] = rand();
    }

    // Vector addition on CPU
    startTime(&timer);
    vecadd_cpu(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Vector addition on GPU
    startTime(&timer);
    vecadd_gpu(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Free memory
    free(x);
    free(y);
    free(z);

    return 0;
}
