#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

//nvidia reduce pdf url :https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

// Parallel Reduction: Interleaved Addressing
extern __global__ void reduce0(int *g_idata, int *g_odata);
//0.26ms
__global__ void reduce0(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


//0.26
//Reduction #2: Interleaved Addressing
extern __global__ void reduce1(int *g_idata, int *g_odata);
__global__ void reduce1(int *g_idata, int *g_odata) {
extern __shared__ int sdata[];
// each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
sdata[tid] = g_idata[i];
__syncthreads();
// do reduction in shared mem
//problem: highly divergent warps are very inefficient,and % operator is slow
for (unsigned int s=1; s < blockDim.x; s *= 2) {
if (tid % (2*s) == 0) {
sdata[tid] += sdata[tid + s];
}

__syncthreads();
}
// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


//0.24
//Parallel Reduction: Interleaved Addressing
extern __global__ void reduce2(int *g_idata, int *g_odata);
__global__ void reduce2(int *g_idata, int *g_odata) {
extern __shared__ int sdata[];
// each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
sdata[tid] = g_idata[i];
__syncthreads();
// do reduction in shared mem
//problem:shared memory bank conflicts
for (unsigned int s=1; s < blockDim.x; s *= 2) {
int index = 2 * s * tid;
if (index < blockDim.x) {
sdata[index] += sdata[index + s];
}
__syncthreads();
}
// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


//0.235 Parallel Reduction: Sequential Addressing
extern __global__ void reduce3(int *g_idata, int *g_odata);
__global__ void reduce3(int *g_idata, int *g_odata) {
extern __shared__ int sdata[];
// each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
sdata[tid] = g_idata[i];
__syncthreads();
// do reduction in shared mem
// sequential addressing is conflict free
//problem:Half of the threads are idle on first loop iteration!

for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();

}

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

//0.235 Reduction #4: First Add During Load
//Instruction Bottleneck
extern __global__ void reduce4(int *g_idata, int *g_odata);
__global__ void reduce4(int *g_idata, int *g_odata) {
extern __shared__ int sdata[];

//perform first level of reduction,
//reading from global memory, writing to shared memory
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
__syncthreads();

// do reduction in shared mem
// sequential addressing is conflict free
//problem:Half of the threads are idle on first loop iteration!

for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();

}

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}


//unroll loops
/*
Unrolling the Last Warp
As reduction proceeds, # “active” threads decreases
When s <= 32, we have only one warp left
Instructions are SIMD synchronous within a warp
That means when s <= 32:
We don’t need to __syncthreads()
We don’t need “if (tid < s)” because it doesn’t save any
work
Let’s unroll the last 6 iterations of the inner loop
*/
//important:for this to be correct ,we must use the "volatile" keyword!
__device__ void warpReduce (volatile int*sdata,int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}


/*
Note: This saves useless work in all warps, not just the last one!
Without unrolling, all warps execute every iteration of the for loop and if statement 
*/
extern __global__ void reduce5(int *g_idata, int *g_odata);
__global__ void reduce5(int *g_idata, int *g_odata) {
extern __shared__ int sdata[];

//perform first level of reduction,
//reading from global memory, writing to shared memory
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
__syncthreads();

// do reduction in shared mem
// sequential addressing is conflict free
//problem:Half of the threads are idle on first loop iteration!

for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();

}
if (tid < 32) warpReduce(sdata,tid);

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}



template <unsigned int blockSize>
__device__ void warpReduce (volatile int*sdata,int tid) {
if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


template <unsigned int blockSize>
extern __global__ void reduce6(int *g_idata, int *g_odata);

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata) {
extern __shared__ int sdata[];

//perform first level of reduction,
//reading from global memory, writing to shared memory
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
__syncthreads();

// do reduction in shared mem
// sequential addressing is conflict free
//problem:Half of the threads are idle on first loop iteration!

for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
    if (blockSize >=512){
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >=256){
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >=128){
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
}
if (tid < 32) warpReduce<blockSize>(sdata,tid);

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}




//final kernel
template <unsigned int blockSize>
extern __global__ void reduce7(int *g_idata, int *g_odata,int n);

/*
Combine sequential and parallel reduction
    Each thread loads and sums multiple elements into
shared memory
    Tree-based reduction in shared memory
Brent’s theorem says each thread should sum
O(log n) elements
    i.e. 1024 or 2048 elements per block vs. 256
In my experience, beneficial to push it even further
    Possibly better latency hiding with more work per thread
    More threads per block reduces levels in tree of recursive
kernel invocations
    High kernel launch overhead in last levels with few blocks
On G80, best perf with 64-256 blocks of 128 threads
1   024-4096 elements per thread
*/
template <unsigned int blockSize>
__global__ void reduce7(int *g_idata, int *g_odata,int n ) {
extern __shared__ int sdata[];

//perform first level of reduction,
//reading from global memory, writing to shared memory
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
unsigned int gridSize = blockSize * 2 * gridDim.x;
sdata[tid] = 0;

while(i<n){
    sdata[tid] += g_idata[i] + g_idata[i+blockSize];
    //Note:gridSize loop stride to maintain coalescing!
    i += gridSize;
}
__syncthreads();

// do reduction in shared mem
// sequential addressing is conflict free
//problem:Half of the threads are idle on first loop iteration!

for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
    if (blockSize >=512){
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >=256){
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >=128){
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
}
if (tid < 32) warpReduce<blockSize>(sdata,tid);

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}



void test_reduce(int size) {
    // Allocate host memory
    int *h_idata = (int*)malloc(size * sizeof(int));
    int *h_odata = (int*)malloc((size/1024) * sizeof(int));

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_idata[i] = 1;
    }

    // Allocate device memory
    int *d_idata, *d_odata;
    cudaMalloc(&d_idata, size * sizeof(int));
    cudaMalloc(&d_odata, (size/1024) * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_idata, h_idata, size * sizeof(int), cudaMemcpyHostToDevice);

    // Setup execution parameters
    int threads = 1024;
    int blocks = 512;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Launch kernel
    // reduce5<<<blocks, threads, threads*sizeof(int)>>>(d_idata, d_odata);

    //reduce6 
    switch (threads)
        {
        case 512:
        reduce7<512><<< blocks, threads, threads*sizeof(int) >>>(d_idata, d_odata,size); break;
        case 256:
        reduce7<256><<< blocks, threads, threads*sizeof(int) >>>(d_idata, d_odata,size); break;
        case 128:
        reduce7<128><<< blocks, threads, threads*sizeof(int) >>>(d_idata, d_odata,size); break;
        case 64:
        reduce7< 64><<< blocks, threads, threads*sizeof(int) >>>(d_idata, d_odata,size); break;
        case 32:
        reduce7< 32><<< blocks, threads, threads*sizeof(int) >>>(d_idata, d_odata,size); break;
        case 16:
        reduce7< 16><<< blocks, threads, threads*sizeof(int) >>>(d_idata, d_odata,size); break;
        case 8:
        reduce7< 8><<< blocks, threads, threads*sizeof(int) >>>(d_idata, d_odata,size); break;
        case 4:
        reduce7< 4><<< blocks, threads, threads*sizeof(int) >>>(d_idata, d_odata,size); break;
        case 2:
        reduce7< 2><<< blocks, threads, threads*sizeof(int) >>>(d_idata, d_odata,size); break;
        case 1:
        reduce7< 1><<< blocks, threads, threads*sizeof(int) >>>(d_idata, d_odata,size); break;
        }

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_odata, d_odata, (size/1024) * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify result
    printf("Kernel execution time: %f ms\n", milliseconds);
    printf("First block result: %d\n", h_odata[0]);

    // Cleanup
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int size = 4096 * 1024; // 1M elements
    test_reduce(size);
    return 0;
}