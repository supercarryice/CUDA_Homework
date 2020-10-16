#include<cstdio>
#include<iostream>
#include"cuda.h"
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<cmath>
#include<time.h>
#include <Windows.h>
#define N 106182300
using namespace std;

void add_with_cpu(double A[], int len) {
	double ans = 0;
	clock_t start, end;
	start = clock();
	for (int i = 0; i < len; i++) {
		ans += A[i];
	}
	end = clock();
	cout << "With cpu:    " << "ans:" << ans << "   " << "time:" << end - start << "ms" << endl;

}

__global__ static void add_with_all_atomic(double *A, int len, double *result) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	while (id < len) {
		atomicAdd(result, A[id]);
		id += gridDim.x * blockDim.x;
	}
}

__global__ static void add_with_few_atomic(double *A, int len, double *result) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	double temp = 0.0;
	while (id < len) {
		temp += A[id];
		id += gridDim.x * blockDim.x;
	}
	atomicAdd(result, temp);
}

__global__ static void add_without_atomic(double *A, double *B, int len) {

	extern __shared__ double cache[];
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	double x = 0;

	if (id < len) {
		x = A[id];
	}
	cache[threadIdx.x] = x;
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (threadIdx.x < offset)
			cache[threadIdx.x] += cache[threadIdx.x + offset];

		__syncthreads();
	}
	if (threadIdx.x == 0) {
		B[blockIdx.x] == cache[0];
	}
}




int main() {
	double *A = new double[N];
	double result = 0;
	int len;

	double *dev_A;
	double *dev_result;

	cudaMalloc((void**)&dev_A, N * sizeof(double));
	cudaMalloc((void**)&dev_result, sizeof(double));

	for (int i = 0; i < N; i++) {
		A[i] = (double)(rand() % 101) / 101;
	}
	result = 0;
	len = N;

	cudaMemcpy(dev_A, A, N * sizeof(double),
		cudaMemcpyHostToDevice);


	cudaEvent_t start, stop;
	float elapsedTime;

	// PART1 All atomic
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	add_with_all_atomic << <64, 64 >> > (dev_A, len, dev_result);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);


	cudaMemcpy(&result, dev_result, sizeof(double), cudaMemcpyDeviceToHost);


	cout << "With all atomic: " << "ans:" << result << "   " << "time:" << elapsedTime << "ms" << endl;

	//PART2 Few Atomic
	double *dev_result1;
	cudaMalloc((void**)&dev_result1, sizeof(double));
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	add_with_few_atomic << <64, 64 >> > (dev_A, len, dev_result1);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);


	cudaMemcpy(&result, dev_result1, sizeof(double), cudaMemcpyDeviceToHost);
	cout << "With few atomic: " << "ans:" << result << "   " << "time:" << elapsedTime << "ms" << endl;


	//part3 
	double *dev_result2;
	cudaMalloc((void**)&dev_result2, sizeof(double));
	const int block_size = 512;
	const int num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	double *partial_sums = 0;
	cudaMalloc((void**)&partial_sums, sizeof(double));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	add_without_atomic << <num_blocks, block_size, block_size * sizeof(double) >> > (dev_A, partial_sums, len);
	add_without_atomic << <1, num_blocks, num_blocks * sizeof(double) >> > (partial_sums, partial_sums + num_blocks, num_blocks);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(&dev_result2, partial_sums + num_blocks, sizeof(double), cudaMemcpyDeviceToHost);

	cout << "Without atomic: " << "ans:" << result << "   " << "time:" << elapsedTime << "ms" << endl;

	add_with_cpu(A, len);
}