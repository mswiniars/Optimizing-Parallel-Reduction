#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

#define DIM 1024

__global__ void reductionDouble(double *vect, double *vecOut, int size)
{
	__shared__ double block[DIM];
	unsigned int globalIndex = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int i = threadIdx.x;
	if (globalIndex < size)
		block[i] = vect[globalIndex];
	else
		block[i] = 0;

	__syncthreads();

	for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1)
	{
		if (i < j)
			block[i] += block[i + j];

		__syncthreads();
	}
	if (i == 0)
		vecOut[blockIdx.x] = block[0];
}

__global__ void reductionFloat(float *vect, float *vecOut, int size)
{
	__shared__ float block[DIM];
	unsigned int globalIndex = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int i = threadIdx.x;
	if (globalIndex < size)
		block[i] = vect[globalIndex];
	else
		block[i] = 0;

	__syncthreads();

	for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1)
	{
		if (i < j)
			block[i] += block[i + j];

		__syncthreads();
	}
	if (i == 0)
		vecOut[blockIdx.x] = block[0];
}
void generate_numbers(double *vect, float *vectFloats, int size)
{
	srand(time(NULL));
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int i = 0; i < size; i++)
	{
		if (i % 2 == 0)
		{
			vectFloats[i] = (float)1 / ((float)(i * 2) + float(1));
			vect[i] = (double)1 / ((double)(i * 2) + double(1));
		}
		else
		{
			vectFloats[i] = -(float)1 / ((float)(i * 2) + float(1));
			vect[i] = -(double)1 / ((double)(i * 2) + double(1));
		}
	}
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto fillingTime = duration_cast<microseconds>(t2 - t1).count();
	cout << "Time needed to fill matrix with random number: " << (double)fillingTime / 1000 << " ms." << endl;
}

void sumCPUDouble(double *vect, double &sum, double &time, int size)
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int i = 0; i < size; i++)
		sum += vect[i];
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto timeCPU = duration_cast<microseconds>(t2 - t1).count();
	time = (double)timeCPU;// / 1000;
	cout << "Time needed to calculate the sum of Doubles by CPU: " << time << " us." << endl;
	cout << "Sum od Doubles calculated by CPU is: " << sum << " " << endl;
}
void sumCPUFloat(float *vect, float &sum, int size)
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int i = 0; i < size; i++)
		sum += vect[i];
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto timeCPU = duration_cast<microseconds>(t2 - t1).count();
	cout << "Time needed to calculate the sum of Floats by CPU: " << (double)timeCPU << " us." << endl;
	cout << "Sum of Floats calculated by CPU is: " << sum << " " << endl;
}
void show_vector(double *vect, int size)
{
	for (int i = 0; i < size; i++)
		cout << vect[i] << " ";
	cout << endl;
}
void sumGPUDouble(double *vector, double *vectorOutput, double &time, int vec_size)
{
	int numInputElements = vec_size;
	int numOutputElements;
	int threadsPerBlock = DIM;
	double *dev_vec;
	double *dev_vecOut;
	float dev_time;
	cudaEvent_t start, stop;
	cudaSetDevice(0);
	cudaMalloc((double**)&dev_vec, vec_size * sizeof(double));
	cudaMalloc((double**)&dev_vecOut, vec_size * sizeof(double));
	cudaMemcpy(dev_vec, vector, vec_size * sizeof(double), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	do
	{
		numOutputElements = numInputElements / (threadsPerBlock);
		if (numInputElements % (threadsPerBlock))
			numOutputElements++;
		reductionDouble << < numOutputElements, threadsPerBlock >> > (dev_vec, dev_vecOut, numInputElements);
		numInputElements = numOutputElements;
		if (numOutputElements > 1)
			reductionDouble << < numOutputElements, threadsPerBlock >> > (dev_vecOut, dev_vec, numInputElements);

	} while (numOutputElements > 1);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto timeCPU = duration_cast<microseconds>(t2 - t1).count();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&dev_time, start, stop);
	time = (double)timeCPU;

	cudaDeviceSynchronize();
	cudaMemcpy(vector, dev_vec, vec_size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(vectorOutput, dev_vecOut, vec_size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_vec);
	cudaFree(dev_vecOut);
}
void sumGPUFloat(float *vector, float *vectorOutput, double &time, int vec_size)
{
	int numInputElements = vec_size;
	int numOutputElements;
	int threadsPerBlock = DIM;
	float *dev_vec;
	float *dev_vecOut;
	float dev_time;
	cudaEvent_t start, stop;
	cudaSetDevice(0);
	cudaMalloc((float**)&dev_vec, vec_size * sizeof(float));
	cudaMalloc((float**)&dev_vecOut, vec_size * sizeof(float));
	cudaMemcpy(dev_vec, vector, vec_size * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	do
	{
		numOutputElements = numInputElements / (threadsPerBlock);
		if (numInputElements % (threadsPerBlock))
			numOutputElements++;

		reductionFloat << < numOutputElements, threadsPerBlock >> > (dev_vec, dev_vecOut, numInputElements);
		numInputElements = numOutputElements;
		if (numOutputElements > 1)
			reductionFloat << < numOutputElements, threadsPerBlock >> > (dev_vecOut, dev_vec, numInputElements);

	} while (numOutputElements > 1);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto timeCPU = duration_cast<microseconds>(t2 - t1).count();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&dev_time, start, stop);
	time = (double)timeCPU;

	cudaDeviceSynchronize();
	cudaMemcpy(vector, dev_vec, vec_size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(vectorOutput, dev_vecOut, vec_size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_vec);
	cudaFree(dev_vecOut);
}

int main()
{
	int vec_size;
	while (true)
	{
		cout << "How big matrix do you want to create: ";
		cin >> vec_size;

		double *vector = new double[vec_size];
		double *vecOutput = new double[vec_size];
		float *vectorFloats = new float[vec_size];
		float *vecOutputFloats = new float[vec_size];
		double sumCPU = 0;
		float sumCPUFl = 0;
		double timeCPU, timeGPUDoubles, timeGPUFloats;

		generate_numbers(vector, vectorFloats, vec_size);
		sumCPUDouble(vector, sumCPU, timeCPU, vec_size);
		sumCPUFloat(vectorFloats, sumCPUFl, vec_size);
		sumGPUDouble(vector, vecOutput, timeGPUDoubles, vec_size);
		sumGPUFloat(vectorFloats, vecOutputFloats, timeGPUFloats, vec_size);

		cout << "Time needed to calculate the sum of Doubles by GPU: " << timeGPUDoubles << " us." << endl;
		cout << "Sum of Doubles calculated by GPU: " << setprecision(12) << vecOutput[0] << endl;

		cout << "Time needed to calculate the sum of Floats by GPU: " << timeGPUFloats << " us." << endl;
		cout << "Sum of Floats calculated by GPU: " << setprecision(12) << vecOutputFloats[0] << endl;

		cout << "Difference between sums of Double: " << setprecision(12) << vecOutput[0] - sumCPU << endl;
		cout << "Difference between sums of Floats: " << setprecision(12) << vecOutputFloats[0] - sumCPUFl << endl;

		if (sumCPU == vecOutput[0])
			cout << "Sum calculated by CPU is equal to sum calculated by GPU." << endl;


		if (sumCPUFl == vecOutputFloats[0])
			cout << "Sum of Floats calculated by CPU is equal to sum calculated by GPU." << endl;


		cout << "-------------------------------------------" << endl;
		cout << "SpeedUP CPU to GPU: " << timeCPU / timeGPUDoubles << " times." << endl;
		cout << "SpeedUP GPU Doubles to GPU Floats: " << timeGPUDoubles / timeGPUFloats << endl;

		delete[] vector;
		delete[] vecOutput;
		delete[] vecOutputFloats;
		delete[] vectorFloats;
	}

	return 0;
}