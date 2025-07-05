// Add CUDA and C++ Libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
//for __syncthreads()
#ifndef __cudacc__ 
#define __cudacc__
#endif
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
using namespace std;
// C Functions
void display_matrix_mem_real(double*, int, int);
void H_softThresholding(double*, double*, double*, double*, int, int, int, double);
// CUDA kernel 
__global__ void D_softThresholding(double*, double*, double*, double*, int, int, int, double);
void main()
{
	int nx = 256;
	int ny = 256;
	int nt = 11;
	double p = 0.0001; 
	double* M_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* M_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y_real_cpu = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y_imag_cpu = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y_real_gpu = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y_imag_gpu = (double*)malloc(nx * ny * nt * sizeof(double));
	// Reading Data from the files
	cout << "\t\t Opening File to Read .......................................................\n";
	ifstream M_read_real("M_real.txt");
	ifstream M_read_imag("M_imag.txt");
	// Checking if all the files available in root directory or not
	if (!M_read_real || !M_read_imag)
	{
		cout << "Cannot open text files \n";
		(void)getchar();
		return;
	}
	else
		cout << "All files found:- \t Reading Data In Progress \n";
	//Files reading started 
	cout << "\t\t Files Reading Started ......................................................\n";
	double file_in = 0.0;
	cout << "\t\t Reading Input Data .........................................................\n";
	// M real and Cmplex 
	for (int i = 0; i < (nx * ny * nt); i++)
	{
		M_read_real >> file_in;
		*(M_real + i) = file_in;
		M_read_imag >> *(M_imag + i);
	}

	double sum_m = 0;
	double e = 2.0;
	for (int i = 0; i < (nx * ny * nt); i++)
	{
		sum_m = sum_m + pow(*(M_real + i), e) + pow(*(M_imag + i), e);
	}
	double norm_m = sqrt(sum_m);
	cout << "\t\t norm_M of input M_ral and M_imag is = " << norm_m << endl;
	// Host multiplication
	int start_c = clock();
	H_softThresholding(y_real_cpu, y_imag_cpu, M_real, M_imag,nx, ny, nt, p);
	int stop_c = clock();
	cout << "\n CPU Time of soft_thresholding is :" << (stop_c - start_c) << endl;
	// printf("\n CPU Output :");
	// display_matrix_mem_real(out_real_cpu, nx*ny, nx*ny);
	// GPU_Implementation_Start
	// create a stop watch to record time for CUDA Kernel;
	cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	// Define Pointers for GPU or Device Memory
	// cudaEventRecord(start_g);
	double* d_M_real, * d_M_imag, * d_y_real, * d_y_imag;
	// Allocate Device Memory 
	cudaMalloc((void**)&d_M_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_M_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_y_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_y_imag, sizeof(double) * nx * ny * nt);
	// Copy Data from the host to the device pointer
	cudaMemcpy(d_M_real, M_real, sizeof(double) * nx * ny * nt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_M_imag, M_imag, sizeof(double) * nx * ny * nt, cudaMemcpyHostToDevice);
	// Kernel configuration
	// dim3 nthreads(1, 512); // change
	// dim3 nblocks((int)ceil(nt / 1), (int)ceil(nx * ny / 512));
	// Launch Kernel
	cudaEventRecord(start_g);
	D_softThresholding <<<nx*ny, nt>> > (d_y_real, d_y_imag, d_M_real, d_M_imag, nx, ny, nt, p);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_g);
	// Copy Result from Device to Host
	cudaMemcpy(y_real_gpu, d_y_real, sizeof(double) * nx * ny * nt, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_imag_gpu, d_y_imag, sizeof(double) * nx * ny * nt, cudaMemcpyDeviceToHost);
	// Test Kernel Processing 
	if (cudaMemcpy(y_real_gpu, d_y_real, sizeof(double) * nx * ny * nt, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		throw runtime_error("Error - cudaMemcpy of d_y_real");
	};
	if (cudaMemcpy(y_imag_gpu, d_y_imag, sizeof(double) * nx * ny * nt, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		throw runtime_error("Error - cudaMemcpy of d_y_imag");
	};
	// Device and Thread Synchronization
	// cudaEventRecord(stop_g);
	// cudaEventSynchronize(stop_g);
	// Print Time
	float t = 0;
	cudaEventElapsedTime(&t, start_g, stop_g);
	printf("\n GPU time is : %f", t);
	// Print Output
	// printf("\n GPU Output :");
	// display_matrix_mem_real(M_imag_GPU, nx*ny, nx*ny);
	// Free GPU memory
	cudaFree(d_M_real);
	cudaFree(d_M_imag);
	cudaFree(d_y_real);
	cudaFree(d_y_imag);
	// save output data into text files 
	ofstream my_real("out_real.txt");
	ofstream my_imag("out_imag.txt");
	for (int i = 0; i < nx * ny * nt; i++)
	{
		double out_r = 0.0;
		double out_i = 0.0;
		out_r = *(y_real_gpu + i);
		my_real << out_r << endl;;
		out_i = *(y_imag_gpu + i);
		my_imag << out_i << endl;
	}
	// close files after saving data
	my_real.close();
	my_imag.close();
}



// C function definition 
void H_softThresholding(double* y_real, double* y_imag, double* s_real, double* s_imag, int nx, int ny, int nt, double p)
{
	double temp_real = 0, temp_imag = 0;
	double max;
	double abs_in;
	for (int i = 0; i < nx * ny * nt; i++)
	{
		double e = 2.0;
		abs_in = sqrt(pow(*(s_real + i), e) + pow(*(s_imag + i), e));
		if (abs_in != 0)
		{
			temp_real = *(s_real + i) / abs_in;
			temp_imag = *(s_imag + i) / abs_in;
		}
		if (abs_in - p > 0)
			max = abs_in - p;
		else
			max = 0;
		*(y_real + i) = temp_real * max;
		*(y_imag + i) = temp_imag * max;
	}
}
// CUDA kernel 
__global__ void D_softThresholding(double* y_real, double* y_imag, double* s_real, double* s_imag, int nx, int ny, int nt, double p)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double temp_real = 0, temp_imag = 0;
	double max;
	double abs_in;
	double e = 2.0;
	if (i < nx*ny*nt)
	//for (int i = 0; i < nx * ny * nt; i++)
	//{
		abs_in = sqrt(pow(*(s_real + i), e) + pow(*(s_imag + i), e));
		if (abs_in != 0)
		{
			temp_real = *(s_real + i) / abs_in;
			temp_imag = *(s_imag + i) / abs_in;
		}
		if (abs_in - p > 0)
			max = abs_in - p;
		else
			max = 0;
		*(y_real + i) = temp_real * max;
		*(y_imag + i) = temp_imag * max;
	//}
}

// Display function
void display_matrix_mem_real(double* A, int row, int col)
{
	cout << "\n\n";
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << *(A + i * col + j) << "\t";
		}
		cout << "\n";
	}
}


// The end 