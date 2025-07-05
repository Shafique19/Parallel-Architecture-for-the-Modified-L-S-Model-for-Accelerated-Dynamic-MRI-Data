// Add CUDA and C++ Libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//for __syncthreads()
#ifndef __cudacc__ 
#define __cudacc__
#endif
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
using namespace std;
// checking functions
void display_matrix_mem_real(double*, int, int);
// Host function for transpose 
void H_transpose(double*, double*, double*, double*, int, int);
// CUDA kernel for multiplication
__global__ void D_transpose(double*, double*, double*, double*, int, int);

void main()
{
	int nx = 256;
	int ny = 256;
	int kn = 11;
	double* M_real = (double*)malloc(nx * ny * kn * sizeof(double));
	double* M_imag = (double*)malloc(nx * ny * kn * sizeof(double));
	double* y_real_cpu = (double*)malloc(nx * ny * kn * sizeof(double));
	double* y_imag_cpu = (double*)malloc(nx * ny * kn * sizeof(double));
	double* y_real_gpu = (double*)malloc(nx * ny * kn * sizeof(double));
	double* y_imag_gpu = (double*)malloc(nx * ny * kn * sizeof(double));
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
	for (int i = 0; i < (nx * ny * kn); i++)
	{
		M_read_real >> file_in;
		*(M_real + i) = file_in;
		M_read_imag >> *(M_imag + i);
	}

	double sum_m = 0;
	double e = 2.0;
	for (int i = 0; i < (nx * ny * kn); i++)
	{
		sum_m = sum_m + pow(*(M_real + i), e) + pow(*(M_imag + i), e);
	}
	double norm_m = sqrt(sum_m);
	cout << "\t\t norm_M of input M_ral and M_imag is = " << norm_m << endl;
	
	// Host transpose
	int start_c = clock();
	H_transpose(y_real_cpu, y_imag_cpu, M_real, M_imag, nx*ny, kn);
	int stop_c = clock();
	cout << "\n CPU Time is :" << (stop_c - start_c) << endl;
	// printf("\n CPU Output :");
	// display_matrix_mem_real(out_real_cpu, nx*ny, nx*ny);
	// GPU_Implementation_Start
	// create a stop watch to record time for CUDA Kernel;
	cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	// Define Pointers for GPU or Device Memory
	// cudaEventRecord(start_g);
	double* d_M_real, * d_M_imag, * d_yt_real, * d_yt_imag;
	// Allocate Device Memory 
	cudaMalloc((void**)&d_M_real, sizeof(double) * nx * ny * kn);
	cudaMalloc((void**)&d_M_imag, sizeof(double) * nx * ny * kn);
	cudaMalloc((void**)&d_yt_real, sizeof(double) * nx * ny * kn);
	cudaMalloc((void**)&d_yt_imag, sizeof(double) * nx * ny * kn);
	// Copy Data from the host to the device pointer
	cudaMemcpy(d_M_real, M_real, sizeof(double) * nx * ny * kn, cudaMemcpyHostToDevice);
	cudaMemcpy(d_M_imag, M_imag, sizeof(double) * nx * ny * kn, cudaMemcpyHostToDevice);
	// Kernel configuration
	dim3 nthreads(1,512); // change
	dim3 nblocks((int)ceil(kn/ 1), (int)ceil(nx*ny/512));
	// Launch Kernel
	cudaEventRecord(start_g);
	D_transpose<< <nblocks, nthreads >> > (d_yt_real, d_yt_imag, d_M_real, d_M_imag, nx * ny, kn);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_g);
	// Copy Result from Device to Host
	cudaMemcpy(y_real_gpu, d_yt_real, sizeof(double) * nx * ny * kn, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_imag_gpu, d_yt_imag, sizeof(double) * nx * ny * kn, cudaMemcpyDeviceToHost);
	// Test Kernel Processing 
	if (cudaMemcpy(y_real_gpu, d_yt_real, sizeof(double) * nx * ny * kn, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		throw runtime_error("Error - cudaMemcpy of d_yt_real");
	};
	if (cudaMemcpy(y_imag_gpu, d_yt_imag, sizeof(double) * nx * ny * kn, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		throw runtime_error("Error - cudaMemcpy of d_yt_imag");
	};
	// Device and Thread Synchronization
	// cudaEventRecord(stop_g);
	cudaEventSynchronize(stop_g);
	// Print Time
	float t = 0;
	cudaEventElapsedTime(&t, start_g, stop_g);
	printf("\n GPU time is : %f", t);
	// Print Output
	// printf("\n GPU Output :");
	// display_matrix_mem_real(y_real_GPU, kn, nx*ny);
	// Free GPU memory
	cudaFree(d_M_real);
	cudaFree(d_M_imag);
	cudaFree(d_yt_real);
	cudaFree(d_yt_imag);
	// save output data into text files 
	ofstream my_real("out_real.txt");
	ofstream my_imag("out_imag.txt");
	for (int i = 0; i < nx * ny * kn; i++)
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

// Host Function for Transpose
void H_transpose(double* yt_real, double* yt_imag, double* A_real, double* A_imag, int A_r, int A_c)
{
	for (int i = 0; i < A_r; i++)
	{
		for (int j = 0; j < A_c; j++)
		{
			*(yt_real + j * A_r + i) = *(A_real + i * A_c + j);
			*(yt_imag + j * A_r + i) = -1 * *(A_imag + i * A_c + j);
		}
	}
}
// GPU Kernel for Transpose
__global__ void D_transpose(double* yt_real, double* yt_imag, double* A_real, double* A_imag, int A_r, int A_c)
{
	// Compute each thread's global row and column index
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < A_r && j <A_c)
	{
		int tid = i * A_c + j;
		int T_tid = j * A_r + i;
		*(yt_real + T_tid) = *(A_real + tid);
		*(yt_imag + T_tid) = -1 * *(A_imag + tid);
	}
}

// Display functions 
// Function to Dispaly Real Matrix
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