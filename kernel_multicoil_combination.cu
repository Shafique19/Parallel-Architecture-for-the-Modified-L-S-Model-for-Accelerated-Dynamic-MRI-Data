// Add CUDA and C++ Libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//for __syncthreads()
#ifndef __cudacc__
#define __cudacc__
#endif
#include <stdio.h>
#include <tchar.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <ctime>
using namespace std;
__global__ void D_mulsqre(double*, double*, double*, double*, double*, double*, double*, double*, double*, int, int, int);
void H_mulsqre(double*, double*, double*, double*, double*, double*, double*, double*, double*, int, int, int);
void main()
{
	int nx = 128;
	int ny = 128;
	int nt = 40;
	int nc = 12;
	// Allocate memory for input data
	double* x_array_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* x_array_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* conj_a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* conj_a_b1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* a_b1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	// Allocate memory for output data
	double* res_2_real = (double*)malloc(nx * ny * nc * nt * sizeof(double));
	double* res_2_imag = (double*)malloc(nx * ny * nc * nt * sizeof(double));
	double* res_2_real_gpu = (double*)malloc(nx * ny * nc * nt * sizeof(double));
	double* res_2_imag_gpu = (double*)malloc(nx * ny * nc * nt * sizeof(double));
	// Reading files from directory
	cout << "\t\t Opening File to Read .......................................................\n";
	ifstream x_read_real("x_array_real.txt");
	ifstream x_read_imag("x_array_imag.txt");
	ifstream c_read_real("conj_a_b1_real.txt");
	ifstream c_read_imag("conj_a_b1_imag.txt");
	ifstream a_read_real("a_b1_real.txt");
	ifstream a_read_imag("a_b1_imag.txt");
	// Check if all files available in directory
	if (!x_read_real || !x_read_imag || !c_read_real || !c_read_imag || !a_read_imag || !a_read_real)
	{
		cout << "Cannot open text files \n";
		(void)getchar();
		return;
	}
	else
		cout << "All files found:- \t Reading Data In Progress \n";
	//Files reading started
	cout << "\t\t Files Reading Started .........................................................\n";
	double file_in = 0.0;
	cout << "\t\t Reading Input Data .........................................................\n";
	// x real and Cmplex
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		x_read_real >> file_in;
		*(x_array_real + i) = file_in;
		x_read_imag >> *(x_array_imag + i);
	}
	double sum_x = 0;
	double e = 2.0;
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		sum_x = sum_x + pow(*(x_array_real + i), e) + pow(*(x_array_imag + i), e);
	}
	double norm_x = sqrt(sum_x);
	cout << "\t\t norm_x of input x_array_real and x_array_imag is = " << norm_x << endl;
	// c_real and complex
	for (int i = 0; i < (nx * ny * nc); i++)
	{
		c_read_real >> file_in;
		*(conj_a_b1_real + i) = file_in;
		c_read_imag >> *(conj_a_b1_imag + i);
	}
	double sum_c = 0;
	for (int i = 0; i < (nx * ny * nc); i++)
	{
		sum_c = sum_c + pow(*(conj_a_b1_real + i), e) + pow(*(conj_a_b1_imag + i), e);
	}
	double norm_c = sqrt(sum_c);
	cout << "\t\t norm_c of input conj_real and conj_imag is = " << norm_c << endl;
	// a_real and complex
	for (int i = 0; i < (nx * ny * nc); i++)
	{
		a_read_real >> file_in;
		*(a_b1_real + i) = file_in;
		a_read_imag >> *(a_b1_imag + i);
	}
	double sum_a = 0;
	for (int i = 0; i < (nx * ny * nc); i++)
	{
		sum_a = sum_a + pow(*(a_b1_real + i), e) + pow(*(a_b1_imag + i), e);
	}
	double norm_a = sqrt(sum_a);
	cout << "\t\t norm_a of input a_b1_real and a_b1_imag is = " << norm_a << endl;
	// Host main code
	// Allocate Host memory
	double* x_array_1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_a_b1_real_cpu = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_a_b1_imag_cpu = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_a_b1_real_gpu = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_a_b1_imag_gpu = (double*)malloc(nx * ny * nc * sizeof(double));
	double* abs_a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));

	for (int tt = 0; tt < nt; tt++)
	{
		for (int i = 0; i < nc; i++)
		{
			for (int j = 0; j < nx * ny; j++)
			{
				*(x_array_1_real + nx * ny * i + j) = *(x_array_real + nx * ny * tt + nx * ny * nt * i + j);
				*(x_array_1_imag + nx * ny * i + j) = *(x_array_imag + nx * ny * tt + nx * ny * nt * i + j);
			}
		}

		// CPU (Host) computation time analysis
		// int start_c = clock();
		// H_mulsqre(x_array_a_b1_real_cpu, x_array_a_b1_imag_cpu, abs_a_b1_real, x_array_1_real, x_array_1_imag, conj_a_b1_real, conj_a_b1_imag, a_b1_real, a_b1_imag, nx, ny, nc);
		// int stop_c = clock();
		// cout << "\n CPU_computation_Time:" << (stop_c - start_c) << endl;
		// CUDA implementation
		cudaEvent_t start_g, stop_g;
		cudaEventCreate(&start_g);
		cudaEventCreate(&stop_g);
		// cudaEventRecord(start_g);
		// Define output Pointers for GPU
		double*d_x_array_a_b1_real, *d_x_array_a_b1_imag, * d_abs_a_b1_real;
		cudaMalloc((void**)&d_x_array_a_b1_real, sizeof(double) * nx * ny * nc);
		cudaMalloc((void**)&d_x_array_a_b1_imag, sizeof(double) * nx * ny * nc);
		cudaMalloc((void**)&d_abs_a_b1_real, sizeof(double) * nx * ny * nc);
		// Define Input pointer
		double* d_x_array_1_real, * d_x_array_1_imag, * d_conj_a_b1_real, * d_conj_a_b1_imag, * d_a_b1_real, * d_a_b1_imag;
		cudaMalloc((void**)&d_x_array_1_real, sizeof(double) * nx * ny * nc);
		cudaMalloc((void**)&d_x_array_1_imag, sizeof(double) * nx * ny * nc);
		cudaMalloc((void**)&d_conj_a_b1_real, sizeof(double) * nx * ny * nc);
		cudaMalloc((void**)&d_conj_a_b1_imag, sizeof(double) * nx * ny * nc);
		cudaMalloc((void**)&d_a_b1_real, sizeof(double) * nx * ny * nc);
		cudaMalloc((void**)&d_a_b1_imag, sizeof(double) * nx * ny * nc);
		// Copy Data from the host to the device pointer
		cudaMemcpy(d_x_array_1_real, x_array_real, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
		cudaMemcpy(d_x_array_1_imag, x_array_imag, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
		cudaMemcpy(d_conj_a_b1_real, conj_a_b1_real, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
		cudaMemcpy(d_conj_a_b1_imag, conj_a_b1_imag, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
		cudaMemcpy(d_a_b1_real, a_b1_real, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
		cudaMemcpy(d_a_b1_imag, a_b1_imag, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
		// Create threads and Blocks for CUDA kernel
		dim3 threads(1, 512);
		dim3 blocks((int)ceil(1 / 1), (int)ceil(nx*ny*nc/512));
		cudaEventRecord(start_g);
		D_mulsqre <<< blocks, threads >>> (d_x_array_a_b1_real, d_x_array_a_b1_imag, d_abs_a_b1_real, d_x_array_1_real, d_x_array_1_imag, d_conj_a_b1_real, d_conj_a_b1_imag, d_a_b1_real, d_a_b1_imag, nx, ny, nc);
		cudaDeviceSynchronize();
		cudaEventRecord(stop_g);
		// Copy Result from Device to Host
		cudaMemcpy(x_array_a_b1_real_gpu, d_x_array_a_b1_real, sizeof(double) * nx * ny * nc, cudaMemcpyDeviceToHost);
		cudaMemcpy(x_array_a_b1_imag_gpu, d_x_array_a_b1_imag, sizeof(double) * nx * ny * nc, cudaMemcpyDeviceToHost);
		cudaMemcpy(abs_a_b1_real, d_abs_a_b1_real, sizeof(double) * nx * ny * nc, cudaMemcpyDeviceToHost);
		// call a CUDA kernel(or launch a kernel);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
			// Possibly: exit(-1) if program cannot continue....
		}
		// sync
		// cudaEventRecord(stop_g);
		cudaEventSynchronize(stop_g);
		// Print Time
		float t = 0;
		cudaEventElapsedTime(&t, start_g, stop_g);
		printf("\n GPU_First_Time: %f", t);
		// Free GPU memory
		cudaFree(d_x_array_a_b1_real);
		cudaFree(d_x_array_a_b1_imag);
		cudaFree(d_abs_a_b1_real);
		cudaFree(d_x_array_1_real);
		cudaFree(d_x_array_1_imag);
		cudaFree(d_conj_a_b1_real);
		cudaFree(d_conj_a_b1_imag);
		cudaFree(d_a_b1_real);
		cudaFree(d_a_b1_imag);
		// save output data into text files
		ofstream my_real("out_real.txt");
		ofstream my_imag("out_imag.txt");
		for (int i = 0; i < nx * ny * nc; i++)
		{
			double out_r = 0.0;
			double out_i = 0.0;
			out_r = *(x_array_a_b1_real_cpu + i);
			my_real << out_r << endl;;
			out_i = *(x_array_a_b1_imag_cpu + i);
			my_imag << out_i << endl;
		}
		// close files after saving data
		my_real.close();
		my_imag.close();
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				double sumx_real = 0, sumx_imag = 0;
				double sumb1 = 0;
				for (int k = 0; k < nc; k++)
				{
					sumx_real = sumx_real + *(x_array_a_b1_real_cpu + nx * ny * k + nx * i + j);
					sumx_imag = sumx_imag + *(x_array_a_b1_imag_cpu + nx * ny * k + nx * i + j);
					sumb1 = sumb1 + *(abs_a_b1_real + nx * ny * k + nx * i + j);
				}
				*(res_2_real + nx * ny * tt + ny * i + j) = sumx_real / sumb1;
				*(res_2_imag + nx * ny * tt + ny * i + j) = sumx_imag / sumb1;
			}
		}

	}
}

__global__ void D_mulsqre(double* d_x_array_a_b1_real,  double* d_x_array_a_b1_imag, double* d_abs_a_b1_real, double* d_x_array_1_real, double* d_x_array_1_imag, double* d_conj_a_b1_real,double* d_conj_a_b1_imag, double* d_a_b1_real, double* d_a_b1_imag, int  nx, int ny, int nc)
{
	int uid = blockIdx.y * blockDim.y + threadIdx.y;
	double e = 2.0;
	// for (int i = 0; i < nx * ny * nc; i++)
	// {
		*(d_x_array_a_b1_real + uid) = *(d_x_array_1_real + uid) * *(d_conj_a_b1_real + uid) - *(d_x_array_1_imag + uid) * *(d_conj_a_b1_imag + uid);
		*(d_x_array_a_b1_imag + uid) = *(d_x_array_1_real + uid) * *(d_conj_a_b1_imag + uid) + *(d_x_array_1_imag + uid) * *(d_conj_a_b1_real + uid);
		*(d_abs_a_b1_real + uid) = pow(*(d_a_b1_real + uid), e) + pow(*(d_a_b1_imag + uid), e);
	// }
}

void H_mulsqre(double* x_array_a_b1_real, double* x_array_a_b1_imag, double* abs_a_b1_real, double* x_array_1_real, double* x_array_1_imag, double* conj_a_b1_real, double* conj_a_b1_imag, double* a_b1_real, double* a_b1_imag, int  nx, int ny, int nc)
{
	double e = 2.0;
	for (int i = 0; i < nx * ny * nc; i++)
	{
		*(x_array_a_b1_real + i) = *(x_array_1_real + i) * *(conj_a_b1_real + i) - *(x_array_1_imag + i) * *(conj_a_b1_imag + i);
		*(x_array_a_b1_imag + i) = *(x_array_1_real + i) * *(conj_a_b1_imag + i) + *(x_array_1_imag + i) * *(conj_a_b1_real + i);
		*(abs_a_b1_real + i) = pow(*(a_b1_real + i), e) + pow(*(a_b1_imag + i), e);
	}
}























































































































































































































































/*
/ Add CUDA and C++ Libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//for __syncthreads()
#ifndef __cudacc__ 
#define __cudacc__
#endif
#include <stdio.h>
#include <tchar.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <ctime>
using namespace std;
// Host function:
void H_conjugate(double*, double*, double*, double*, double*, double*, double*, double*, int, int, int, int);
// CUDA kernel
__global__ void D_conjugate(double*, double*, double*, double*, double*, double*, double*, double*, int, int, int, int, double*, double*, double*, double*, double*);
void main()
{
	int nx = 128;
	int ny = 128;
	int nt = 40;
	int nc = 12;
	// Allocate memory for input data 
	double* x_array_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* x_array_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* conj_a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* conj_a_b1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* a_b1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	// Allocate memory for output data
	double* res_2_real = (double*)malloc(nx * ny * nc * nt * sizeof(double));
	double* res_2_imag = (double*)malloc(nx * ny * nc * nt * sizeof(double));
	double* res_2_real_gpu = (double*)malloc(nx * ny * nc * nt * sizeof(double));
	double* res_2_imag_gpu = (double*)malloc(nx * ny * nc * nt * sizeof(double));
	// Reading files from directory
	cout << "\t\t Opening File to Read .......................................................\n";
	ifstream x_read_real("x_array_real.txt");
	ifstream x_read_imag("x_array_imag.txt");
	ifstream c_read_real("conj_a_b1_real.txt");
	ifstream c_read_imag("conj_a_b1_imag.txt");
	ifstream a_read_real("a_b1_real.txt");
	ifstream a_read_imag("a_b1_imag.txt");
	// Check if all files available in directory
	if (!x_read_real || !x_read_imag || !c_read_real || !c_read_imag || !a_read_imag || !a_read_real)
	{
		cout << "Cannot open text files \n";
		(void)getchar();
		return;
	}
	else
		cout << "All files found:- \t Reading Data In Progress \n";
	//Files reading started 
	cout << "\t\t Files Reading Started .........................................................\n";
	double file_in = 0.0;
	cout << "\t\t Reading Input Data .........................................................\n";
	// x real and Cmplex 
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		x_read_real >> file_in;
		*(x_array_real + i) = file_in;
		x_read_imag >> *(x_array_imag + i);
	}
	double sum_x = 0;
	double e = 2.0;
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		sum_x = sum_x + pow(*(x_array_real + i), e) + pow(*(x_array_imag + i), e);
	}
	double norm_x = sqrt(sum_x);
	cout << "\t\t norm_x of input x_array_real and x_array_imag is = " << norm_x << endl;
	// c_real and complex 
	for (int i = 0; i < (nx * ny * nc); i++)
	{
		c_read_real >> file_in;
		*(conj_a_b1_real + i) = file_in;
		c_read_imag >> *(conj_a_b1_imag + i);
	}
	double sum_c = 0;
	for (int i = 0; i < (nx * ny * nc); i++)
	{
		sum_c = sum_c + pow(*(conj_a_b1_real + i), e) + pow(*(conj_a_b1_imag + i), e);
	}
	double norm_c = sqrt(sum_c);
	cout << "\t\t norm_c of input conj_real and conj_imag is = " << norm_c << endl;
	// a_real and complex 
	for (int i = 0; i < (nx * ny * nc); i++)
	{
		a_read_real >> file_in;
		*(a_b1_real + i) = file_in;
		a_read_imag >> *(a_b1_imag + i);
	}
	double sum_a = 0;
	for (int i = 0; i < (nx * ny * nc); i++)
	{
		sum_a = sum_a + pow(*(a_b1_real + i), e) + pow(*(a_b1_imag + i), e);
	}
	double norm_a = sqrt(sum_a);
	cout << "\t\t norm_a of input a_b1_real and a_b1_imag is = " << norm_a << endl;
	// CPU (Host) computation time analysis 
	int start_c = clock();
	H_conjugate(res_2_real, res_2_imag, x_array_real, x_array_imag, conj_a_b1_real, conj_a_b1_imag, a_b1_real, a_b1_imag, nx, ny, nt, nc);
	int stop_c = clock();
	cout << "\n CPU_computation_Time:" << (stop_c - start_c) << endl;
	// CUDA kernel implementation
	// create a stop watch to record time for CUDA Kernel;
	cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	// cudaEventRecord(start_g);
	// Define Pointers for GPU or Device Memory
	double* d_res_2_real, * d_res_2_imag, * d_x_array_real, * d_x_array_imag;
	double* d_conj_a_b1_real, * d_conj_a_b1_imag, * d_a_b1_real, * d_a_b1_imag;
	// Pass pointer in GPU memory
	double* d_x_array_1_real, * d_x_array_1_imag, * d_x_array_a_b1_real;
	double* d_x_array_a_b1_imag, * d_abs_a_b1_real, * d_abs_a_b1_imag;
	// Allocate Device Memory 
	cudaMalloc((void**)&d_res_2_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_res_2_imag, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_x_array_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_x_array_imag, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_conj_a_b1_real, sizeof(double) * nx * ny * nc);
	cudaMalloc((void**)&d_conj_a_b1_imag, sizeof(double) * nx * ny * nc);
	cudaMalloc((void**)&d_a_b1_real, sizeof(double) * nx * ny * nc);
	cudaMalloc((void**)&d_a_b1_imag, sizeof(double) * nx * ny * nc);
	// Allocate kernel processing pointers:
	cudaMalloc((void**)&d_x_array_1_real, sizeof(double) * nx * ny * nc);
	cudaMalloc((void**)&d_x_array_1_imag, sizeof(double) * nx * ny * nc);
	cudaMalloc((void**)&d_x_array_a_b1_real, sizeof(double) * nx * ny * nc);
	cudaMalloc((void**)&d_x_array_a_b1_imag, sizeof(double) * nx * ny * nc);
	cudaMalloc((void**)&d_abs_a_b1_real, sizeof(double) * nx * ny * nc);
	cudaMalloc((void**)&d_abs_a_b1_imag, sizeof(double) * nx * ny * nc);
	// Copy Data from the host to the device pointer
	cudaMemcpy(d_x_array_real, x_array_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_array_imag, x_array_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_conj_a_b1_real, conj_a_b1_real, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_conj_a_b1_imag, conj_a_b1_imag, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a_b1_real, a_b1_real, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a_b1_imag, a_b1_imag, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
	// Create threads and Blocks for CUDA kernel
	dim3 threads(1, 32);
	dim3 blocks((int)ceil(1 / 1), (int)ceil(nt / 32));
	// Launch Kernel
	cudaEventRecord(start_g);
	D_conjugate << <blocks, threads >> > (d_res_2_real, d_res_2_imag, d_x_array_real, d_x_array_imag, d_conj_a_b1_real, d_conj_a_b1_imag, d_a_b1_real,
		d_a_b1_imag, nx, ny, nt, nc, d_x_array_1_real, d_x_array_1_imag, d_x_array_a_b1_real, d_x_array_a_b1_imag, d_abs_a_b1_real);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_g);
	// Copy Result from Device to Host
	cudaMemcpy(res_2_real_gpu, d_res_2_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(res_2_imag_gpu, d_res_2_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
	// call a CUDA kernel(or launch a kernel);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		// Possibly: exit(-1) if program cannot continue....
	}
	// sync
	// cudaEventRecord(stop_g);
	cudaEventSynchronize(stop_g);
	// Print Time
	float t = 0;
	cudaEventElapsedTime(&t, start_g, stop_g);
	printf("\n GPU_ IFFT_D1_First_Time: %f", t);
	// Free GPU memory
	cudaFree(d_res_2_real);
	cudaFree(d_res_2_imag);
	cudaFree(d_x_array_real);
	cudaFree(d_x_array_imag);
	cudaFree(d_conj_a_b1_real);
	cudaFree(d_conj_a_b1_imag);
	cudaFree(d_a_b1_real);
	cudaFree(d_a_b1_imag);
	cudaFree(d_x_array_1_real);
	cudaFree(d_x_array_1_imag);
	cudaFree(d_x_array_a_b1_real);
	cudaFree(d_x_array_a_b1_imag);
	cudaFree(d_abs_a_b1_real);
	// save output data into text files 
	ofstream my_real("out_real.txt");
	ofstream my_imag("out_imag.txt");
	for (int i = 0; i < nx * ny * nt * nc; i++)
	{
		double out_r = 0.0;
		double out_i = 0.0;
		out_r = *(res_2_real_gpu + i);
		my_real << out_r << endl;;
		out_i = *(res_2_imag_gpu + i);
		my_imag << out_i << endl;
	}
	// close files after saving data
	my_real.close();
	my_imag.close();
}




// Host function
void H_conjugate(double* res_2_real, double* res_2_imag, double* x_array_real, double* x_array_imag, double* conj_a_b1_real, double* conj_a_b1_imag, double* a_b1_real, double* a_b1_imag, int nx, int ny, int nt, int nc)
{
	// Allocate Host memory 
	double* x_array_1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_a_b1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* abs_a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double e = 2.0;

	for (int tt = 0; tt < nt; tt++)
	{
		for (int i = 0; i < nc; i++)
		{
			for (int j = 0; j < nx * ny; j++)
			{
				*(x_array_1_real + nx * ny * i + j) = *(x_array_real + nx * ny * tt + nx * ny * nt * i + j);
				*(x_array_1_imag + nx * ny * i + j) = *(x_array_imag + nx * ny * tt + nx * ny * nt * i + j);
			}
		}
		//		cout << "x_array_1 at Line 1005 is:";
		//		display_3d_matrix_mem_complex (x_array_1_real, x_array_1_imag, nx, ny, nc);
		for (int i = 0; i < nx * ny * nc; i++)
		{
			*(x_array_a_b1_real + i) = *(x_array_1_real + i) * *(conj_a_b1_real + i) - *(x_array_1_imag + i) * *(conj_a_b1_imag + i);
			*(x_array_a_b1_imag + i) = *(x_array_1_real + i) * *(conj_a_b1_imag + i) + *(x_array_1_imag + i) * *(conj_a_b1_real + i);
			*(abs_a_b1_real + i) = pow(*(a_b1_real + i), e) + pow(*(a_b1_imag + i), e);
		}
		//		cout << "x_array_a_b1 at Line 1016 is:";
		//		display_3d_matrix_mem_complex (x_array_a_b1_real, x_array_a_b1_imag, nx, ny, nc);
		//		cout << "abs_a_b1 at Line 1016 is:";
		//		display_3d_matrix_mem_complex (abs_a_b1_real, abs_a_b1_real, nx, ny, nc);
		// Line 982
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				double sumx_real = 0, sumx_imag = 0;
				double sumb1 = 0;
				for (int k = 0; k < nc; k++)
				{
					sumx_real = sumx_real + *(x_array_a_b1_real + nx * ny * k + nx * i + j);
					sumx_imag = sumx_imag + *(x_array_a_b1_imag + nx * ny * k + nx * i + j);
					sumb1 = sumb1 + *(abs_a_b1_real + nx * ny * k + nx * i + j);
				}
				*(res_2_real + nx * ny * tt + ny * i + j) = sumx_real / sumb1;
				*(res_2_imag + nx * ny * tt + ny * i + j) = sumx_imag / sumb1;
			}
		}
		//		cout << "res_2 at Line 1027 is:";
		//		display_3d_matrix_mem_complex (res_2_real, res_2_imag, nx, ny, nt);

	}
}

// Cuda Kernel 

__global__ void D_conjugate(double* d_res_2_real, double* d_res_2_imag, double* d_x_array_real, double* d_x_array_imag, double* d_conj_a_b1_real, double* d_conj_a_b1_imag, double* d_a_b1_real, double* d_a_b1_imag, int nx, int ny, int nt, int nc, double* d_x_array_1_real, double* d_x_array_1_imag, double* d_x_array_a_b1_real, double* d_x_array_a_b1_imag, double* d_abs_a_b1_real)
{
	int tt = blockIdx.x * blockDim.x + threadIdx.x;
	double e = 2.0;

	//for (int tt = 0; tt < nt; tt++)
	//{
	for (int i = 0; i < nc; i++)
	{
		for (int j = 0; j < nx * ny; j++)
		{
			*(d_x_array_1_real + nx * ny * i + j) = *(d_x_array_real + nx * ny * tt + nx * ny * nt * i + j);
			*(d_x_array_1_imag + nx * ny * i + j) = *(d_x_array_imag + nx * ny * tt + nx * ny * nt * i + j);
		}
	}
	//		cout << "x_array_1 at Line 1005 is:";
	//		display_3d_matrix_mem_complex (x_array_1_real, x_array_1_imag, nx, ny, nc);
	for (int i = 0; i < nx * ny * nc; i++)
	{
		*(d_x_array_a_b1_real + i) = *(d_x_array_1_real + i) * *(d_conj_a_b1_real + i) - *(d_x_array_1_imag + i) * *(d_conj_a_b1_imag + i);
		*(d_x_array_a_b1_imag + i) = *(d_x_array_1_real + i) * *(d_conj_a_b1_imag + i) + *(d_x_array_1_imag + i) * *(d_conj_a_b1_real + i);
		*(d_abs_a_b1_real + i) = pow(*(d_a_b1_real + i), e) + pow(*(d_a_b1_imag + i), e);
	}
	//		cout << "x_array_a_b1 at Line 1016 is:";
	//		display_3d_matrix_mem_complex (x_array_a_b1_real, x_array_a_b1_imag, nx, ny, nc);
	//		cout << "abs_a_b1 at Line 1016 is:";
	//		display_3d_matrix_mem_complex (abs_a_b1_real, abs_a_b1_real, nx, ny, nc);
	// Line 982
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			double sumx_real = 0, sumx_imag = 0;
			double sumb1 = 0;
			for (int k = 0; k < nc; k++)
			{
				sumx_real = sumx_real + *(d_x_array_a_b1_real + nx * ny * k + nx * i + j);
				sumx_imag = sumx_imag + *(d_x_array_a_b1_imag + nx * ny * k + nx * i + j);
				sumb1 = sumb1 + *(d_abs_a_b1_real + nx * ny * k + nx * i + j);
			}
			*(d_res_2_real + nx * ny * tt + ny * i + j) = sumx_real / sumb1;
			*(d_res_2_imag + nx * ny * tt + ny * i + j) = sumx_imag / sumb1;
		}
	}
	//		cout << "res_2 at Line 1027 is:";
	//		display_3d_matrix_mem_complex (res_2_real, res_2_imag, nx, ny, nt);

//}
}

*/



















































































































// Cuda Kernel 
/*
__global__ void D_conjugate(double* d_res_2_real, double* d_res_2_imag, double* d_x_array_real, double* d_x_array_imag, double* d_conj_a_b1_real, double* d_conj_a_b1_imag, double* d_a_b1_real, double* d_a_b1_imag, int nx, int ny, int nt, int nc, double* d_x_array_1_real, double* d_x_array_1_imag, double* d_x_array_a_b1_real, double* d_x_array_a_b1_imag, double* d_abs_a_b1_real)
{
	int tt = blockIdx.y * blockDim.y + threadIdx.y;
	double e = 2.0;

	//for (int tt = 0; tt < nt; tt++)
	//{
		for (int i = 0; i < nc; i++)
		{
			for (int j = 0; j < nx * ny; j++)
			{
				*(d_x_array_1_real + nx * ny * i + j) = *(d_x_array_real + nx * ny * tt + nx * ny * nt * i + j);
				*(d_x_array_1_imag + nx * ny * i + j) = *(d_x_array_imag + nx * ny * tt + nx * ny * nt * i + j);
			}
		}
		//		cout << "x_array_1 at Line 1005 is:";
		//		display_3d_matrix_mem_complex (x_array_1_real, x_array_1_imag, nx, ny, nc);
		for (int i = 0; i < nx * ny * nc; i++)
		{
			*(d_x_array_a_b1_real + i) = *(d_x_array_1_real + i) * *(d_conj_a_b1_real + i) - *(d_x_array_1_imag + i) * *(d_conj_a_b1_imag + i);
			*(d_x_array_a_b1_imag + i) = *(d_x_array_1_real + i) * *(d_conj_a_b1_imag + i) + *(d_x_array_1_imag + i) * *(d_conj_a_b1_real + i);
			*(d_abs_a_b1_real + i) = pow(*(d_a_b1_real + i), e) + pow(*(d_a_b1_imag + i), e);
		}
		//		cout << "x_array_a_b1 at Line 1016 is:";
		//		display_3d_matrix_mem_complex (x_array_a_b1_real, x_array_a_b1_imag, nx, ny, nc);
		//		cout << "abs_a_b1 at Line 1016 is:";
		//		display_3d_matrix_mem_complex (abs_a_b1_real, abs_a_b1_real, nx, ny, nc);
		// Line 982
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				double sumx_real = 0, sumx_imag = 0;
				double sumb1 = 0;
				for (int k = 0; k < nc; k++)
				{
					sumx_real = sumx_real + *(d_x_array_a_b1_real + nx * ny * k + nx * i + j);
					sumx_imag = sumx_imag + *(d_x_array_a_b1_imag + nx * ny * k + nx * i + j);
					sumb1 = sumb1 + *(d_abs_a_b1_real + nx * ny * k + nx * i + j);
				}
				*(d_res_2_real + nx * ny * tt + ny * i + j) = sumx_real / sumb1;
				*(d_res_2_imag + nx * ny * tt + ny * i + j) = sumx_imag / sumb1;
			}
		}
		//		cout << "res_2 at Line 1027 is:";
		//		display_3d_matrix_mem_complex (res_2_real, res_2_imag, nx, ny, nt);

	//}
}
*/
/*
// Host function
void H_conjugate(double* res_2_real, double* res_2_imag, double* x_array_real, double* x_array_imag, double* conj_a_b1_real, double* conj_a_b1_imag, double* a_b1_real, double* a_b1_imag, int nx, int ny, int nt, int nc)
{
	// Allocate Host memory 
	double* x_array_1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_a_b1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* abs_a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double e = 2.0;

	for (int tt = 0; tt < nt; tt++)
	{
		for (int i = 0; i < nc; i++)
		{
			for (int j = 0; j < nx * ny; j++)
			{
				*(x_array_1_real + nx * ny * i + j) = *(x_array_real + nx * ny * tt + nx * ny * nt * i + j);
				*(x_array_1_imag + nx * ny * i + j) = *(x_array_imag + nx * ny * tt + nx * ny * nt * i + j);
			}
		}
		//		cout << "x_array_1 at Line 1005 is:";
		//		display_3d_matrix_mem_complex (x_array_1_real, x_array_1_imag, nx, ny, nc);
		for (int i = 0; i < nx * ny * nc; i++)
		{
			*(x_array_a_b1_real + i) = *(x_array_1_real + i) * *(conj_a_b1_real + i) - *(x_array_1_imag + i) * *(conj_a_b1_imag + i);
			*(x_array_a_b1_imag + i) = *(x_array_1_real + i) * *(conj_a_b1_imag + i) + *(x_array_1_imag + i) * *(conj_a_b1_real + i);
			*(abs_a_b1_real + i) = pow(*(a_b1_real + i), e) + pow(*(a_b1_imag + i), e);
		}
		//		cout << "x_array_a_b1 at Line 1016 is:";
		//		display_3d_matrix_mem_complex (x_array_a_b1_real, x_array_a_b1_imag, nx, ny, nc);
		//		cout << "abs_a_b1 at Line 1016 is:";
		//		display_3d_matrix_mem_complex (abs_a_b1_real, abs_a_b1_real, nx, ny, nc);
		// Line 982
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				double sumx_real = 0, sumx_imag = 0;
				double sumb1 = 0;
				for (int k = 0; k < nc; k++)
				{
					sumx_real = sumx_real + *(x_array_a_b1_real + nx * ny * k + nx * i + j);
					sumx_imag = sumx_imag + *(x_array_a_b1_imag + nx * ny * k + nx * i + j);
					sumb1 = sumb1 + *(abs_a_b1_real + nx * ny * k + nx * i + j);
				}
				*(res_2_real + nx * ny * tt + ny * i + j) = sumx_real / sumb1;
				*(res_2_imag + nx * ny * tt + ny * i + j) = sumx_imag / sumb1;
			}
		}
		//		cout << "res_2 at Line 1027 is:";
		//		display_3d_matrix_mem_complex (res_2_real, res_2_imag, nx, ny, nt);

	}
}
*/
/*
// CPU (Host) computation time analysis 
int start_c = clock();
H_conjugate(res_2_real, res_2_imag, x_array_real, x_array_imag, conj_a_b1_real, conj_a_b1_imag, a_b1_real, a_b1_imag, nx, ny, nt, nc);
int stop_c = clock();
cout << "\n CPU_computation_Time:" << (stop_c - start_c) << endl;
// CUDA kernel implementation
// create a stop watch to record time for CUDA Kernel;
cudaEvent_t start_g, stop_g;
cudaEventCreate(&start_g);
cudaEventCreate(&stop_g);
cudaEventRecord(start_g);
// Define Pointers for GPU or Device Memory
double* d_res_2_real, * d_res_2_imag, * d_x_array_real, * d_x_array_imag;
double* d_conj_a_b1_real, * d_conj_a_b1_imag, * d_a_b1_real, * d_a_b1_imag;
// Pass pointer in GPU memory
double* d_x_array_1_real, * d_x_array_1_imag, * d_x_array_a_b1_real;
double* d_x_array_a_b1_imag, * d_abs_a_b1_real, * d_abs_a_b1_imag;
// Allocate Device Memory 
cudaMalloc((void**)&d_res_2_real, sizeof(double)* nx* ny* nt* nc);
cudaMalloc((void**)&d_res_2_imag, sizeof(double)* nx* ny* nt* nc);
cudaMalloc((void**)&d_x_array_real, sizeof(double)* nx* ny* nt* nc);
cudaMalloc((void**)&d_x_array_imag, sizeof(double)* nx* ny* nt* nc);
cudaMalloc((void**)&d_conj_a_b1_real, sizeof(double)* nx* ny* nc);
cudaMalloc((void**)&d_conj_a_b1_imag, sizeof(double)* nx* ny* nc);
cudaMalloc((void**)&d_a_b1_real, sizeof(double)* nx* ny* nc);
cudaMalloc((void**)&d_a_b1_imag, sizeof(double)* nx* ny* nc);
// Allocate kernel processing pointers:
cudaMalloc((void**)&d_x_array_1_real, sizeof(double)* nx* ny* nc);
cudaMalloc((void**)&d_x_array_1_imag, sizeof(double)* nx* ny* nc);
cudaMalloc((void**)&d_x_array_a_b1_real, sizeof(double)* nx* ny* nc);
cudaMalloc((void**)&d_x_array_a_b1_imag, sizeof(double)* nx* ny* nc);
cudaMalloc((void**)&d_abs_a_b1_real, sizeof(double)* nx* ny* nc);
cudaMalloc((void**)&d_abs_a_b1_imag, sizeof(double)* nx* ny* nc);
// Copy Data from the host to the device pointer
cudaMemcpy(d_x_array_real, x_array_real, sizeof(double)* nx* ny* nt* nc, cudaMemcpyHostToDevice);
cudaMemcpy(d_x_array_imag, x_array_imag, sizeof(double)* nx* ny* nt* nc, cudaMemcpyHostToDevice);
cudaMemcpy(d_conj_a_b1_real, conj_a_b1_real, sizeof(double)* nx* ny* nc, cudaMemcpyHostToDevice);
cudaMemcpy(d_conj_a_b1_imag, conj_a_b1_imag, sizeof(double)* nx* ny* nc, cudaMemcpyHostToDevice);
cudaMemcpy(d_a_b1_real, a_b1_real, sizeof(double)* nx* ny* nc, cudaMemcpyHostToDevice);
cudaMemcpy(d_a_b1_imag, a_b1_imag, sizeof(double)* nx* ny* nc, cudaMemcpyHostToDevice);
// Create threads and Blocks for CUDA kernel
dim3 threads(1, 1);
dim3 blocks((int)ceil(1 / 1), (int)ceil(nt / 1));
// Launch Kernel
// cudaEventRecord(start_g);
D_conjugate << <blocks, threads >> > (d_res_2_real, d_res_2_imag, d_x_array_real, d_x_array_imag, d_conj_a_b1_real, d_conj_a_b1_imag, d_a_b1_real,
	d_a_b1_imag, nx, ny, nt, nc, d_x_array_1_real, d_x_array_1_imag, d_x_array_a_b1_real, d_x_array_a_b1_imag, d_abs_a_b1_real);
cudaDeviceSynchronize();
// cudaEventRecord(stop_g);
// Copy Result from Device to Host
cudaMemcpy(res_2_real_gpu, d_res_2_real, sizeof(double)* nx* ny* nt* nc, cudaMemcpyDeviceToHost);
cudaMemcpy(res_2_imag_gpu, d_res_2_imag, sizeof(double)* nx* ny* nt* nc, cudaMemcpyDeviceToHost);
// call a CUDA kernel(or launch a kernel);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess)
{
	printf("CUDA Error: %s\n", cudaGetErrorString(err));
	// Possibly: exit(-1) if program cannot continue....
}
// sync
cudaEventRecord(stop_g);
cudaEventSynchronize(stop_g);
// Print Time
float t = 0;
cudaEventElapsedTime(&t, start_g, stop_g);
printf("\n GPU_ IFFT_D1_First_Time: %f", t);
// Free GPU memory
cudaFree(d_res_2_real);
cudaFree(d_res_2_imag);
cudaFree(d_x_array_real);
cudaFree(d_x_array_imag);
cudaFree(d_conj_a_b1_real);
cudaFree(d_conj_a_b1_imag);
cudaFree(d_a_b1_real);
cudaFree(d_a_b1_imag);
cudaFree(d_x_array_1_real);
cudaFree(d_x_array_1_imag);
cudaFree(d_x_array_a_b1_real);
cudaFree(d_x_array_a_b1_imag);
cudaFree(d_abs_a_b1_real);
// save output data into text files 
ofstream my_real("out_real.txt");
ofstream my_imag("out_imag.txt");
for (int i = 0; i < nx * ny * nt * nc; i++)
{
	double out_r = 0.0;
	double out_i = 0.0;
	out_r = *(res_2_real_gpu + i);
	my_real << out_r << endl;;
	out_i = *(res_2_imag_gpu + i);
	my_imag << out_i << endl;
}
// close files after saving data
my_real.close();
my_imag.close();
}
*/
