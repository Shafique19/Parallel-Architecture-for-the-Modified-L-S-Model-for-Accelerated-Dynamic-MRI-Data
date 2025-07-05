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
#include <math.h>
#include <cmath>
#include <ctime>
using namespace std;
// CUDA Kernel and C main function
void H_applymask(double*, double*, double*, double*, double*, int, int, int, int);
__global__ void D_applymask(double*, double*, double*, double*, double*, int, int, int, int, double*, double*);
// Main Function start here 
void main()
{
	int nx = 256;
	int ny = 256;
	int nt = 11;
	int nc = 30;
	double* M_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* M_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* Mask_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* Mask_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y_real_cpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag_cpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_real_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	// Reading files from directory
	cout << "\t\t Opening File to Read .......................................................\n";
	ifstream M_read_real("M_real.txt");
	ifstream M_read_imag("M_imag.txt");
	ifstream Mask_read_real("Mask_real.txt");
	ifstream Mask_read_imag("Mask_imag.txt");
	// Check if all files available in directory
	if (!M_read_real || !M_read_imag || !Mask_read_real || !Mask_read_imag)
	{
		cout << "Cannot open text files \n";
		(void)getchar();
		return;
	}
	else
		cout << "All files found:- \t Reading Data In Progress \n";
	//Files reading started 
	cout << "\t\t Files Reading Started ........................................................\n";
	double file_in = 0.0;
	cout << "\t\t Reading Input Data ...........................................................\n";
	// M real and Cmplex 
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		M_read_real >> file_in;
		*(M_real + i) = file_in;
		M_read_imag >> *(M_imag + i);
	}
	double sum_m = 0;
	double e = 2.0;
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		sum_m = sum_m + pow(*(M_real + i), e) + pow(*(M_imag + i), e);
	}
	double norm_m = sqrt(sum_m);
	cout << "\t\t norm_M of input M_ral and M_imag is = " << norm_m << endl;

	// Mask real and Cmplex 
	for (int i = 0; i < (nx * ny * nt); i++)
	{
		Mask_read_real >> file_in;
		*(Mask_real + i) = file_in;
		Mask_read_imag >> *(Mask_imag + i);
	}
	double sum_mask = 0;
	for (int i = 0; i < (nx * ny * nt); i++)
	{
		sum_mask = sum_mask + pow(*(Mask_real + i), e) + pow(*(Mask_imag + i), e);
	}
	double norm_mask = sqrt(sum_mask);
	cout << "\t\t norm_Mask of input Mask_ral and Mask_imag is = " << norm_mask << endl;
	// CPU Task
	int start_c = clock();
	H_applymask(y_real_cpu, y_imag_cpu, M_real, M_imag, Mask_real, nx, ny, nt, nc);
	int stop_c = clock();
	cout << "\n CPU_Apply_mask_Time:" << (stop_c - start_c) << endl;
	// CUDA Kernel
	// create a stop watch to record time for CUDA Kernel;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Define Pointers for GPU or Device Memory
	cudaEventRecord(start);
	double* d_M_real, * d_M_imag, * d_Mask_real, * d_y_imag, * d_y_real;
	double* d_res_x_real, * d_res_x_imag;
	// Allocate Device Memory 
	cudaMalloc((void**)&d_y_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_y_imag, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_M_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_M_imag, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_Mask_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_res_x_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_res_x_imag, sizeof(double) * nx * ny * nt * nc);
	// Copy Data from the host to the device pointer
	cudaMemcpy(d_M_real, M_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_M_imag, M_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Mask_real, Mask_real, sizeof(double) * nx * ny * nt, cudaMemcpyHostToDevice);
	// cudaEventRecord(start);
	dim3 nthreads(512, 1); // change
	dim3 nblocks((int)ceil(nx*ny*nt/512), (int)ceil(nc/1));
	D_applymask << < nblocks, nthreads >> > (d_y_real, d_y_imag, d_M_real, d_M_imag, d_Mask_real, nx, ny, nt, nc, d_res_x_real, d_res_x_imag);
	cudaDeviceSynchronize();
	// cudaEventRecord(stop);
	// Copy Result from Device to Host
	cudaMemcpy(y_real_gpu, d_y_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_imag_gpu, d_y_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
	// call a CUDA kernel(or launch a kernel);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		// Possibly: exit(-1) if program cannot continue....
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	// Print Time
	float k = 0;
	cudaEventElapsedTime(&k, start, stop);
	printf("\n GPU_Apply_mask_Time: %f", k);
	// Free GPU memory
	cudaFree(d_M_real);
	cudaFree(d_M_imag);
	cudaFree(d_Mask_real);
	cudaFree(d_y_real);
	cudaFree(d_y_imag);
	cudaFree(d_res_x_real);
	cudaFree(d_res_x_imag);
	// save output data into text files 
	ofstream my_real("out_real.txt");
	ofstream my_imag("out_imag.txt");
	for (int i = 0; i < nx * ny * nt * nc; i++)
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
//
// CUDA Kernel For Applying Sampling Mask :
__global__ void D_applymask(double* res_y_real, double* res_y_imag, double* res_real, double* res_imag, double* a_mask_real, int nx, int ny, int nt, int nc, double*d_res_x_real, double*d_res_x_imag)
{
	// double* res_x_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	// double* res_x_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y * blockDim.y + threadIdx.y;

	//for (int ch = 0; ch < nc; ch++)
	//{
	//for (int i = 0; i < nx * ny * nt; i++)
	//{
		*(d_res_x_real + id) = *(res_real + ch * nx * ny * nt + id);
		*(d_res_x_imag + id) = *(res_imag + ch * nx * ny * nt + id);
	//}
	//for (int i = 0; i < nx * ny * nt; i++)
	//{
		*(res_y_real + ch * nx * ny * nt + id) = *(d_res_x_real + id) * *(a_mask_real + id);
		*(res_y_imag + ch * nx * ny * nt + id) = *(d_res_x_imag + id) * *(a_mask_real + id);
	//}

	//}
}
// // // // 
// Applying Sampling Mask :
void H_applymask(double* res_y_real, double* res_y_imag, double* res_real, double* res_imag, double* a_mask_real, int nx, int ny, int nt, int nc)
{
	double* res_x_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* res_x_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	for (int ch = 0; ch < nc; ch++)
	{
		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(res_x_real + i) = *(res_real + ch * nx * ny * nt + i);
			*(res_x_imag + i) = *(res_imag + ch * nx * ny * nt + i);
		}
		//		cout << "res_x is:";
		//		display_3d_matrix_mem_complex (res_x_real, res_x_imag, nx, ny, nt);
		//		cout << "a_mask is:";
		//		display_3d_matrix_mem_complex (a_mask_real, a_mask_imag, nx, ny, nt);

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(res_y_real + nx * ny * nt * ch + i) = *(res_x_real + i) * *(a_mask_real + i);
			*(res_y_imag + nx * ny * nt * ch + i) = *(res_x_imag + i) * *(a_mask_real + i);
		}

	}
}
// The end