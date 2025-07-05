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
#include <stdlib.h>
using namespace std;
// functions to display
void display_matrix_mem_real(double*, int, int);
// CUDA Kernel and C main function
void H4_multiplication(double*, double*, double*, double*, double*, double*, int, int, int, int);
__global__ void D4_multiplication(double*, double*, double*, double*, double*, double*, int, int, int, int, double*, double*, double*, double*);
// Main Function start here 
void main()
{
	int nx = 256;
	int ny = 256;
	int nt = 11;
	int nc = 30;
	double* lps_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* lps_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* b_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* b_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* y_real_cpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag_cpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_real_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	// Reading files from directory
	cout << "\t\t Opening File to Read .......................................................\n";
	ifstream lps_read_real("lps_real.txt");
	ifstream lps_read_imag("lps_imag.txt");
	ifstream b_read_real("b_real.txt");
	ifstream b_read_imag("b_imag.txt");
	// Check if all files available in directory
	if (!lps_read_real || !lps_read_imag || !b_read_real || !b_read_imag)
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
	// lps_real and lps_imag Norm
	for (int i = 0; i < (nx * ny * nt); i++)
	{
		lps_read_real >> file_in;
		*(lps_real + i) = file_in;
		lps_read_imag >> *(lps_imag + i);
	}
	double sum_lps = 0;
	double e = 2.0;
	for (int i = 0; i < (nx * ny * nt); i++)
	{
		sum_lps = sum_lps + pow(*(lps_real + i), e) + pow(*(lps_imag + i), e);
	}
	double norm_lps = sqrt(sum_lps);
	cout << "\t\t norm_lps of input lps_ral and lps_imag is = " << norm_lps << endl;
	// b_real and b_imag Norm
	for (int i = 0; i < (nx * ny * nc); i++)
	{
		b_read_real >> file_in;
		*(b_real + i) = file_in;
		b_read_imag >> *(b_imag + i);
	}
	double sum_b = 0;
	for (int i = 0; i < (nx * ny * nc); i++)
	{
		sum_b = sum_b + pow(*(b_real + i), e) + pow(*(b_imag + i), e);
	}
	double norm_b = sqrt(sum_b);
	cout << "\t\t norm_b of input b_ral and b_imag is = " << norm_b << endl;
	// CPU Task
	int start_c = clock();
	// H4_multiplication(y_real_cpu, y_imag_cpu, lps_real, lps_imag, b_real, b_imag, nx, ny, nt, nc);
	int stop_c = clock();
	cout << "\n CPU_multiplication_Time:" << (stop_c - start_c) << endl;
	// Cuda Kernel 
	// create a stop watch to record time for CUDA Kernel;
	cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	cudaEventRecord(start_g);
	// Define Pointers for GPU or Device Memory
	double* d_y_real, * d_y_imag, * d_lps_real, * d_lps_imag, * d_b_real, * d_b_imag;
	double* d_b_x1_real, * d_b_x1_imag, * d_a_b1_x1_real, * d_a_b1_x1_imag;
	// Allocate Device Memory 
	cudaMalloc((void**)&d_lps_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_lps_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_b_real, sizeof(double) * nx * ny * nc);
	cudaMalloc((void**)&d_b_imag, sizeof(double) * nx * ny * nc);
	cudaMalloc((void**)&d_y_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_y_imag, sizeof(double) * nx * ny * nt * nc);
	// Passing pointers
	cudaMalloc((void**)&d_b_x1_real, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_b_x1_imag, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_a_b1_x1_real, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_a_b1_x1_imag, sizeof(double) * nx * ny);
	// Copy Data from the host to the device pointer
	cudaMemcpy(d_lps_real, lps_real, sizeof(double) * nx * ny * nt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lps_imag, lps_imag, sizeof(double) * nx * ny * nt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_real, b_real, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_imag, b_imag, sizeof(double) * nx * ny * nc, cudaMemcpyHostToDevice);
	// Create threads and Blocks for CUDA kernel
	dim3 block(32, 1);
	dim3 grid((int)ceil(nx*ny/32), (int)ceil(1));
	// Launch Kernel
	// cudaEventRecord(start_g);
	D4_multiplication << <(nx * ny / 32), 32>> > (d_y_real, d_y_imag, d_lps_real, d_lps_imag, d_b_real, d_b_imag, nx, ny, nt, nc, d_b_x1_real, d_b_x1_imag, d_a_b1_x1_real, d_a_b1_x1_imag);
	cudaDeviceSynchronize();
	// cudaEventRecord(stop_g);
	// Copy Result from Device to Host
	cudaMemcpy(y_real_gpu, d_y_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_imag_gpu, d_y_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop_g);
	cudaEventSynchronize(stop_g);
	// Print Time
	float t = 0;
	cudaEventElapsedTime(&t, start_g, stop_g);
	printf("\n GPU_ Multiplication: %f", t);
	// Free GPU memory
	cudaFree(d_y_real);
	cudaFree(d_y_imag);
	cudaFree(d_lps_real);
	cudaFree(d_lps_imag);
	cudaFree(d_b_real);
	cudaFree(d_b_imag);
	// Passing pointers free
	cudaFree(d_b_x1_real);
	cudaFree(d_b_x1_imag);
	cudaFree(d_a_b1_x1_real);
	cudaFree(d_a_b1_x1_imag);
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
// Data Consistency CUDA Kerenel for 4D Multiplication 
__global__ void D4_multiplication(double* y_real, double* y_imag, double* LPS_real, double* LPS_imag, double* a_b1_real, double* a_b1_imag, int nx, int ny, int nt, int nc, double* d_b_x1_real, double* d_b_x1_imag, double* d_a_b1_x1_real, double* d_a_b1_x1_imag)
{
	/*double* b_x1_real = (double*)malloc(nx * ny * sizeof(double));
	double* b_x1_imag = (double*)malloc(nx * ny * sizeof(double));
	double* a_b1_x1_real = (double*)malloc(nx * ny * sizeof(double));
	double* a_b1_x1_imag = (double*)malloc(nx * ny * sizeof(double));
	*/
	// int tt = blockIdx.y * blockDim.y + threadIdx.y;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// if (tt < nt && ch < nc)
	for (int tt = 0; tt < nt; tt++)
	{
		for (int ch = 0; ch < nc; ch++)
		{
	//for (int i = 0; i < nx * ny; i++)
	//{
		*(d_b_x1_real + id) = *(LPS_real + tt * nx * ny + id);
		*(d_b_x1_imag + id) = *(LPS_imag + tt * nx * ny + id);
	//}
	//for (int i = 0; i < nx * ny; i++)
	//{
		*(d_a_b1_x1_real + id) = *(a_b1_real + ch * nx * ny + id);
		*(d_a_b1_x1_imag + id) = *(a_b1_imag + ch * nx * ny + id);
	//}

	//for (int i = 0; i < nx * ny; i++)
	//{
		*(y_real + ch * nx * ny * nt + tt * nx * ny + id) = (*(d_b_x1_real + id) * *(d_a_b1_x1_real + id)) - (*(d_b_x1_imag + id) * *(d_a_b1_x1_imag + id));
		*(y_imag + ch * nx * ny * nt + tt * nx * ny + id) = (*(d_b_x1_real + id) * *(d_a_b1_x1_imag + id)) + (*(d_b_x1_imag + id) * *(d_a_b1_x1_real + id));
	}
	}
//}
}
// Data Consistency 1st Step : LPS*b1 : Multiplication
void H4_multiplication(double* X2_real, double* X2_imag, double* LPS_real, double* LPS_imag, double* a_b1_real, double* a_b1_imag, int nx, int ny, int nt, int nc)
{
	double* b_x1_real = (double*)malloc(nx * ny * sizeof(double));
	double* b_x1_imag = (double*)malloc(nx * ny * sizeof(double));
	double* a_b1_x1_real = (double*)malloc(nx * ny * sizeof(double));
	double* a_b1_x1_imag = (double*)malloc(nx * ny * sizeof(double));

	for (int tt = 0; tt < nt; tt++)
	{
		for (int ch = 0; ch < nc; ch++)
		{
			for (int i = 0; i < nx * ny; i++)
			{
				*(b_x1_real + i) = *(LPS_real + tt * nx * ny + i);
				*(b_x1_imag + i) = *(LPS_imag + tt * nx * ny + i);
			}
			for (int i = 0; i < nx * ny; i++)
			{
				*(a_b1_x1_real + i) = *(a_b1_real + ch * nx * ny + i);
				*(a_b1_x1_imag + i) = *(a_b1_imag + ch * nx * ny + i);
			}

			for (int i = 0; i < nx * ny; i++)
			{
				*(X2_real + ch * nx * ny * nt + tt * nx * ny + i) = (*(b_x1_real + i) * *(a_b1_x1_real + i)) - (*(b_x1_imag + i) * *(a_b1_x1_imag + i));
				*(X2_imag + ch * nx * ny * nt + tt * nx * ny + i) = (*(b_x1_real + i) * *(a_b1_x1_imag + i)) + (*(b_x1_imag + i) * *(a_b1_x1_real + i));
			}
		}
	}
}

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
