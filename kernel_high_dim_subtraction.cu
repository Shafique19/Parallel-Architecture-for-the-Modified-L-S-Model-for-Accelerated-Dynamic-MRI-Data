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
void display_matrix_mem_complex(double*, double*, int, int);
void display_3d_matrix_mem_complex(double*, double*, int, int, int);
void display_4d_matrix_mem_complex(double*, double*, int, int, int, int);
// Host function for subtraction
void H_subtraction(double*, double*, double*, double*, double*, double*, int, int);
// GPU kernel for subtraction
__global__ void D_subtraction(double*, double*, double*, double*, double*, double*, int, int);
// Start main function
void main()
{
	int nx = 256;
	int ny = 256;
	int nt = 11;
	int nc = 30; 
	double* M_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* M_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* S_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* S_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	// reading data from text files 
	cout << "\t\t Opening File to Read .......................................................\n";
	ifstream M_read_real("M_real.txt");
	ifstream M_read_imag("M_imag.txt");
	ifstream S_read_real("S_real.txt");
	ifstream S_read_imag("S_imag.txt");
	//Checking if all the files available in root directory or not
	if (!M_read_real || !M_read_imag || !S_read_real || !S_read_imag)
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
	// M real and imaginary 
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		M_read_real >> file_in;
		*(M_real + i) = file_in;
		M_read_imag >> *(M_imag + i);
	}
	// find norm M
	double sum_m = 0.0;
	double e = 2.0;
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		sum_m = sum_m + pow(*(M_real + i), e) + pow(*(M_imag + i), e);
	}
	double norm_m = sqrt(sum_m);
	cout << "\t\t norm_M of input M_ral and M_imag is = " << norm_m << endl;
	// S real and imaginary
	cout << "\t\t Reading Initial S_real and S_imag........................................................\n";
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		S_read_real >> file_in;
		*(S_real + i) = file_in;
		S_read_imag >> *(S_imag + i);
	}
	// find norm of S
	double sum_s = 0.0;
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		sum_s = sum_s + pow(*(S_real + i), e) + pow(*(S_imag + i), e);
	}
	double norm_s = sqrt(sum_s);
	cout << "\t\t norm_M of input S_real and S_imag is = " << norm_s << endl;

	//Host_subtraction (M-S)_Start
	double* y_real_cpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag_cpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_real_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	int start_c = clock();
	H_subtraction(y_real_cpu, y_imag_cpu, M_real, M_imag, S_real, S_imag, nx * ny * nt, nc);
	int stop_c = clock();
	cout << "\n CPU Time is :" << (stop_c - start_c) << endl;
	printf("\n CPU Output :");
	// display_matrix_mem_real(y_real_cpu, nx*ny, nt);
	//Host_subtraction (M-S)_Done
	//GPU_subtraction_Start
	cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	//cudaEventRecord(start_g);
	double* d_M_real, * d_M_imag, * d_S_real, * d_S_imag, * d_y_real, * d_y_imag;
	// Allocate Device Memory 
	cudaMalloc((void**)&d_M_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_M_imag, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_S_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_S_imag, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_y_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_y_imag, sizeof(double) * nx * ny * nt * nc);
	// Copy Data from the host to the device pointer
	cudaMemcpy(d_M_real, M_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_M_imag, M_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_S_real, S_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_S_imag, S_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	// kernel configuration
	dim3 nthreads(6, 64);
	dim3 nblocks((int)ceil(nc /6), (int)ceil(nx * ny * nt/64));
	// Launch Kernel
	cudaEventRecord(start_g);
	D_subtraction << <nblocks, nthreads >> > (d_y_real, d_y_imag, d_M_real, d_M_imag, d_S_real, d_S_imag, nx * ny * nt, nc);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_g);
	// Copy Result from Device to Host
	cudaMemcpy(y_real_gpu, d_y_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_imag_gpu, d_y_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
	// Device and Thread Synchronization
	// cudaEventRecord(stop_g);
	cudaEventSynchronize(stop_g);
	// Print Time
	float t = 0;
	cudaEventElapsedTime(&t, start_g, stop_g);
	printf("\n GPU time is : %f", t);
	// Print Output
	// printf("\n GPU Output :");
	// display_matrix_mem_real(y_real_GPU, nx*ny, nt);
	// Free GPU memory
	cudaFree(d_M_real);
	cudaFree(d_M_imag);
	cudaFree(d_S_real);
	cudaFree(d_S_imag);
	cudaFree(d_y_real);
	cudaFree(d_y_imag);
	// Save files in text format
	ofstream my_real("out_real.txt");
	ofstream my_imag("out_imag.txt");
	for (int i = 0; i < nx * ny * nt * nc; i++)
	{
		double out_r = 0.0;
		double out_i = 0.0;
		out_r = *(y_real_cpu+i);
		my_real << out_r << endl;;
		out_i = *(y_imag_cpu+i);
		my_imag << out_i << endl;
	}
	// close files after saving data
	my_real.close();
	my_imag.close();
}
// Functions used in main function
// Host Functions Used 
void H_subtraction(double* y_real, double* y_imag, double* A_real, double* A_imag, double* B_real, double* B_imag, int AB_r, int AB_c)
{
	for (int i = 0; i < AB_r; i++)
	{
		for (int j = 0; j < AB_c; j++)
		{
			*(y_real + i * AB_c + j) = *(A_real + i * AB_c + j) - *(B_real + i * AB_c + j);
			*(y_imag + i * AB_c + j) = *(A_imag + i * AB_c + j) - *(B_imag + i * AB_c + j);
		}
	}
}
// CUDA Kernel for subtraction
__global__ void D_subtraction(double* y_real, double* y_imag, double* M_real, double* M_imag, double* S_real, double* S_imag, int xy, int nt)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y; // row major (row id)
	int j = blockIdx.x * blockDim.x + threadIdx.x; // row major (column id)
	int tid = i * nt + j;
	if (i < xy && j < nt)
	{
		//c[tid] = a[tid] + b[tid];
		*(y_real + tid) = *(M_real + tid) - *(S_real + tid);
		*(y_imag + tid) = *(M_imag + tid) - *(S_imag + tid);
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
// Function to Dispaly Complex Matrix
void display_matrix_mem_complex(double* A_real, double* A_imag, int row, int col)
{
	cout << "\n\n";
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << *(A_real + i * row + j) << " + " << *(A_imag + i * row + j) << "\t";
		}
		cout << "\n";
	}
}

//// Function to Dispaly 3D Matrix
void display_3d_matrix_mem_complex(double* A_real, double* A_imag, int row, int col, int d3)
{
	cout << "\n\n";
	for (int i = 0; i < d3; i++)
	{
		for (int j = 0; j < row; j++)
		{
			for (int k = 0; k < col; k++)
			{
				cout << *(A_real + row * col * i + j * col + k) << " + " << *(A_imag + row * col * i + j * col + k) << "\t";
			}
			cout << "\n";
		}
		cout << "\n";
	}
}
// Function to Dispaly 4D Matrix
void display_4d_matrix_mem_complex(double* A_real, double* A_imag, int row, int col, int d3, int d4)
{
	cout << "\n\n";
	for (int i = 0; i < d4; i++)
	{
		for (int j = 0; j < d3; j++)
		{
			for (int k = 0; k < row; k++)
			{
				for (int l = 0; l < col; l++)
				{
					cout << *(A_real + row * col * d3 * i + row * col * j + col * k + l) << " + " << *(A_imag + row * col * d3 * i + row * col * j + col * k + l) << "\t";
				}
				cout << "\n";
			}
			cout << "\n";
		}
		cout << "\n";
	}
}
// code prepared by M.Shafique PhD scholar CUI