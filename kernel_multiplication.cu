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
// Host function for multiplication
void H_multiplication(double*, double*, double*, double*, double*, double*, int, int, int);
// CUDA kernel for multiplication
__global__ void D_multiplication(double*, double*, double*, double*, double*, double*, int, int, int);
__global__ void D_rmultiplication(double*, double*, double*, double*, double*, int, int, int);
void main()
{
	int nx = 256;
	int ny = 256;
	int nt = 11;
	int kn = 11;
	int k = 4;
	double* M_real = (double*)malloc(nx * ny * kn * sizeof(double));
	double* M_imag = (double*)malloc(nx * ny * kn * sizeof(double));
	double* S_real = (double*)malloc(kn * k * sizeof(double));
	double* S_imag = (double*)malloc(kn * k * sizeof(double));
	// double* M_real = (double*)malloc(kn * kn * sizeof(double));
	// double* M_imag = (double*)malloc(kn * kn * sizeof(double));
	// double* RTM_real = (double*)malloc(nt * Kn * sizeof(double));
	// double* S_real = (double*)malloc(nx * ny * kn * sizeof(double));
	// double* S_imag = (double*)malloc(nx * ny * kn * sizeof(double));
	// double* S_real = (double*)malloc(kn * kn * sizeof(double));
	// double* S_imag = (double*)malloc(kn * kn * sizeof(double));
	// double* y_real = (double*)malloc(nx * ny * nt * sizeof(double));
	// double* y_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// double* y_real_cpu = (double*)malloc(kn * kn * sizeof(double));
	// double* y_imag_cpu = (double*)malloc(kn * kn * sizeof(double));
	// double* y_real_gpu = (double*)malloc(kn * kn * sizeof(double));
	// double* y_imag_gpu = (double*)malloc(kn * kn * sizeof(double));
	// double* y_real_cpu = (double*)malloc(nt * kn * sizeof(double));
	// double* y_imag_cpu = (double*)malloc(nt * kn * sizeof(double));
	// double* y_real_gpu = (double*)malloc(nt * kn * sizeof(double));
	// double* y_imag_gpu = (double*)malloc(kn * kn * sizeof(double));
	// double* y_real_cpu = (double*)malloc(nx * ny * kn * sizeof(double));
	// double* y_imag_cpu = (double*)malloc(nx * ny * kn * sizeof(double));
	// double* y_real_gpu = (double*)malloc(nx * ny * kn * sizeof(double));
	// double* y_imag_gpu = (double*)malloc(nx * ny * kn * sizeof(double));
	double* y_real_cpu = (double*)malloc(nx * ny * k * sizeof(double));
	double* y_imag_cpu = (double*)malloc(nx * ny * k * sizeof(double));
	double* y_real_gpu = (double*)malloc(nx * ny * k * sizeof(double));
	double* y_imag_gpu = (double*)malloc(nx * ny * k * sizeof(double));
	// Reading Data from the files
	cout << "\t\t Opening File to Read .......................................................\n";
	ifstream M_read_real("M_real.txt");
	ifstream M_read_imag("M_imag.txt");
	// ifstream RTM_read_real("RTM_real.txt");
	ifstream S_read_real("S_real.txt");
	ifstream S_read_imag("S_imag.txt");
	// Checking if all the files available in root directory or not
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
	// M real and Cmplex 
	for (int i = 0; i < (nx*ny*kn); i++)
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
	// S_real and Cmplex
	cout << "\t\t Reading Initial S_real and S_imag........................................................\n";
	for (int i = 0; i < (kn * k); i++)
	{
		 S_read_real >> file_in;
		*(S_real + i) = file_in;
		 S_read_imag >> *(S_imag + i);
	}
	double sum_s = 0;
	for (int i = 0; i < (kn * k); i++)
	{
		sum_s = sum_s + pow(*(S_real + i), e) + pow(*(S_imag + i), e);
	}
	double norm_s = sqrt(sum_s);
	// cout << "\t\t norm_M of input S_real and S_imag is = " << norm_s << endl;
	cout << "\t\t norm_S of input S_real+imag = " << norm_s << endl;
	// Host multiplication
	int start_c = clock();
	H_multiplication(y_real_cpu, y_imag_cpu, M_real, M_imag, S_real, S_imag, nx*ny, kn, k);
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
	double* d_M_real, * d_M_imag, * d_S_real, * d_S_imag, * d_y_real, * d_y_imag; 
	// Allocate Device Memory 
	cudaMalloc((void**)&d_M_real, sizeof(double) * nx * ny * kn);
	cudaMalloc((void**)&d_M_imag, sizeof(double) * nx * ny * kn);
	cudaMalloc((void**)&d_S_real, sizeof(double) * kn * k);
    cudaMalloc((void**)&d_S_imag, sizeof(double) * kn * k);
	// cudaMalloc((void**)&d_S_real, sizeof(double) * nx * ny * kn);
	// cudaMalloc((void**)&d_S_imag, sizeof(double) * nx * ny * kn);
	// cudaMalloc((void**)&d_M_real, sizeof(double) * kn * kn);
	// cudaMalloc((void**)&d_M_imag, sizeof(double) * kn * kn);
	// cudaMalloc((void**)&d_RTM_real, sizeof(double) * nt * Kn);
	// cudaMalloc((void**)&d_S_real, sizeof(double) * kn * kn);
	// cudaMalloc((void**)&d_S_imag, sizeof(double) * kn * kn);
	// cudaMalloc((void**)&d_S_real, sizeof(double) * nx * ny * kn);
	// cudaMalloc((void**)&d_S_imag, sizeof(double) * nx * ny * kn);
	// cudaMalloc((void**)&d_y_real, sizeof(double) * kn * kn);
	// cudaMalloc((void**)&d_y_imag, sizeof(double) * kn * kn);
	// cudaMalloc((void**)&d_y_real, sizeof(double) * nt * kn);
	// cudaMalloc((void**)&d_y_imag, sizeof(double) * nt * kn);
	// cudaMalloc((void**)&d_y_real, sizeof(double) * nx * ny * kn);
	// cudaMalloc((void**)&d_y_imag, sizeof(double) * nx * ny * kn);
	cudaMalloc((void**)&d_y_real, sizeof(double) * nx * ny * k);
	cudaMalloc((void**)&d_y_imag, sizeof(double) * nx * ny * k);
	// Copy Data from the host to the device pointer
	cudaMemcpy(d_M_real, M_real, sizeof(double) * nx * ny * kn, cudaMemcpyHostToDevice);
	cudaMemcpy(d_M_imag, M_imag, sizeof(double) * nx * ny * kn, cudaMemcpyHostToDevice);
	cudaMemcpy(d_S_real, S_real, sizeof(double) * kn * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S_imag, S_imag, sizeof(double) * kn * k, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_M_real, M_real, sizeof(double) * kn * kn, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_M_imag, M_imag, sizeof(double) * kn * kn, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_RTM_real, RTM_real, sizeof(double) * nt * Kn, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_S_real, S_real, sizeof(double) * nx * ny * kn, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_S_imag, S_imag, sizeof(double) * nx * ny * kn, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_S_real, S_real, sizeof(double) * kn * kn, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_S_imag, S_imag, sizeof(double) * kn * kn, cudaMemcpyHostToDevice);
	// Kernel configuration
	dim3 nthreads(k, 128); // change
	dim3 nblocks((int)ceil(k/k), (int)ceil(nx*ny /128));
	// Launch Kernel
	cudaEventRecord(start_g);
	D_multiplication<<<nblocks, nthreads >> > (d_y_real, d_y_imag, d_M_real, d_M_imag, d_S_real, d_S_imag, nx * ny, kn, k);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_g);
	// Copy Result from Device to Host
	cudaMemcpy(y_real_gpu, d_y_real, sizeof(double) * nx * ny * k, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_imag_gpu, d_y_imag, sizeof(double) * nx * ny * k, cudaMemcpyDeviceToHost);
	// cudaMemcpy(y_real_gpu, d_y_real, sizeof(double) * kn * kn, cudaMemcpyDeviceToHost);
	// cudaMemcpy(y_imag_gpu, d_y_imag, sizeof(double) * nt * kn, cudaMemcpyDeviceToHost);
	// cudaMemcpy(y_real_gpu, d_y_real, sizeof(double) * nt * kn, cudaMemcpyDeviceToHost);
	// cudaMemcpy(y_imag_gpu, d_y_imag, sizeof(double) * kn * kn, cudaMemcpyDeviceToHost);
	// cudaMemcpy(y_real_gpu, d_y_real, sizeof(double) * nx * ny * kn, cudaMemcpyDeviceToHost);
	// cudaMemcpy(y_imag_gpu, d_y_imag, sizeof(double) * nx * ny * kn, cudaMemcpyDeviceToHost);
	// Test Kernel Processing 
	if (cudaMemcpy(y_real_gpu, d_y_real, sizeof(double) * nx * ny * k, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		throw runtime_error("Error - cudaMemcpy of d_y_real");
	};
	if (cudaMemcpy(y_imag_gpu, d_y_imag, sizeof(double) * nx * ny * k, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		throw runtime_error("Error - cudaMemcpy of d_y_imag");
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
	// display_matrix_mem_real(M_imag_GPU, nx*ny, nx*ny);
	// Free GPU memory
	cudaFree(d_M_real);
	cudaFree(d_M_imag);
	cudaFree(d_S_real);
	cudaFree(d_S_imag);
	// cudaFree(d_RTM_real);
	cudaFree(d_y_real);
	cudaFree(d_y_imag);
	// save output data into text files 
	ofstream my_real("out_real.txt");
	ofstream my_imag("out_imag.txt");
	for (int i = 0; i < nx * ny * k; i++)
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



// Host Multiplication
void H_multiplication(double* Y_real, double* Y_imag, double* A_real, double* A_imag, double* B_real, double* B_imag, int A_r, int A_c, int B_c)
{
	double* sum_real = (double*)malloc(1 * sizeof(double));
	double* sum_imag = (double*)malloc(1 * sizeof(double));
	double* t_real = (double*)malloc(1 * sizeof(double));
	double* t_imag = (double*)malloc(1 * sizeof(double));

	for (int i = 0; i < A_r; i++)
	{
		for (int j = 0; j < B_c; j++)
		{
			*(t_real) = 0;
			*(t_imag) = 0;
			*(sum_real) = 0;
			*(sum_imag) = 0;
			double x1, y1, u1, v2;
			for (int k = 0; k < A_c; k++)
			{
				x1 = *(A_real + i * A_c + k);
				y1 = *(A_imag + i * A_c + k);
				u1 = *(B_real + k * B_c + j);
				v2 = *(B_imag + k * B_c + j);
				*(t_real) = x1 * u1 - y1 * v2;
				*(t_imag) = x1 * v2 + y1 * u1;
				*(sum_real) = *(sum_real)+*(t_real);
				*(sum_imag) = *(sum_imag)+*(t_imag);
			}
			*(Y_real + i * B_c + j) = *(sum_real);
			*(Y_imag + i * B_c + j) = *(sum_imag);
		}
	}
	free(sum_real);
	free(sum_imag);
	free(t_real);
	free(t_imag);
}
// CUDA Kernel for Multiplication
__global__ void D_multiplication(double* C_real, double* C_imag, double* A_real, double* A_imag, double* B_real, double* B_imag, int R_A, int C_A, int C_B)
{
	// Compute each thread's global row and column index
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	double t1_real = 0;
	double t1_imag = 0;
	double sum_real = 0;
	double sum_imag = 0;
	double x1, y1, u1, v2;
	if (i < R_A && j < C_B)
		for (int k = 0; k < C_A; k++)
		{
			x1 = *(A_real + i * C_A + k);
			y1 = *(A_imag + i * C_A + k);
			u1 = *(B_real + k * C_B + j);
			v2 = *(B_imag + k * C_B + j);
			t1_real = x1 * u1 - y1 * v2;
			t1_imag = x1 * v2 + y1 * u1;
			sum_real = sum_real + t1_real;
			sum_imag = sum_imag + t1_imag;
		}
	*(C_real + i * C_B + j) = sum_real;
	*(C_imag + i * C_B + j) = sum_imag;
}

// Multiplication with real part only
void H_rmultiplication(double* Y_real, double* Y_imag, double* A_real, double* A_imag, double* B_real, int A_r, int A_c, int B_c)
{
	double* sum_real = (double*)malloc(1 * sizeof(double));
	double* sum_imag = (double*)malloc(1 * sizeof(double));
	double* t_real = (double*)malloc(1 * sizeof(double));
	double* t_imag = (double*)malloc(1 * sizeof(double));
	for (int i = 0; i < A_r; i++)
	{
		for (int j = 0; j < B_c; j++)
		{
			*(sum_real) = 0;
			*(sum_imag) = 0;
			for (int k = 0; k < A_c; k++)
			{
				*(sum_real) = *(sum_real)+*(A_real + i * A_c + k) * *(B_real + k * B_c + j);
				*(sum_imag) = *(sum_imag)+*(A_imag + i * A_c + k) * *(B_real + k * B_c + j);
			}
			*(Y_real + i * B_c + j) = *(sum_real);
			*(Y_imag + i * B_c + j) = *(sum_imag);
		}
	}
}

// CUDA kernel multiplication with real part only

__global__ void D_rmultiplication(double* Y_real, double* Y_imag, double* A_real, double* A_imag, double* B_real, int A_r, int A_c, int B_c)
{
	// Compute each thread's global row and column index
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	double sum_real = 0;
	double sum_imag = 0;
	if (i < A_r && j < B_c)
		// for (int i = 0; i < A_r; i++)
		// {
			// for (int j = 0; j < B_c; j++)
			// {

		for (int k = 0; k < A_c; k++)
		{
			(sum_real) = (sum_real)+*(A_real + i * A_c + k) * *(B_real + k * B_c + j);
			(sum_imag) = (sum_imag)+*(A_imag + i * A_c + k) * *(B_real + k * B_c + j);
		}
	*(Y_real + i * B_c + j) = (sum_real);
	*(Y_imag + i * B_c + j) = (sum_imag);
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