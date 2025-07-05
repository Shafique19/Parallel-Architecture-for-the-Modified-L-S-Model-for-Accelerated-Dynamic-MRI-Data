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
void H_fftshift_D1(double*, double*, double*, double*, int, int, int, int);
__global__ void D_fftshift_D1(double*, double*, double*, double*, int, int, int, int, double*, double*, double*, double*, double*, double*);
// Main function start 
void main()
{
	int nx = 256;
	int ny = 256;
	int nt = 11;
	int nc = 30;
	double* M_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* M_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_real_cpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag_cpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_real_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
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
	// CPU Task
	int start_ffts = clock();
	// H_fftshift_D1(y_real_cpu, y_imag_cpu, M_real, M_imag, nx, ny, nt, nc);
	int stop_ffts = clock();
	cout << "\n CPU_TempFFT Time:" << (stop_ffts - start_ffts) << endl;
	// create a stop watch to record time for CUDA Kernel;
	cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	// Define Pointers for GPU or Device Memory
	cudaEventRecord(start_g);
	double* d_y_real, * d_y_imag, * d_x_real, * d_x_imag;
	double* d_X1_real, * d_X1_imag, * d_temp1_real, * d_temp1_imag, * d_temp2_real, * d_temp2_imag;
	// Allocate Device Memory 
	cudaMalloc((void**)&d_y_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_y_imag, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_x_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_x_imag, sizeof(double) * nx * ny * nt * nc);
	// Allocate Device memory pointers 
	cudaMalloc((void**)&d_X1_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_X1_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_temp1_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_temp1_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_temp2_real, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_temp2_imag, sizeof(double) * nx * ny);
	// Copy Data from the host to the device pointer
	cudaMemcpy(d_x_real, M_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_imag, M_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	// Create threads and Blocks for CUDA kernel
	// Launch Kernel
	// cudaEventRecord(start_g);
	D_fftshift_D1 <<<1, nc >> > (d_y_real, d_y_imag, d_x_real, d_x_imag, nx, ny, nt, nc, d_X1_real, d_X1_imag, d_temp1_real, d_temp1_imag, d_temp2_real, d_temp2_imag);
	cudaDeviceSynchronize();
	// cudaEventRecord(stop_g);
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
	// sync
	cudaEventRecord(stop_g);
	cudaEventSynchronize(stop_g);
	// Print Time
	float t = 0;
	cudaEventElapsedTime(&t, start_g, stop_g);
	printf("\n GPU_ FFT_Shift_D1_First_Time: %f", t);
	// Free GPU memory
	cudaFree(d_x_real);
	cudaFree(d_x_imag);
	cudaFree(d_y_real);
	cudaFree(d_y_imag);
	cudaFree(d_X1_real);
	cudaFree(d_X1_imag);
	cudaFree(d_temp1_real);
	cudaFree(d_temp1_imag);
	cudaFree(d_temp2_real);
	cudaFree(d_temp2_imag);
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
// Device_D_FFTshift_Function
__global__ void D_fftshift_D1(double* Y2_real, double* Y2_imag, double* X2_real, double* X2_imag, int nx, int ny, int nt, int nc, double* d_X1_real, double* d_X1_imag, double* d_temp1_real, double* d_temp1_imag, double* d_temp2_real, double* d_temp2_imag)
{
	/*double* X1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* X1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp2_real = (double*)malloc(nx * ny * sizeof(double));
	double* temp2_imag = (double*)malloc(nx * ny * sizeof(double));
	*/
	int dim = 1;
	int K = nx / 2;
	int jj = blockIdx.x * blockDim.x + threadIdx.x;
	// int jj = threadIdx.x;
	//for (int jj = 0; jj < nc; jj++)
	//{
	int l = 0;

	for (int k = 0; k < nx * ny * nt; k++)
	{

		*(d_X1_real + l) = *(X2_real + nx * ny * nt * jj + k);
		*(d_X1_imag + l) = *(X2_imag + nx * ny * nt * jj + k);
		l = l + 1;
	}

	for (int i = 0; i < nx * ny * nt; i++)
	{
		*(d_temp1_real + i) = 0;
		*(d_temp1_imag + i) = 0;

		if (i < nx * ny)
		{
			*(d_temp2_real + i) = 0;
			*(d_temp2_imag + i) = 0;
		}
	}

	for (int j = 0; j < K; j++)
	{
		//int ll = 0; 
		for (int i = 0; i < nx; i++)
		{
			for (int k = 0; k < ny; k++)
			{
				for (int m = 0; m < nt; m++)
				{
					if (i < (nx - 1))
					{
						*(d_temp1_real + ny * i + m * ny * (nx - 1) + k) = *(d_X1_real + ny * i + nx * ny * m + k);
						*(d_temp1_imag + ny * i + m * ny * (nx - 1) + k) = *(d_X1_imag + ny * i + nx * ny * m + k);
					}
					else
					{
						*(d_temp2_real + m * ny + k) = *(d_X1_real + ny * i + nx * ny * m + k);
						*(d_temp2_imag + m * ny + k) = *(d_X1_imag + ny * i + nx * ny * m + k);
						//ll = ll + 1;
					}
				}
			}
		}

		for (int i = 0; i < nx; i++)
		{
			for (int k = 0; k < ny; k++)
			{
				for (int m = 0; m < nt; m++)
				{
					if (i < 1)
					{
						*(d_X1_real + ny * i + ny * i + nx * ny * m + k) = *(d_temp2_real + m * ny + k);
						*(d_X1_imag + ny * i + ny * i + nx * ny * m + k) = *(d_temp2_imag + m * ny + k);
					}
					else
					{
						*(d_X1_real + ny * i + nx * ny * m + k) = *(d_temp1_real + ny * (i - 1) + m * ny * (nx - 1) + k);
						*(d_X1_imag + ny * i + nx * ny * m + k) = *(d_temp1_imag + ny * (i - 1) + m * ny * (nx - 1) + k);
						//ll = ll + 1;
					}
				}
			}
		}

		int l = 0;
		for (int k = 0; k < nx * ny * nt; k++)
		{
			*(X2_real + nx * ny * nt * jj + k) = *(d_X1_real + l);
			*(X2_imag + nx * ny * nt * jj + k) = *(d_X1_imag + l);
			l = l + 1;
		}
	}
	//}
}

// // // // 
void H_fftshift_D1(double* Y2_real, double* Y2_imag, double* X2_real, double* X2_imag, int nx, int ny, int nt, int nc)
{
	double* x1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* x1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* X1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* X1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp2_real = (double*)malloc(nx * ny * sizeof(double));
	double* temp2_imag = (double*)malloc(nx * ny * sizeof(double));
	// int dim = 1;
	int K = nx / 2;
	for (int jj = 0; jj < nc; jj++)
	{
		int l = 0;

		for (int k = 0; k < nx * ny * nt; k++)
		{

			*(X1_real + l) = *(X2_real + nx * ny * nt * jj + k);
			*(X1_imag + l) = *(X2_imag + nx * ny * nt * jj + k);
			l = l + 1;
		}

		//		cout << "X1 is : 2";
		//		display_3d_matrix_mem_complex (X1_real, X1_imag, nx, ny, nt);


		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(temp1_real + i) = 0;
			*(temp1_imag + i) = 0;

			if (i < nx * ny)
			{
				*(temp2_real + i) = 0;
				*(temp2_imag + i) = 0;
			}
		}

		for (int j = 0; j < K; j++)
		{
			//int ll = 0; 
			for (int i = 0; i < nx; i++)
			{
				for (int k = 0; k < ny; k++)
				{
					for (int m = 0; m < nt; m++)
					{
						if (i < (nx - 1))
						{
							*(temp1_real + ny * i + m * ny * (nx - 1) + k) = *(X1_real + ny * i + nx * ny * m + k);
							*(temp1_imag + ny * i + m * ny * (nx - 1) + k) = *(X1_imag + ny * i + nx * ny * m + k);
						}
						else
						{
							*(temp2_real + m * ny + k) = *(X1_real + ny * i + nx * ny * m + k);
							*(temp2_imag + m * ny + k) = *(X1_imag + ny * i + nx * ny * m + k);
							//ll = ll + 1;
						}
					}
				}
			}
			//			cout << "temp1 is: ";
			//			display_3d_matrix_mem_complex (temp1_real, temp1_imag, (nx-1), ny, nt);
			//			cout << "temp2 is: ";
			//			display_3d_matrix_mem_complex (temp2_real, temp2_imag, 1, ny, nt);

			for (int i = 0; i < nx; i++)
			{
				for (int k = 0; k < ny; k++)
				{
					for (int m = 0; m < nt; m++)
					{
						if (i < 1)
						{
							*(X1_real + ny * i + ny * i + nx * ny * m + k) = *(temp2_real + m * ny + k);
							*(X1_imag + ny * i + ny * i + nx * ny * m + k) = *(temp2_imag + m * ny + k);
						}
						else
						{
							*(X1_real + ny * i + nx * ny * m + k) = *(temp1_real + ny * (i - 1) + m * ny * (nx - 1) + k);
							*(X1_imag + ny * i + nx * ny * m + k) = *(temp1_imag + ny * (i - 1) + m * ny * (nx - 1) + k);
							//ll = ll + 1;
						}
					}
				}
			}
			//			cout << "X1 after rotate is: ";
			//			display_3d_matrix_mem_complex (X1_real, X1_imag, nx, ny, nt);

			int l = 0;
			for (int k = 0; k < nx * ny * nt; k++)
			{
				*(X2_real + nx * ny * nt * jj + k) = *(X1_real + l);
				*(X2_imag + nx * ny * nt * jj + k) = *(X1_imag + l);
				l = l + 1;
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
