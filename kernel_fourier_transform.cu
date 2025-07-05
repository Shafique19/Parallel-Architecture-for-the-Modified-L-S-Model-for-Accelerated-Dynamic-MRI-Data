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
void display_matrix_mem_real(double*, int, int);
void H_Ifft_D1(double*, double*, double*, double*, int, int, int, int);
__global__ void D_Ifft_D1(double*, double*, double*, double*, int, int, int, int,
	double*, double*, double*, double*, double*, double*, double*,
	double*, double*, double*, double*, double*, double*, double*,
	double*, double*, double*, double*);
// Main Function start here 
void main()
{
	int nx = 256;
	int ny = 256;
	int nt = 11;
	int nc = 30;
	double* M_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* M_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_real_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	// Reading files from directory
	cout << "\t\t Opening File to Read .......................................................\n";
	ifstream M_read_real("M_real.txt");
	ifstream M_read_imag("M_imag.txt");
	// Check if all files available in directory
	if (!M_read_real || !M_read_imag)
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
	int start_c = clock();
	// H_Ifft_D1(y_real, y_imag, M_real, M_imag, nx, ny, nt, nc);
	int stop_c = clock();
	cout << "\n CPU_TempFFT Time:" << (stop_c - start_c) << endl;
	// Inverse FFT (IFFT CUDA Kerenl)
	// create a stop watch to record time for CUDA Kernel;
	cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	// Define Pointers for GPU or Device Memory
	cudaEventRecord(start_g);
	double* d_yy_real, * d_yy_imag, * d_xx_real, * d_xx_imag;
	// Allocate Device Memory 
	cudaMalloc((void**)&d_yy_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_yy_imag, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_xx_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_xx_imag, sizeof(double) * nx * ny * nt * nc);
	// Pass device pointers 
	double* d_x1_real, * d_x1_imag, * d_x_real, * d_x_imag, * d_y_real, * d_y_imag;
	double*d_Xk_real, * d_Xk_imag, * d_out_real, * d_out_imag, * d_n_real, * d_n_imag;
	double* d_k0_real, * d_k0_imag, * d_nk_real, * d_nk_imag, * d_WNnk_real, * d_WNnk_imag; 
	cudaMalloc((void**)&d_x1_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_x1_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_x_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_x_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_y_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_y_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_Xk_real, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_Xk_imag, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_out_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_out_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_n_real, sizeof(double) * nt);
	cudaMalloc((void**)&d_n_imag, sizeof(double) * nt);
	cudaMalloc((void**)&d_k0_real, sizeof(double) * nt);
	cudaMalloc((void**)&d_k0_imag, sizeof(double) * nt);
	cudaMalloc((void**)&d_nk_real, sizeof(double) * nx * nx * nx);
	cudaMalloc((void**)&d_nk_imag, sizeof(double) * nx * nx * nx);
	cudaMalloc((void**)&d_WNnk_real, sizeof(double) * nx * nx * nx);
	cudaMalloc((void**)&d_WNnk_imag, sizeof(double) * nx * nx * nx);
	// Copy Data from the host to the device pointer
	cudaMemcpy(d_xx_real, M_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_xx_imag, M_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	// Create threads and Blocks for CUDA kernel
	// dim3 blocks(1, 10);
	// dim3 grids((1, nc / 10));
	// Launch Kernel
	// cudaEventRecord(start_g);
	D_Ifft_D1 << <1, nc >> > (d_yy_real, d_yy_imag, d_xx_real, d_xx_imag, nx, ny, nt, nc, d_x1_real, d_x1_imag,
		d_x_real, d_x_imag, d_y_real, d_y_imag, d_Xk_real, d_Xk_imag, d_out_real, d_out_imag, d_n_real, d_n_imag, 
		d_k0_real, d_k0_imag,d_nk_real, d_nk_imag, d_WNnk_real, d_WNnk_imag);
	cudaDeviceSynchronize();
	// cudaEventRecord(stop_g);
	// Copy Result from Device to Host
	cudaMemcpy(y_real_gpu, d_yy_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_imag_gpu, d_yy_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
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
	cudaFree(d_xx_real);
	cudaFree(d_xx_imag);
	cudaFree(d_yy_real);
	cudaFree(d_yy_imag);
	cudaFree(d_x1_real);
	cudaFree(d_x1_imag);
	cudaFree(d_x_real);
	cudaFree(d_x_imag);
	cudaFree(d_y_real);
	cudaFree(d_y_imag);
	cudaFree(d_Xk_real);
	cudaFree(d_Xk_imag);
	cudaFree(d_out_real);
	cudaFree(d_out_imag);
	cudaFree(d_n_real);
	cudaFree(d_n_imag);
	cudaFree(d_k0_real);
	cudaFree(d_k0_imag);
	cudaFree(d_nk_real);
	cudaFree(d_nk_imag);
	cudaFree(d_WNnk_real);
	cudaFree(d_WNnk_imag);

	// save output data into text files 
	ofstream my_real("out_real.txt");
	ofstream my_imag("out_imag.txt");
	for (int i = 0; i < nx*ny*nt*nc; i++)
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
// Data Consistency FFT_D1 
__global__ void D_Ifft_D1(double* Y2_real, double* Y2_imag, double* X2_real, double* X2_imag, int nx, int ny, int nt, int nc,
	double* d_x1_real, double* d_x1_imag, double* d_x_real, double* d_x_imag, double* d_y_real, double* d_y_imag,double* d_Xk_real,
	double* d_Xk_imag, double* d_out_real, double* d_out_imag, double* d_n_real, double* d_n_imag,double* d_k0_real, double* d_k0_imag,
	double* d_nk_real, double* d_nk_imag, double* d_WNnk_real, double* d_WNnk_imag)
{
	// double* x1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	// double* x1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// double* x_real = (double*)malloc(nx * ny * nt * sizeof(double));
	// double* x_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// double* y_real = (double*)malloc(nx * ny * nt * sizeof(double));
	// double* y_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// double* Xk_real = (double*)malloc(nx * ny * sizeof(double));
	// double* Xk_imag = (double*)malloc(nx * ny * sizeof(double));
	// double* out_real = (double*)malloc(nx * ny * nt * sizeof(double));
	// double* out_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// double* n_real = (double*)malloc(nt * sizeof(double));
	// double* n_imag = (double*)malloc(nt * sizeof(double));
	// double* k0_real = (double*)malloc(nt * sizeof(double));
	// double* k0_imag = (double*)malloc(nt * sizeof(double));
	// double* nk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	// double* nk_imag = (double*)malloc(nx * nx * nx * sizeof(double));
	// double* WNnk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	// double* WNnk_imag = (double*)malloc(nx * nx * nx * sizeof(double));
	int sign_cos, sign_sin;
	double WN_real, WN_imag;
	double PI = 3.14159265;
	int nr = nc;
	int kk = blockIdx.x * blockDim.x + threadIdx.x;
	//for (int kk = 0; kk < nr; kk++)
	//{
	int l = 0;

	for (int k = 0; k < nx * ny * nt; k++)
	{

		*(d_x1_real + l) = *(X2_real + nx * ny * nt * kk + k);
		*(d_x1_imag + l) = *(X2_imag + nx * ny * nt * kk + k);
		l = l + 1;
	}
	//		cout << "x1 is : 2";
	//		display_3d_matrix_mem_complex (x1_real, x1_imag, nx, ny, nt);

	for (int i = 0; i < nt; i++)
	{
		for (int j = 0; j < nx * ny; j++)
		{
			*(d_x_real + j) = *(d_x1_real + nx * ny * i + j);
			*(d_x_imag + j) = *(d_x1_imag + nx * ny * i + j);
			*(d_y_real + j) = 0;
			*(d_y_imag + j) = 0;
		}
		//			cout << "x is : 2";
		//			display_matrix_mem_complex (x_real, x_imag, nx, ny);

		int C = ny;
		for (int c = 0; c < C; c++)
		{
			for (int jj = 0; jj < nx; jj++)
			{
				*(d_Xk_real + jj * ny) = *(d_x_real + ny * jj + c);
				*(d_Xk_imag + jj * ny) = *(d_x_imag + ny * jj + c);
				//								*(Xk_real + jj) = *(x_real + ny*jj + c);
				//								*(Xk_imag + jj) = *(x_imag + ny*jj + c);
			}
			//				cout << "Xk is :";
			//				display_matrix_mem_complex (Xk_real, Xk_imag, nx, 1);

			int N = nx;
			for (int ii = 0; ii < N; ii++)
			{
				*(d_n_real + ii) = ii;
				*(d_k0_real + ii) = ii;
				*(d_n_imag + ii) = ii;
				*(d_k0_imag + ii) = ii;
			}

			//							WN_real = 0.9988;		WN_imag = -0.0491;
			WN_real = cos(2 * PI / ny);	WN_imag = sin(-2 * PI / ny);
			if (WN_real < 0)
				sign_cos = -1;
			else
				sign_cos = 1;

			if (WN_imag < 0)
				sign_sin = -1;
			else
				sign_sin = 1;
			double e = 2.0; 
			double mag = sqrt(pow(WN_real, e) + pow(WN_imag, e));
			double angle = atan(WN_imag / WN_real) * 180 / PI;
			//N = nx; 
			for (int ii = 0; ii < nx; ii++)
			{
				for (int jj = 0; jj < nx; jj++)
				{
					double sum0_real = 0;	// double sum0_imag = 0;
					for (int cc = 0; cc < 1; cc++)
					{
						sum0_real = sum0_real + (*(d_n_real + ii + cc) * *(d_k0_real + cc + jj));	// Check whether this indexing is correct
						//sum0_imag = sum0_imag + (*(n_real + ii*nx + cc) * *(k0_imag + cc + jj)) + (*(n_imag + ii*nx + cc) * *(k0_real + cc*nt + jj));
					}
					*(d_nk_real + ii * nx + jj) = sum0_real * -1;
					//*(nk_imag   + ii*nx + jj) = sum0_imag;
					if (*(d_nk_real + ii * nx + jj) == 0)
					{
						*(d_WNnk_real + ii * nx + jj) = 1;
						*(d_WNnk_imag + ii * nx + jj) = 0;
					}
					else
					{
						*(d_WNnk_real + ii * nx + jj) = pow(mag, *(d_nk_real + ii * nx + jj)) * cos(angle * *(d_nk_real + ii * nx + jj) * PI / 180) * sign_cos;		// WNnk (i, j) = WN ^ nk (i, j); Correct this one
						*(d_WNnk_imag + ii * nx + jj) = pow(mag, *(d_nk_real + ii * nx + jj)) * sin(angle * *(d_nk_real + ii * nx + jj) * PI / 180) * sign_sin;		// WNnk (i, j) = WN ^ nk (i, j); Correct this one
					}
				}
			}

			//				cout << "WNnk is :";
			//				display_matrix_mem_complex (WNnk_real, WNnk_imag, nx, ny);
			//				cout << "nk is : 2";
			//				display_matrix_mem_complex (nk_real, nk_imag, nx, ny);


			for (int ii = 0; ii < nx; ii++)
			{
				for (int jj = 0; jj < 1; jj++)
				{
					double sum0_real = 0;	double sum0_imag = 0;
					for (int cc = 0; cc < ny; cc++)
					{
						sum0_real = sum0_real + (*(d_WNnk_real + ii * ny + cc) * *(d_Xk_real + cc * nx + jj)) - (*(d_WNnk_imag + ii * ny + cc) * *(d_Xk_imag + cc * nx + jj));
						sum0_imag = sum0_imag + (*(d_WNnk_real + ii * ny + cc) * *(d_Xk_imag + cc * nx + jj)) + (*(d_WNnk_imag + ii * ny + cc) * *(d_Xk_real + cc * nx + jj));
					}

					*(d_y_real + ii * ny + c) = sum0_real / N;
					*(d_y_imag + ii * ny + c) = sum0_imag / N;
				}
			}
		}
		//			cout << "y is : 2";
		//			display_matrix_mem_complex (y_real, y_imag, nx, ny);
		for (int ii = 0; ii < nx * ny; ii++)
		{
			*(d_out_real + i * nx * ny + ii) = *(d_y_real + ii);
			*(d_out_imag + i * nx * ny + ii) = *(d_y_imag + ii);
		}

		//			cout << "out is :";
		//			display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);

	}

	l = 0;
	for (int ii = 0; ii < nx * ny * nt; ii++)
	{
		*(X2_real + nx * ny * nt * kk + ii) = *(d_out_real + l);
		*(X2_imag + nx * ny * nt * kk + ii) = *(d_out_imag + l);
		l = l + 1;
	}
	//}
}
// // // // 
// Data Consistency FFT_D1 
void H_Ifft_D1(double* Y2_real, double* Y2_imag, double* X2_real, double* X2_imag, int nx, int ny, int nt, int nc)
{
	double* x1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* x1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* x_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* x_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* Xk_real = (double*)malloc(nx * ny * sizeof(double));
	double* Xk_imag = (double*)malloc(nx * ny * sizeof(double));
	double* out_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* out_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* n_real = (double*)malloc(nt * sizeof(double));
	double* n_imag = (double*)malloc(nt * sizeof(double));
	double* k0_real = (double*)malloc(nt * sizeof(double));
	double* k0_imag = (double*)malloc(nt * sizeof(double));
	double* nk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	double* nk_imag = (double*)malloc(nx * nx * nx * sizeof(double));
	double* WNnk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	double* WNnk_imag = (double*)malloc(nx * nx * nx * sizeof(double));
	int sign_cos, sign_sin;
	double WN_real, WN_imag;
	double PI = 3.14159265;
	int nr = nc;

	for (int kk = 0; kk < nr; kk++)
	{
		int l = 0;

		for (int k = 0; k < nx * ny * nt; k++)
		{

			*(x1_real + l) = *(X2_real + nx * ny * nt * kk + k);
			*(x1_imag + l) = *(X2_imag + nx * ny * nt * kk + k);
			l = l + 1;
		}
		//		cout << "x1 is : 2";
		//		display_3d_matrix_mem_complex (x1_real, x1_imag, nx, ny, nt);

		for (int i = 0; i < nt; i++)
		{
			for (int j = 0; j < nx * ny; j++)
			{
				*(x_real + j) = *(x1_real + nx * ny * i + j);
				*(x_imag + j) = *(x1_imag + nx * ny * i + j);
				*(y_real + j) = 0;
				*(y_imag + j) = 0;
			}
			//			cout << "x is : 2";
			//			display_matrix_mem_complex (x_real, x_imag, nx, ny);

			int C = ny;
			for (int c = 0; c < C; c++)
			{
				for (int jj = 0; jj < nx; jj++)
				{
					*(Xk_real + jj * ny) = *(x_real + ny * jj + c);
					*(Xk_imag + jj * ny) = *(x_imag + ny * jj + c);
					//								*(Xk_real + jj) = *(x_real + ny*jj + c);
					//								*(Xk_imag + jj) = *(x_imag + ny*jj + c);
				}
				//				cout << "Xk is :";
				//				display_matrix_mem_complex (Xk_real, Xk_imag, nx, 1);

				int N = nx;
				for (int ii = 0; ii < N; ii++)
				{
					*(n_real + ii) = ii;
					*(k0_real + ii) = ii;
					*(n_imag + ii) = ii;
					*(k0_imag + ii) = ii;
				}

				//							WN_real = 0.9988;		WN_imag = -0.0491;
				WN_real = cos(2 * PI / ny);	WN_imag = sin(-2 * PI / ny);
				if (WN_real < 0)
					sign_cos = -1;
				else
					sign_cos = 1;

				if (WN_imag < 0)
					sign_sin = -1;
				else
					sign_sin = 1;
				double e = 2.0; 
				double mag = sqrt(pow(WN_real, e) + pow(WN_imag, e));
				double angle = atan(WN_imag / WN_real) * 180 / PI;
				//N = nx; 
				for (int ii = 0; ii < nx; ii++)
				{
					for (int jj = 0; jj < nx; jj++)
					{
						double sum0_real = 0;	// double sum0_imag = 0;
						for (int cc = 0; cc < 1; cc++)
						{
							sum0_real = sum0_real + (*(n_real + ii + cc) * *(k0_real + cc + jj));	// Check whether this indexing is correct
							//sum0_imag = sum0_imag + (*(n_real + ii*nx + cc) * *(k0_imag + cc + jj)) + (*(n_imag + ii*nx + cc) * *(k0_real + cc*nt + jj));
						}
						*(nk_real + ii * nx + jj) = sum0_real * -1;
						//*(nk_imag   + ii*nx + jj) = sum0_imag;
						if (*(nk_real + ii * nx + jj) == 0)
						{
							*(WNnk_real + ii * nx + jj) = 1;
							*(WNnk_imag + ii * nx + jj) = 0;
						}
						else
						{
							*(WNnk_real + ii * nx + jj) = pow(mag, *(nk_real + ii * nx + jj)) * cos(angle * *(nk_real + ii * nx + jj) * PI / 180) * sign_cos;		// WNnk (i, j) = WN ^ nk (i, j); Correct this one
							*(WNnk_imag + ii * nx + jj) = pow(mag, *(nk_real + ii * nx + jj)) * sin(angle * *(nk_real + ii * nx + jj) * PI / 180) * sign_sin;		// WNnk (i, j) = WN ^ nk (i, j); Correct this one
						}
					}
				}

				//				cout << "WNnk is :";
				//				display_matrix_mem_complex (WNnk_real, WNnk_imag, nx, ny);
				//				cout << "nk is : 2";
				//				display_matrix_mem_complex (nk_real, nk_imag, nx, ny);


				for (int ii = 0; ii < nx; ii++)
				{
					for (int jj = 0; jj < 1; jj++)
					{
						double sum0_real = 0;	double sum0_imag = 0;
						for (int cc = 0; cc < ny; cc++)
						{
							sum0_real = sum0_real + (*(WNnk_real + ii * ny + cc) * *(Xk_real + cc * nx + jj)) - (*(WNnk_imag + ii * ny + cc) * *(Xk_imag + cc * nx + jj));
							sum0_imag = sum0_imag + (*(WNnk_real + ii * ny + cc) * *(Xk_imag + cc * nx + jj)) + (*(WNnk_imag + ii * ny + cc) * *(Xk_real + cc * nx + jj));
						}

						*(y_real + ii * ny + c) = sum0_real / N;
						*(y_imag + ii * ny + c) = sum0_imag / N;
					}
				}
			}
			//			cout << "y is : 2";
			//			display_matrix_mem_complex (y_real, y_imag, nx, ny);
			for (int ii = 0; ii < nx * ny; ii++)
			{
				*(out_real + i * nx * ny + ii) = *(y_real + ii);
				*(out_imag + i * nx * ny + ii) = *(y_imag + ii);
			}

			//			cout << "out is :";
			//			display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);

		}

		l = 0;
		for (int ii = 0; ii < nx * ny * nt; ii++)
		{
			*(X2_real + nx * ny * nt * kk + ii) = *(out_real + l);
			*(X2_imag + nx * ny * nt * kk + ii) = *(out_imag + l);
			l = l + 1;
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
//////////////////////