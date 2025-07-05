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
// Host function:
void H_Ifft_IE(double*, double*, double*, double*, double*, int, int, int, int);
// CUDA kernel
__global__ void D_Ifft_IE(double*, double*, double*, double*, double*, int, int, int, int,
	double*, double*, double*, double* , double*, double*, double*, double*,
	double*, double*, double*, double*, double*, double*, double*, double*,
	double*, double*, double*, double*, double*, double*, double*, double*);
// Test functions 
void display_matrix_mem_real(double*, int, int);
void display_matrix_mem_complex(double*, double*, int, int);
void display_3d_matrix_mem_complex(double*, double*, int, int, int);
void display_4d_matrix_mem_complex(double*, double*, int, int, int, int);
// Main Function start here 
void main()
{
	int nx = 256;
	int ny = 256;
	int nt = 11;
	int nc = 30;
	// pointers defined for input data 
	double* b_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* b_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* a_mask_real = (double*)malloc(nx * ny * nt * sizeof(double));
	// pointers defined for output data
	double* y_real_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag_gpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_real_cpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* y_imag_cpu = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	// Reading files from directory
	cout << "\t\t Opening File to Read .......................................................\n";
	ifstream b_read_real("b_real.txt");
	ifstream b_read_imag("b_imag.txt");
	ifstream mask_read_real("a_mask_real.txt");
	// Check if all files available in directory
	if (!b_read_real || !b_read_imag || !mask_read_real)
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
	// b real and Cmplex 
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		b_read_real >> file_in;
		*(b_real + i) = file_in;
		b_read_imag >> *(b_imag + i);
	}
	double sum_b = 0;
	double e = 2.0; 
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		sum_b = sum_b + pow(*(b_real + i), e) + pow(*(b_imag + i), e);
	}
	double norm_b = sqrt(sum_b);
	cout << "\t\t norm_b of input b_real and b_imag is = " << norm_b << endl;
	// mask_real
	for (int i = 0; i < (nx * ny * nt); i++)
	{
		mask_read_real >> file_in;
		*(a_mask_real + i) = file_in;
		// mask_read_imag >> *(a_mask_imag + i);
	}
	double sum_mask = 0;
	for (int i = 0; i < (nx * ny * nt); i++)
	{
		sum_mask = sum_mask + pow(*(a_mask_real + i), e); // +pow(*(b_imag + i), e);
	}
	double norm_mask = sqrt(sum_mask);
	cout << "\t\t norm_mask of input b_mask_real is = " << norm_mask << endl;
	// CPU (Host) computation time analysis 
	int start_c = clock();
	// H_Ifft_IE(y_real_cpu, y_real_cpu, b_real, b_imag, a_mask_real, nx, ny, nt, nc);
	int stop_c = clock();
	cout << "\n CPU_computation_Time:" << (stop_c - start_c) << endl;
	// CUDA kernel implementation
	// create a stop watch to record time for CUDA Kernel;
	cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	// cudaEventRecord(start_g);
	// Define Pointers for GPU or Device Memory
	double* d_yo_real, * d_yo_imag, * d_b_real, * d_b_imag, * d_a_mask_real;
	// Allocate Device Memory 
	cudaMalloc((void**)&d_yo_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_yo_imag, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_b_real, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_b_imag, sizeof(double) * nx * ny * nt * nc);
	cudaMalloc((void**)&d_a_mask_real, sizeof(double) * nx * ny * nt);
	// Allocate kernel processing pointers:
	double* d_b1_real, * d_b1_imag, * d_out_real, * d_out_imag, * d_X1_real, * d_X1_imag;
	cudaMalloc((void**)&d_b1_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_b1_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_out_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_out_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_X1_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_X1_imag, sizeof(double) * nx * ny * nt);
	//
	double* d_temp1_real, * d_temp1_imag, * d_temp2_real, * d_temp2_imag, * d_x_real, * d_x_imag;
	cudaMalloc((void**)&d_temp1_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_temp1_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_temp2_real, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_temp2_imag, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_x_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_x_imag, sizeof(double) * nx * ny * nt);
	//
	double* d_y_real, * d_y_imag, * d_Xk_real, * d_Xk_imag, * d_n_real, * d_n_imag;
	cudaMalloc((void**)&d_y_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_y_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_Xk_real, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_Xk_imag, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_n_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_n_imag, sizeof(double) * nx * ny * nt);
	//
	double* d_k0_real, * d_k0_imag, * d_nk_real, * d_nk_imag, * d_WNnk_real, * d_WNnk_imag;
	cudaMalloc((void**)&d_k0_real, sizeof(double) * nt);
	cudaMalloc((void**)&d_k0_imag, sizeof(double) * nt);
	cudaMalloc((void**)&d_nk_real, sizeof(double) * nx * nx * nx);
	cudaMalloc((void**)&d_nk_imag, sizeof(double) * nx * nx * nx);
	cudaMalloc((void**)&d_WNnk_real, sizeof(double) * nx * nx * nx);
	cudaMalloc((void**)&d_WNnk_imag, sizeof(double) * nx * nx * nx);

	// Copy Data from the host to the device pointer
	cudaMemcpy(d_b_real, b_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_imag, b_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a_mask_real, a_mask_real, sizeof(double) * nx * ny * nt, cudaMemcpyHostToDevice);
	// Create threads and Blocks for CUDA kernel
	// dim3 threads(1, 1); 
	// dim3 blocks((int)ceil(1 / 1), (int)ceil(nt/1));
	// Launch Kernel
	cudaEventRecord(start_g);
	D_Ifft_IE << <1, nc>> > (d_yo_real, d_yo_imag, d_b_real, d_b_imag, d_a_mask_real, nx, ny, nt, nc, d_b1_real, d_b1_imag, d_out_real, d_out_imag,
		d_X1_real, d_X1_imag, d_temp1_real, d_temp1_imag, d_temp2_real, d_temp2_imag, d_x_real, d_x_imag, d_y_real, d_y_imag, d_Xk_real, d_Xk_imag, d_n_real, d_n_imag,
		d_k0_real, d_k0_imag, d_nk_real, d_nk_imag, d_WNnk_real, d_WNnk_imag);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_g);
	// Copy Result from Device to Host
	cudaMemcpy(y_real_gpu, d_yo_real, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_imag_gpu, d_yo_imag, sizeof(double) * nx * ny * nt * nc, cudaMemcpyDeviceToHost);
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
	cudaFree(d_b_real);
	cudaFree(d_b_imag);
	cudaFree(d_yo_real);
	cudaFree(d_yo_imag);
	cudaFree(d_a_mask_real);
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
// Host function for inverse encoding operator 
void H_Ifft_IE(double* x_array_real, double* x_array_imag, double* b_real, double* b_imag, double* a_mask_real, int nx, int ny, int nt, int nc)
{
	// Allocate Host memory 
	double* b1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* b1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* out_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* out_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* X1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* X1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp2_real = (double*)malloc(nx * ny * sizeof(double));
	double* temp2_imag = (double*)malloc(nx * ny * sizeof(double));
	double* x_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* x_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* Xk_real = (double*)malloc(nx * ny * sizeof(double));
	double* Xk_imag = (double*)malloc(nx * ny * sizeof(double));
	double* n_real = (double*)malloc(nt * sizeof(double));
	double* n_imag = (double*)malloc(nt * sizeof(double));
	double* k0_real = (double*)malloc(nt * sizeof(double));
	double* k0_imag = (double*)malloc(nt * sizeof(double));
	double* nk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	double* nk_imag = (double*)malloc(nx * nx * nx * sizeof(double));
	double* WNnk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	double* WNnk_imag = (double*)malloc(nx * nx * nx * sizeof(double));
	// Define local variables 
	int sign_cos, sign_sin;
	double WN_real, WN_imag;
	double PI = 3.14159265;
	double e = 2.0;
	int nx_db = nx;
	// int a_adjoint = 1;
	int ny_db = ny;
	double sqrt_ny = sqrt(ny_db);
	for (int ch = 0; ch < nc; ch++)
	{
		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(b1_real + i) = *(b_real + nx * ny * nt * ch + i);
			*(b1_imag + i) = *(b_imag + nx * ny * nt * ch + i);
		}

		for (int i = 0; i < nx * ny * nt; i++)
		{
			//			*(X2_real + nx*ny*nt*ch + i) = *(b1_real + i) * *(a_mask_real + i);
			//			*(X2_imag + nx*ny*nt*ch + i) = *(b1_imag + i) * *(a_mask_real + i);
			*(out_real + i) = *(b1_real + i) * *(a_mask_real + i);
			*(out_imag + i) = *(b1_imag + i) * *(a_mask_real + i);
		}
		// int dim = 1;
		int K = ny / 2;

		for (int jj = 0; jj < 1; jj++)
		{
			int l = 0;
			for (int k = 0; k < nx * ny * nt; k++)
			{
				*(X1_real + l) = *(out_real + nx * ny * nt * jj + k);
				*(X1_imag + l) = *(out_imag + nx * ny * nt * jj + k);
				l = l + 1;
			}
			//			cout << "X1 is : 2";
			//			display_3d_matrix_mem_complex (X1_real, X1_imag, nx, ny, nt);

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
				//
				int l = 0;
				for (int k = 0; k < nx * ny * nt; k++)
				{
					*(out_real + nx * ny * nt * jj + k) = *(X1_real + l);
					*(out_imag + nx * ny * nt * jj + k) = *(X1_imag + l);
					l = l + 1;
				}
			}
		}
		//
		for (int i = 0; i < nt; i++)
		{
			for (int j = 0; j < nx * ny; j++)
			{
				*(x_real + j) = *(out_real + nx * ny * i + j);
				*(x_imag + j) = *(out_imag + nx * ny * i + j);
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

				double mag = sqrt(pow(WN_real, e) + pow(WN_imag, e));
				double angle = atan(WN_imag / WN_real) * 180 / PI;
				//N = nx; 
				for (int ii = 0; ii < nx; ii++)
				{
					for (int jj = 0; jj < nx; jj++)
					{
						double sum0_real = 0;	//double sum0_imag = 0;
						for (int cc = 0; cc < 1; cc++)
						{
							sum0_real = sum0_real + (*(n_real + ii + cc) * *(k0_real + cc + jj));	// Check whether this indexing is correct
							//sum0_imag = sum0_imag + (*(n_real + ii*nx + cc) * *(k0_imag + cc + jj)) + (*(n_imag + ii*nx + cc) * *(k0_real + cc*nt + jj));
						}
						*(nk_real + ii * nx + jj) = sum0_real;
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
						for (int cc = 0; cc < nx; cc++)
						{
							sum0_real = sum0_real + (*(WNnk_real + ii * ny + cc) * *(Xk_real + cc * nx + jj)) - (*(WNnk_imag + ii * ny + cc) * *(Xk_imag + cc * nx + jj));
							sum0_imag = sum0_imag + (*(WNnk_real + ii * ny + cc) * *(Xk_imag + cc * nx + jj)) + (*(WNnk_imag + ii * ny + cc) * *(Xk_real + cc * nx + jj));
						}

						*(y_real + ii * ny + c) = sum0_real;
						*(y_imag + ii * ny + c) = sum0_imag;
					}
				}
			}

			//			cout << "y is : 2";
			//			display_matrix_mem_complex (y_real, y_imag, nx, ny);

			int l = 0;
			for (int ii = 0; ii < nx * ny; ii++)
			{
				*(out_real + nx * ny * i + ii) = *(y_real + l);
				*(out_imag + nx * ny * i + ii) = *(y_imag + l);
				l = l + 1;
			}
		}
		//      check point 
		//		cout << "X2 at 889 is :";
		//		display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);

		for (int jj = 0; jj < 1; jj++)
		{
			int l = 0;
			for (int k = 0; k < nx * ny * nt; k++)
			{
				*(X1_real + l) = *(out_real + nx * ny * nt * jj + k);
				*(X1_imag + l) = *(out_imag + nx * ny * nt * jj + k);
				l = l + 1;
			}
			//      check point 
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
					*(out_real + nx * ny * nt * jj + k) = *(X1_real + l);
					*(out_imag + nx * ny * nt * jj + k) = *(X1_imag + l);
					l = l + 1;
				}
			}
		}

		// Line 868

		double sqrt_nx = sqrt(nx_db);

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(out_real + i) = *(out_real + i) / sqrt_nx;
			*(out_imag + i) = *(out_imag + i) / sqrt_nx;
		}

		// int dim = 2;
		K = ny / 2;
		//		cout << "X at Line 913 is:";
		//		display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);

		for (int jj = 0; jj < 1; jj++)
		{
			int l = 0;
			for (int k = 0; k < nx * ny * nt; k++)
			{
				*(X1_real + l) = *(out_real + nx * ny * nt * jj + k);
				*(X1_imag + l) = *(out_imag + nx * ny * nt * jj + k);
				l = l + 1;
			}
			//		cout << "X1 is : ";
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
				for (int i = 0; i < ny; i++)
				{
					for (int k = 0; k < nx; k++)
					{
						for (int m = 0; m < nt; m++)
						{
							if (i < (ny - 1))
							{
								*(temp1_real + nx * i + m * nx * (ny - 1) + k) = *(X1_real + ny * k + nx * ny * m + i);
								*(temp1_imag + nx * i + m * nx * (ny - 1) + k) = *(X1_imag + ny * k + nx * ny * m + i);
							}
							else
							{
								*(temp2_real + m * nx + k) = *(X1_real + nx * ny * m + ny * k + i);
								*(temp2_imag + m * nx + k) = *(X1_imag + nx * ny * m + ny * k + i);
								//ll = ll + 1;
							}
						}
					}
				}
				//			cout << "temp1 is: ";
				//			display_3d_matrix_mem_complex (temp1_real, temp1_imag, nx, (ny-1), nt);
				//			cout << "temp2 is: ";
				//			display_3d_matrix_mem_complex (temp2_real, temp2_imag, nx, 1, nt);

				for (int i = 0; i < ny; i++)
				{
					for (int k = 0; k < nx; k++)
					{
						for (int m = 0; m < nt; m++)
						{
							if (i < 1)
							{
								*(X1_real + nx * ny * m + ny * k + i) = *(temp2_real + m * nx + k);
								*(X1_imag + nx * ny * m + ny * k + i) = *(temp2_imag + m * nx + k);
							}
							else
							{
								*(X1_real + ny * k + nx * ny * m + i) = *(temp1_real + nx * (i - 1) + m * nx * (ny - 1) + k);
								*(X1_imag + ny * k + nx * ny * m + i) = *(temp1_imag + nx * (i - 1) + m * nx * (ny - 1) + k);
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
					*(out_real + nx * ny * nt * jj + k) = *(X1_real + l);
					*(out_imag + nx * ny * nt * jj + k) = *(X1_imag + l);
					l = l + 1;
				}
			}
		}
		// check point 
		//	cout << "X at Line 929 is:";
		//	display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);
		// Line 898

		for (int j = 0; j < nx * ny; j++)
		{
			*(nk_real + j) = 0;
			*(nk_imag + j) = 0;
			*(Xk_real + j) = 0;
			*(Xk_imag + j) = 0;
			*(WNnk_real + j) = 0;
			*(WNnk_imag + j) = 0;
		}
		for (int i = 0; i < nt; i++)
		{
			for (int j = 0; j < nx * ny; j++)
			{
				*(x_real + j) = *(out_real + nx * ny * i + j);
				*(x_imag + j) = *(out_imag + nx * ny * i + j);
				*(y_real + j) = 0;
				*(y_imag + j) = 0;
			}
			//         check point 
			//			cout << "x is : 2";
			//			display_matrix_mem_complex (x_real, x_imag, nx, ny);

			int C = nx;
			for (int c = 0; c < C; c++)
			{
				for (int jj = 0; jj < nx; jj++)
				{
					*(Xk_real + jj * ny) = *(x_real + ny * c + jj);
					*(Xk_imag + jj * ny) = *(x_imag + ny * c + jj);
				}
				// check point 
				//				cout << "Xk is :";
				//				display_matrix_mem_complex (Xk_real, Xk_imag, ny, 1);

				int N = nx;
				for (int ii = 0; ii < N; ii++)
				{
					*(n_real + ii) = ii;
					*(k0_real + ii) = ii;
					*(n_imag + ii) = ii;
					*(k0_imag + ii) = ii;
				}

				// WN_real = 0.9988;		WN_imag = -0.0491;
				WN_real = cos(2 * PI / nx);	WN_imag = sin(-2 * PI / nx);
				if (WN_real < 0)
					sign_cos = -1;
				else
					sign_cos = 1;

				if (WN_imag < 0)
					sign_sin = -1;
				else
					sign_sin = 1;

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
						*(nk_real + ii * nx + jj) = sum0_real;
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
				//// check point 
				for (int ii = 0; ii < ny; ii++)
				{
					for (int jj = 0; jj < 1; jj++)
					{
						double sum0_real = 0;	double sum0_imag = 0;
						for (int cc = 0; cc < ny; cc++)
						{
							sum0_real = sum0_real + (*(WNnk_real + ii * ny + cc) * *(Xk_real + cc * nx + jj)) - (*(WNnk_imag + ii * ny + cc) * *(Xk_imag + cc * nx + jj));
							sum0_imag = sum0_imag + (*(WNnk_real + ii * ny + cc) * *(Xk_imag + cc * nx + jj)) + (*(WNnk_imag + ii * ny + cc) * *(Xk_real + cc * nx + jj));
						}

						*(y_real + c * ny + ii) = sum0_real;
						*(y_imag + c * ny + ii) = sum0_imag;
					}
				}
			}
			// check point 
			//			cout << "y is : 2";
			//			display_matrix_mem_complex (y_real, y_imag, nx, ny);

			int l = 0;
			for (int ii = 0; ii < nx * ny; ii++)
			{
				*(out_real + nx * ny * i + ii) = *(y_real + l);
				*(out_imag + nx * ny * i + ii) = *(y_imag + l);
				l = l + 1;
			}
		}
		// check point 
		//		cout << "out at Line 975 is:";
		//		display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);
		for (int jj = 0; jj < 1; jj++)
		{
			int l = 0;

			for (int k = 0; k < nx * ny * nt; k++)
			{

				*(X1_real + l) = *(out_real + nx * ny * nt * jj + k);
				*(X1_imag + l) = *(out_imag + nx * ny * nt * jj + k);
				l = l + 1;
			}
			// check point 
			//		cout << "X1 is : ";
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
				for (int i = 0; i < ny; i++)
				{
					for (int k = 0; k < nx; k++)
					{
						for (int m = 0; m < nt; m++)
						{
							if (i < (ny - 1))
							{
								*(temp1_real + nx * i + m * nx * (ny - 1) + k) = *(X1_real + ny * k + nx * ny * m + i);
								*(temp1_imag + nx * i + m * nx * (ny - 1) + k) = *(X1_imag + ny * k + nx * ny * m + i);
							}
							else
							{
								*(temp2_real + m * nx + k) = *(X1_real + nx * ny * m + ny * k + i);
								*(temp2_imag + m * nx + k) = *(X1_imag + nx * ny * m + ny * k + i);
								//ll = ll + 1;
							}
						}
					}
				}
				// check point 
				//			cout << "temp1 is: ";
				//			display_3d_matrix_mem_complex (temp1_real, temp1_imag, nx, (ny-1), nt);
				//			cout << "temp2 is: ";
				//			display_3d_matrix_mem_complex (temp2_real, temp2_imag, nx, 1, nt);

				for (int i = 0; i < ny; i++)
				{
					for (int k = 0; k < nx; k++)
					{
						for (int m = 0; m < nt; m++)
						{
							if (i < 1)
							{
								*(X1_real + nx * ny * m + ny * k + i) = *(temp2_real + m * nx + k);
								*(X1_imag + nx * ny * m + ny * k + i) = *(temp2_imag + m * nx + k);
							}
							else
							{
								*(X1_real + ny * k + nx * ny * m + i) = *(temp1_real + nx * (i - 1) + m * nx * (ny - 1) + k);
								*(X1_imag + ny * k + nx * ny * m + i) = *(temp1_imag + nx * (i - 1) + m * nx * (ny - 1) + k);
								//ll = ll + 1;
							}
						}
					}
				}
				// check point 
				//			cout << "X1 after rotate is: ";
				//			display_3d_matrix_mem_complex (X1_real, X1_imag, nx, ny, nt);

				int l = 0;
				for (int k = 0; k < nx * ny * nt; k++)
				{
					*(out_real + nx * ny * nt * jj + k) = *(X1_real + l);
					*(out_imag + nx * ny * nt * jj + k) = *(X1_imag + l);
					l = l + 1;
				}
			}
		}
		// check point
		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(x_array_real + nx * ny * nt * ch + i) = *(out_real + i) / sqrt_ny;
			*(x_array_imag + nx * ny * nt * ch + i) = *(out_imag + i) / sqrt_ny;
		}

	}
}

// Device kernel for inverse encoding operator:

// Host function for inverse encoding operator 
__global__ void D_Ifft_IE(double* x_array_real, double* x_array_imag, double* b_real, double* b_imag, double* a_mask_real, int nx, int ny, int nt, int nc,
	double* d_b1_real, double* d_b1_imag, double* d_out_real, double* d_out_imag, double* d_X1_real, double* d_X1_imag, double* d_temp1_real, double* d_temp1_imag,
	double* d_temp2_real, double* d_temp2_imag, double* d_x_real, double* d_x_imag, double* d_y_real, double* d_y_imag, double* d_Xk_real, double* d_Xk_imag,
	double* d_n_real, double* d_n_imag, double* d_k0_real, double* d_k0_imag, double* d_nk_real, double* d_nk_imag, double* d_WNnk_real, double* d_WNnk_imag)
{
	// Allocate Host memory 
	//double* b1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* b1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* out_real = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* out_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* X1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* X1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* temp1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* temp1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* temp2_real = (double*)malloc(nx * ny * sizeof(double));
	//double* temp2_imag = (double*)malloc(nx * ny * sizeof(double));
	//double* x_real = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* x_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* y_real = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* y_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* Xk_real = (double*)malloc(nx * ny * sizeof(double));
	//double* Xk_imag = (double*)malloc(nx * ny * sizeof(double));
	//double* n_real = (double*)malloc(nt * sizeof(double));
	//double* n_imag = (double*)malloc(nt * sizeof(double));
	//double* k0_real = (double*)malloc(nt * sizeof(double));
	//double* k0_imag = (double*)malloc(nt * sizeof(double));
	//double* nk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	//double* nk_imag = (double*)malloc(nx * nx * nx * sizeof(double));
	//double* WNnk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	//double* WNnk_imag = (double*)malloc(nx * nx * nx * sizeof(double));
	// Define local variables 
	int sign_cos, sign_sin;
	double WN_real, WN_imag;
	double PI = 3.14159265;
	double nx_db = nx;
	// int a_adjoint = 1;
	double ny_db = ny;
	double sqrt_ny = sqrt(ny_db);

	// Threading 
	int ch = blockIdx.x * blockDim.x + threadIdx.x;
	// int j = blockIdx.x * blockDim.x + threadIdx.x;

	// for (int ch = 0; ch < nc; ch++)
	//{
		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(d_b1_real + i) = *(b_real + nx * ny * nt * ch + i);
			*(d_b1_imag + i) = *(b_imag + nx * ny * nt * ch + i);
		}

		for (int i = 0; i < nx * ny * nt; i++)
		{
			//			*(X2_real + nx*ny*nt*ch + i) = *(b1_real + i) * *(a_mask_real + i);
			//			*(X2_imag + nx*ny*nt*ch + i) = *(b1_imag + i) * *(a_mask_real + i);
			*(d_out_real + i) = *(d_b1_real + i) * *(a_mask_real + i);
			*(d_out_imag + i) = *(d_b1_imag + i) * *(a_mask_real + i);
		}
		// int dim = 1;
		int K = ny / 2;

		for (int jj = 0; jj < 1; jj++)
		{
			int l = 0;
			for (int k = 0; k < nx * ny * nt; k++)
			{
				*(d_X1_real + l) = *(d_out_real + nx * ny * nt * jj + k);
				*(d_X1_imag + l) = *(d_out_imag + nx * ny * nt * jj + k);
				l = l + 1;
			}
			//			cout << "X1 is : 2";
			//			display_3d_matrix_mem_complex (X1_real, X1_imag, nx, ny, nt);

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
				//
				int l = 0;
				for (int k = 0; k < nx * ny * nt; k++)
				{
					*(d_out_real + nx * ny * nt * jj + k) = *(d_X1_real + l);
					*(d_out_imag + nx * ny * nt * jj + k) = *(d_X1_imag + l);
					l = l + 1;
				}
			}
		}
		//
		for (int i = 0; i < nt; i++)
		{
			for (int j = 0; j < nx * ny; j++)
			{
				*(d_x_real + j) = *(d_out_real + nx * ny * i + j);
				*(d_x_imag + j) = *(d_out_imag + nx * ny * i + j);
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
						*(d_nk_real + ii * nx + jj) = sum0_real;
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
						for (int cc = 0; cc < nx; cc++)
						{
							sum0_real = sum0_real + (*(d_WNnk_real + ii * ny + cc) * *(d_Xk_real + cc * nx + jj)) - (*(d_WNnk_imag + ii * ny + cc) * *(d_Xk_imag + cc * nx + jj));
							sum0_imag = sum0_imag + (*(d_WNnk_real + ii * ny + cc) * *(d_Xk_imag + cc * nx + jj)) + (*(d_WNnk_imag + ii * ny + cc) * *(d_Xk_real + cc * nx + jj));
						}

						*(d_y_real + ii * ny + c) = sum0_real;
						*(d_y_imag + ii * ny + c) = sum0_imag;
					}
				}
			}

			//			cout << "y is : 2";
			//			display_matrix_mem_complex (y_real, y_imag, nx, ny);

			int l = 0;
			for (int ii = 0; ii < nx * ny; ii++)
			{
				*(d_out_real + nx * ny * i + ii) = *(d_y_real + l);
				*(d_out_imag + nx * ny * i + ii) = *(d_y_imag + l);
				l = l + 1;
			}
		}
		//      check point 
		//		cout << "X2 at 889 is :";
		//		display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);

		for (int jj = 0; jj < 1; jj++)
		{
			int l = 0;
			for (int k = 0; k < nx * ny * nt; k++)
			{
				*(d_X1_real + l) = *(d_out_real + nx * ny * nt * jj + k);
				*(d_X1_imag + l) = *(d_out_imag + nx * ny * nt * jj + k);
				l = l + 1;
			}
			//      check point 
			//		cout << "X1 is : 2";
			//		display_3d_matrix_mem_complex (X1_real, X1_imag, nx, ny, nt);

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

				//			cout << "X1 after rotate is: ";
				//			display_3d_matrix_mem_complex (X1_real, X1_imag, nx, ny, nt);

				int l = 0;
				for (int k = 0; k < nx * ny * nt; k++)
				{
					*(d_out_real + nx * ny * nt * jj + k) = *(d_X1_real + l);
					*(d_out_imag + nx * ny * nt * jj + k) = *(d_X1_imag + l);
					l = l + 1;
				}
			}
		}

		// Line 868

		double sqrt_nx = sqrt(nx_db);

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(d_out_real + i) = *(d_out_real + i) / sqrt_nx;
			*(d_out_imag + i) = *(d_out_imag + i) / sqrt_nx;
		}

		// int dim = 2;
		K = ny / 2;
		//		cout << "X at Line 913 is:";
		//		display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);

		for (int jj = 0; jj < 1; jj++)
		{
			int l = 0;
			for (int k = 0; k < nx * ny * nt; k++)
			{
				*(d_X1_real + l) = *(d_out_real + nx * ny * nt * jj + k);
				*(d_X1_imag + l) = *(d_out_imag + nx * ny * nt * jj + k);
				l = l + 1;
			}
			//		cout << "X1 is : ";
			//		display_3d_matrix_mem_complex (X1_real, X1_imag, nx, ny, nt);
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
				for (int i = 0; i < ny; i++)
				{
					for (int k = 0; k < nx; k++)
					{
						for (int m = 0; m < nt; m++)
						{
							if (i < (ny - 1))
							{
								*(d_temp1_real + nx * i + m * nx * (ny - 1) + k) = *(d_X1_real + ny * k + nx * ny * m + i);
								*(d_temp1_imag + nx * i + m * nx * (ny - 1) + k) = *(d_X1_imag + ny * k + nx * ny * m + i);
							}
							else
							{
								*(d_temp2_real + m * nx + k) = *(d_X1_real + nx * ny * m + ny * k + i);
								*(d_temp2_imag + m * nx + k) = *(d_X1_imag + nx * ny * m + ny * k + i);
								//ll = ll + 1;
							}
						}
					}
				}
				//			cout << "temp1 is: ";
				//			display_3d_matrix_mem_complex (temp1_real, temp1_imag, nx, (ny-1), nt);
				//			cout << "temp2 is: ";
				//			display_3d_matrix_mem_complex (temp2_real, temp2_imag, nx, 1, nt);

				for (int i = 0; i < ny; i++)
				{
					for (int k = 0; k < nx; k++)
					{
						for (int m = 0; m < nt; m++)
						{
							if (i < 1)
							{
								*(d_X1_real + nx * ny * m + ny * k + i) = *(d_temp2_real + m * nx + k);
								*(d_X1_imag + nx * ny * m + ny * k + i) = *(d_temp2_imag + m * nx + k);
							}
							else
							{
								*(d_X1_real + ny * k + nx * ny * m + i) = *(d_temp1_real + nx * (i - 1) + m * nx * (ny - 1) + k);
								*(d_X1_imag + ny * k + nx * ny * m + i) = *(d_temp1_imag + nx * (i - 1) + m * nx * (ny - 1) + k);
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
					*(d_out_real + nx * ny * nt * jj + k) = *(d_X1_real + l);
					*(d_out_imag + nx * ny * nt * jj + k) = *(d_X1_imag + l);
					l = l + 1;
				}
			}
		}
		// check point 
		//	cout << "X at Line 929 is:";
		//	display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);
		// Line 898

		for (int j = 0; j < nx * ny; j++)
		{
			*(d_nk_real + j) = 0;
			*(d_nk_imag + j) = 0;
			*(d_Xk_real + j) = 0;
			*(d_Xk_imag + j) = 0;
			*(d_WNnk_real + j) = 0;
			*(d_WNnk_imag + j) = 0;
		}
		for (int i = 0; i < nt; i++)
		{
			for (int j = 0; j < nx * ny; j++)
			{
				*(d_x_real + j) = *(d_out_real + nx * ny * i + j);
				*(d_x_imag + j) = *(d_out_imag + nx * ny * i + j);
				*(d_y_real + j) = 0;
				*(d_y_imag + j) = 0;
			}
			//         check point 
			//			cout << "x is : 2";
			//			display_matrix_mem_complex (x_real, x_imag, nx, ny);

			int C = nx;
			for (int c = 0; c < C; c++)
			{
				for (int jj = 0; jj < nx; jj++)
				{
					*(d_Xk_real + jj * ny) = *(d_x_real + ny * c + jj);
					*(d_Xk_imag + jj * ny) = *(d_x_imag + ny * c + jj);
				}
				// check point 
				//				cout << "Xk is :";
				//				display_matrix_mem_complex (Xk_real, Xk_imag, ny, 1);

				int N = nx;
				for (int ii = 0; ii < N; ii++)
				{
					*(d_n_real + ii) = ii;
					*(d_k0_real + ii) = ii;
					*(d_n_imag + ii) = ii;
					*(d_k0_imag + ii) = ii;
				}

				// WN_real = 0.9988;		WN_imag = -0.0491;
				WN_real = cos(2 * PI / nx);	WN_imag = sin(-2 * PI / nx);
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
						*(d_nk_real + ii * nx + jj) = sum0_real;
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
				//// check point 
				for (int ii = 0; ii < ny; ii++)
				{
					for (int jj = 0; jj < 1; jj++)
					{
						double sum0_real = 0;	double sum0_imag = 0;
						for (int cc = 0; cc < ny; cc++)
						{
							sum0_real = sum0_real + (*(d_WNnk_real + ii * ny + cc) * *(d_Xk_real + cc * nx + jj)) - (*(d_WNnk_imag + ii * ny + cc) * *(d_Xk_imag + cc * nx + jj));
							sum0_imag = sum0_imag + (*(d_WNnk_real + ii * ny + cc) * *(d_Xk_imag + cc * nx + jj)) + (*(d_WNnk_imag + ii * ny + cc) * *(d_Xk_real + cc * nx + jj));
						}

						*(d_y_real + c * ny + ii) = sum0_real;
						*(d_y_imag + c * ny + ii) = sum0_imag;
					}
				}
			}
			// check point 
			//			cout << "y is : 2";
			//			display_matrix_mem_complex (y_real, y_imag, nx, ny);

			int l = 0;
			for (int ii = 0; ii < nx * ny; ii++)
			{
				*(d_out_real + nx * ny * i + ii) = *(d_y_real + l);
				*(d_out_imag + nx * ny * i + ii) = *(d_y_imag + l);
				l = l + 1;
			}
		}
		// check point 
		//		cout << "out at Line 975 is:";
		//		display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);
		for (int jj = 0; jj < 1; jj++)
		{
			int l = 0;

			for (int k = 0; k < nx * ny * nt; k++)
			{

				*(d_X1_real + l) = *(d_out_real + nx * ny * nt * jj + k);
				*(d_X1_imag + l) = *(d_out_imag + nx * ny * nt * jj + k);
				l = l + 1;
			}
			// check point 
			//		cout << "X1 is : ";
			//		display_3d_matrix_mem_complex (X1_real, X1_imag, nx, ny, nt);


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
				for (int i = 0; i < ny; i++)
				{
					for (int k = 0; k < nx; k++)
					{
						for (int m = 0; m < nt; m++)
						{
							if (i < (ny - 1))
							{
								*(d_temp1_real + nx * i + m * nx * (ny - 1) + k) = *(d_X1_real + ny * k + nx * ny * m + i);
								*(d_temp1_imag + nx * i + m * nx * (ny - 1) + k) = *(d_X1_imag + ny * k + nx * ny * m + i);
							}
							else
							{
								*(d_temp2_real + m * nx + k) = *(d_X1_real + nx * ny * m + ny * k + i);
								*(d_temp2_imag + m * nx + k) = *(d_X1_imag + nx * ny * m + ny * k + i);
								//ll = ll + 1;
							}
						}
					}
				}
				// check point 
				//			cout << "temp1 is: ";
				//			display_3d_matrix_mem_complex (temp1_real, temp1_imag, nx, (ny-1), nt);
				//			cout << "temp2 is: ";
				//			display_3d_matrix_mem_complex (temp2_real, temp2_imag, nx, 1, nt);

				for (int i = 0; i < ny; i++)
				{
					for (int k = 0; k < nx; k++)
					{
						for (int m = 0; m < nt; m++)
						{
							if (i < 1)
							{
								*(d_X1_real + nx * ny * m + ny * k + i) = *(d_temp2_real + m * nx + k);
								*(d_X1_imag + nx * ny * m + ny * k + i) = *(d_temp2_imag + m * nx + k);
							}
							else
							{
								*(d_X1_real + ny * k + nx * ny * m + i) = *(d_temp1_real + nx * (i - 1) + m * nx * (ny - 1) + k);
								*(d_X1_imag + ny * k + nx * ny * m + i) = *(d_temp1_imag + nx * (i - 1) + m * nx * (ny - 1) + k);
								//ll = ll + 1;
							}
						}
					}
				}
				// check point 
				//			cout << "X1 after rotate is: ";
				//			display_3d_matrix_mem_complex (X1_real, X1_imag, nx, ny, nt);

				int l = 0;
				for (int k = 0; k < nx * ny * nt; k++)
				{
					*(d_out_real + nx * ny * nt * jj + k) = *(d_X1_real + l);
					*(d_out_imag + nx * ny * nt * jj + k) = *(d_X1_imag + l);
					l = l + 1;
				}
			}
		}
		// check point
		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(x_array_real + nx * ny * nt * ch + i) = *(d_out_real + i) / sqrt_ny;
			*(x_array_imag + nx * ny * nt * ch + i) = *(d_out_imag + i) / sqrt_ny;
		}

	//}
}

// Test functions:Function to Dispaly Real Matrix
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