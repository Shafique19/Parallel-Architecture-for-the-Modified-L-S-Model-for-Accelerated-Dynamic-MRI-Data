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
// Define no of rows,cols,frames etc.
const int nx = 256;
const int ny = 256;
const int nt = 11;
double nt_db = nt;
double sumx = 0;
// const int nc = 12;
#define PI 3.14159265
__global__ void TempFFT(double*, double*, double*, double*, int, int, int);
__global__ void powsum(double*, double*, double*, double*, double*, double*, int, int, int, double);
__global__ void kernel3(double*, double*, double*, double*, double*, double*, int, int, int);

int main()
{
	// Allocate memory on host pointers
	double* x_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* x_imag = (double*)malloc(nx * ny * nt * sizeof(double));

	double* y_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y_imag = (double*)malloc(nx * ny * nt * sizeof(double));

	double* temp1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp2_real = (double*)malloc(nx * ny * sizeof(double));
	double* temp2_imag = (double*)malloc(nx * ny * sizeof(double));

	double* shft_fft_b_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* shft_fft_b_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* T_M_Lpre_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* T_M_Lpre_imag = (double*)malloc(nx * ny * nt * sizeof(double));

	// Reading text files from directory 
	// Reading Data from the files
	cout << "\t\t opening file to read .......................................................\n";
	ifstream x_read_real("x_real.txt");
	ifstream x_read_imag("x_imag.txt");
	// checking if all the files available in root directory or not
	if (!x_read_real || !x_read_imag)
	{
		cout << "cannot open text files \n";
		(void)getchar();
		return 0;
	}
	else
		cout << "all files found:- \t reading data in progress \n";
	//files reading started 
	cout << "\t\t files reading started ......................................................\n";
	double file_in = 0.0;
	cout << "\t\t reading input data .........................................................\n";
	// m real and cmplex 
	for (int i = 0; i < (nx * ny * nt); i++)
	{
		x_read_real >> file_in;
		*(x_real + i) = file_in;
		x_read_imag >> *(x_imag + i);
	}
	double sum_m = 0;
	double e = 2.0;
	for (int i = 0; i < (nx * ny * nt); i++)
	{
		sum_m = sum_m + pow(*(x_real + i), e) + pow(*(x_imag + i), e);
	}
	double norm_m = sqrt(sum_m);
	cout << "\t\t norm_x of input x_ral and x_imag is = " << norm_m << endl;
	// temp1_and_temp2_generation:
	/*for (int i = 0; i < nx * ny * nt; i++)
	{
		*(temp1_real + i) = 0;
		*(temp1_imag + i) = 0;

		if (i < nx * ny)
		{
			*(temp2_real + i) = 0;
			*(temp2_imag + i) = 0;
		}
	}
	*/
	// GPU_Implementation_Start
	cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	// Define Pointers and allocate memory on GPU
	cudaEventRecord(start_g);
	double* d_x_real, * d_x_imag, * d_y_real, * d_y_imag,* d_temp1_real, * d_temp1_imag, * d_temp2_real, * d_temp2_imag;
	cudaMalloc((void**)&d_x_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_x_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_temp1_real, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_temp1_imag, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_y_real, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_y_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_temp1_real, sizeof(double) * nx * ny * nt );
	cudaMalloc((void**)&d_temp1_imag, sizeof(double) * nx * ny * nt);
	cudaMalloc((void**)&d_temp2_real, sizeof(double) * nx * ny);
	cudaMalloc((void**)&d_temp2_imag, sizeof(double) * nx * ny);
	// Transfer data from host to device
	cudaMemcpy(d_x_real, x_real, sizeof(double) * nx * ny * nt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_imag, x_imag, sizeof(double) * nx * ny * nt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_temp1_real, temp1_real, sizeof(double) * nx * ny * nt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_temp1_imag, temp1_imag, sizeof(double) * nx * ny * nt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_temp2_real, temp2_real, sizeof(double) * nx * ny , cudaMemcpyHostToDevice);
	cudaMemcpy(d_temp2_imag, temp2_imag, sizeof(double) * nx * ny , cudaMemcpyHostToDevice);
	// kernel configuration
	dim3 grid(1, 1);
	dim3 block(nx, ny);
	// cudaEventRecord(start_g);
	TempFFT << <grid, block >> > (d_y_real, d_y_imag, d_x_real, d_x_imag, nx, ny, nt);
	cudaDeviceSynchronize();
	sumx = 0;
	dim3 grid2(nt / 2, nt / 2);
	dim3 block2(nx, ny);
	powsum << <grid2, block2 >> > (d_temp1_real, d_temp1_imag, d_temp2_real, d_temp2_imag, d_y_real, d_y_imag, nx, ny, nt, sumx);
	cudaDeviceSynchronize();
	int K = nt / 2;
	kernel3 << <1, K >> > (d_y_real, d_y_imag, d_temp1_real, d_temp1_imag, d_temp2_real, d_temp2_imag, nx, ny, nt);
	cudaDeviceSynchronize();
	// cudaEventRecord(stop_g, 0);
   // Copy Result from Device to Host
	cudaMemcpy(y_real, d_y_real, sizeof(double) * nx * ny * nt, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_imag, d_y_imag, sizeof(double) * nx * ny * nt, cudaMemcpyDeviceToHost);
	cudaMemcpy(temp1_real, d_temp1_real, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost);
	cudaMemcpy(temp1_imag, d_temp1_imag, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost);
	cudaMemcpy(temp2_real, d_temp2_real, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost);
	cudaMemcpy(temp2_imag, d_temp2_imag, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop_g, 0);
	cudaEventSynchronize(stop_g);
	float t = 0;
	cudaEventElapsedTime(&t, start_g, stop_g);
	printf("\n GPU time is : %f", t);
	// Free GPU memory
	cudaFree(d_y_real);
	cudaFree(d_y_imag);
	cudaFree(d_temp1_real);
	cudaFree(d_temp1_imag);
	cudaFree(d_temp2_real);
	cudaFree(d_temp2_imag);
	// Remaining C++ part 
	for (int i = 0; i < nx * ny * nt; i++)
	{
		*(shft_fft_b_real + i) = *(y_real + i);
		*(shft_fft_b_imag + i) = *(y_imag + i);
	}
	// last part 
	double sqrt_nt;
	sqrt_nt = sqrt(nt_db);

	for (int i = 0; i < nx * ny * nt; i++)
	{
		*(T_M_Lpre_real + i) = *(shft_fft_b_real + i) / sqrt_nt;
		*(T_M_Lpre_imag + i) = *(shft_fft_b_imag + i) / sqrt_nt;
	}
	// save output data into text files 
	ofstream my_real("out_real.txt");
	ofstream my_imag("out_imag.txt");
	for (int i = 0; i < nx * ny * nt; i++)
	{
		double out_r = 0.0;
		double out_i = 0.0;
		out_r = *(T_M_Lpre_real + i);
		my_real << out_r << endl;;
		out_i = *(T_M_Lpre_imag + i);
		my_imag << out_i << endl;
	}
	// close files after saving data
	my_real.close();
	my_imag.close();
	return 0; 
}
// CUDA kernels used in main functions 
__global__ void TempFFT(double* d_y_real, double* d_y_imag, double* d_x_real, double* d_x_imag, int nx, int ny, int nt)
{
	double WN_real, WN_imag;
	double* xn_real = (double*)malloc(nt * sizeof(double));
	double* xn_imag = (double*)malloc(nt * sizeof(double));
	double* n_real = (double*)malloc(nt * sizeof(double));
	double* n_imag = (double*)malloc(nt * sizeof(double));
	double* k0_real = (double*)malloc(nt * sizeof(double));
	double* k0_imag = (double*)malloc(nt * sizeof(double));
	double* nk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	double* nk_imag = (double*)malloc(nx * nx * nx * sizeof(double));
	double* WNnk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	double* WNnk_imag = (double*)malloc(nx * nx * nx * sizeof(double));

	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	//	int idx = c + (r*blockDim.x);


	//	for (int c = 0; c < ny; c++)
	//	{
	//		for (int r = 0; r<nx; r++)
	//		{
	for (int i = 0; i < nt; i++)
	{
		*(xn_real + i) = *(d_x_real + i * nx * ny + r * nx + c);
		*(xn_imag + i) = *(d_x_imag + i * nx * ny + r * nx + c);
	}


	//			cout << "xn is: ";
	//			display_matrix_mem_complex (xn_real, xn_imag, 1, nt);

	for (int i = 0; i < nt; i++)
	{
		*(n_real + i) = i;
		*(k0_real + i) = i;
		*(n_imag + i) = i;
		*(k0_imag + i) = i;
	}


	//WN_real = 0.9877;	WN_imag = -0.1564;
	WN_real = cos(2 * PI / nt);	WN_imag = sin(-2 * PI / nt);
	double e = 2;
	double mag = sqrt(pow(WN_real, e) + pow(WN_imag, e));
	double angle = atan(WN_imag / WN_real) * 180 / PI;

	for (int i = 0; i < nt; i++)
	{
		for (int j = 0; j < nt; j++)
		{
			double sum0_real = 0;	double sum0_imag = 0;
			for (int k = 0; k < 1; k++)
			{
				sum0_real = sum0_real + (*(n_real + i + k) * *(k0_real + k + j));
				//sum0_imag = sum0_imag + (*(n_real + i*nt + k) * *(k0_imag + k*nt + j)) + (*(n_imag + i*nt + k) * *(k0_real + k*nt + j));
			}

			*(nk_real + i * nt + j) = sum0_real;
			*(nk_imag + i * nt + j) = sum0_imag;
			*(WNnk_real + i * nt + j) = pow(mag, *(nk_real + i * nt + j)) * cos(angle * sum0_real * PI / 180);		// WNnk (i, j) = WN ^ nk (i, j); Correct this one
			*(WNnk_imag + i * nt + j) = pow(mag, *(nk_real + i * nt + j)) * sin(angle * sum0_real * PI / 180);		// WNnk (i, j) = WN ^ nk (i, j); Correct this one

		}
	}
	//			display_matrix_mem_complex (nk_real, nk_imag, nx*ny, nt);
	//			display_matrix_mem_complex (WNnk_real, WNnk_imag, nx*ny, nt);

	for (int i = 0; i < nt; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			double sum0_real = 0;	double sum0_imag = 0;
			for (int k = 0; k < nt; k++)
			{
				sum0_real = sum0_real + (*(WNnk_real + i * nt + k) * *(xn_real + k + j)) - (*(WNnk_imag + i * nt + k) * *(xn_imag + k + j));
				sum0_imag = sum0_imag + (*(WNnk_real + i * nt + k) * *(xn_imag + k + j)) + (*(WNnk_imag + i * nt + k) * *(xn_real + k + j));
			}
			*(d_y_real + i * nx * ny + r * nx + c) = sum0_real;
			*(d_y_imag + i * nx * ny + r * nx + c) = sum0_imag;
		}
	}

	//		}
	//
	//	}
}
//
__global__ void powsum(double* d_temp1_real, double* d_temp1_imag, double* d_temp2_real, double* d_temp2_imag, double* d_y_real, double* d_y_imag, int nx, int ny, int nt, double sumx)
{
	//	double *temp1_real  = (double *) malloc (nx*ny*nt * sizeof(double));
	//	double *temp1_imag  = (double *) malloc (nx*ny*nt * sizeof(double));
	//	double *temp2_real  = (double *) malloc (nx*ny * sizeof(double));
	//	double *temp2_imag  = (double *) malloc (nx*ny * sizeof(double));

	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int i = c + (r * blockDim.x);
	//	sumx = sumx + pow (*(d_y_real + i), 2) + pow (*(d_y_imag + i), 2);
	sumx = sumx + (*(d_y_real + i) * *(d_y_real + i)) + (*(d_y_imag + i) * *(d_y_imag + i));
	//	for (int i = 0; i<nx*ny*nt; i++)
	//	{
	*(d_temp1_real + i) = 0;
	*(d_temp1_imag + i) = 0;

	if (i < nx * ny)
	{
		*(d_temp2_real + i) = 0;
		*(d_temp2_imag + i) = 0;
	}
}
//
__global__ void kernel3(double* d_y_real, double* d_y_imag, double* d_temp1_real, double* d_temp1_imag, double* d_temp2_real, double* d_temp2_imag, int nx, int ny, int nt)
{
	int j = threadIdx.x;
	int l = 0;
	for (int i = 0; i < nx * ny * nt; i++)
	{
		if (i < nx * ny * (nt - 1))
		{
			*(d_temp1_real + i) = *(d_y_real + i);
			*(d_temp1_imag + i) = *(d_y_imag + i);
		}
		else
		{
			*(d_temp2_real + l) = *(d_y_real + i);
			*(d_temp2_imag + l) = *(d_y_imag + i);
			l = l + 1;
		}
	}
	//	display_3d_matrix_mem_complex (temp1_real, temp1_imag, nx, ny, (nt-1));
	for (int i = 0; i < nx * ny * nt; i++)
	{
		if (i < nx * ny)
		{
			*(d_y_real + i) = *(d_temp2_real + i);
			*(d_y_imag + i) = *(d_temp2_imag + i);
		}
		else
		{
			//				* (X_real + nx*ny + i) = *(temp1_real - nx*ny + i);
			//				* (X_imag + nx*ny + i) = *(temp1_imag - nx*ny + i);
			*(d_y_real + i) = *(d_temp1_real - nx * ny + i);
			*(d_y_imag + i) = *(d_temp1_imag - nx * ny + i);
		}
	}

}