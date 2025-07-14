/********************************************L+S Reconstruction by using cSVD Decomposition Model****************************************************************/
/********************************************Add some essential Libraries **************************************************************************************/
#include <stdio.h>
#include <tchar.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <ctime>
using namespace std;
const int col = 40;
/******************************************** Call these Functions where used *********************************************************************************/
void SVD_Ejacobi(double*, double*, double*, double*, double*, double*, double*, int, int, double);
void SVD_jacobi(double*, double*, double*, double*, double*, double*, double*, int, int, double);
void display_matrix_mem_complex(double*, double*, int, int);
void display_matrix_mem_real(double*, int, int);
void display_3d_matrix_mem_complex(double*, double*, int, int, int);
void display_4d_matrix_mem_complex(double*, double*, int, int, int, int);

/********************************************Main Function Start  ***********************************************************************************************/
void main()
{
	/********************************************Declare  const variables ******************************************************************************************/
		//clock_t begin;
		//float time_spent;
		//begin = clock();
	int nx, ny, nc, nt, K, s, kn, inc, num;
	nx = 256;
	ny = 256;
	nt = 11;
	nc = 30;
	K = 4;
	s = 7;
	kn = s + K;
	inc = nt * kn;
	num = 50;
	double lambda_L, lambda_S, p, abs_in, param_tol, nt_db, nx_db, ny_db, PI, tol = 1e-15;
	lambda_L = 0.01;
	lambda_S = 0.01;
	param_tol = 0.0025;
	nt_db = nt;
	nx_db = nx;
	ny_db = ny;
	PI = 3.14159265;
	int sign_cos, sign_sin;
	double sqrt_ny = sqrt(ny_db);
	/********************************************Allocating memory for the L+S reconstruction data*****************************************************************/
	// 4D_Data
	double* res_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* res_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* res_x_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* res_x_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* d_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* d_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* x_array_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* x_array_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	// 3D Coil Sensitivities
	double* a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* a_b1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	// 3D Mask
	double* a_mask_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* a_mask_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// 3D Low Rank Information
	double* L_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* L_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// 3D Data Matrix M0 and M
	double* M0_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* M0_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* M_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* M_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// Update L=Lpre
	double* Lpre_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* Lpre_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* M_Lpre_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* M_Lpre_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// Updat S Info
	double* S_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* S_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// Extract Low Rank Info
	double* M_S_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* M_S_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// Transpose Of Low rank infor
	double* M_ST_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* M_ST_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	// For Eigen Decomposition ESVD
	double* UE_real = (double*)malloc(kn * kn * sizeof(double));
	double* UE_imag = (double*)malloc(kn * kn * sizeof(double));
	double* D_real = (double*)malloc(kn * kn * sizeof(double));
	double* VE_real = (double*)malloc(kn * kn * sizeof(double));
	double* VE_imag = (double*)malloc(kn * kn * sizeof(double));
	// For SVD Decomposition of Left eigen vectors and values 
	double* U_real = (double*)malloc(kn * kn * sizeof(double));
	double* U_imag = (double*)malloc(kn * kn * sizeof(double));
	double* Sv_real = (double*)malloc(kn * kn * sizeof(double));
	double* V_real = (double*)malloc(nt * kn * sizeof(double));
	double* V_imag = (double*)malloc(nt * kn * sizeof(double));
	// Temporal FFT and Param E memory allocation
	double* X_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* X_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp2_real = (double*)malloc(nx * ny * sizeof(double));
	double* temp2_imag = (double*)malloc(nx * ny * sizeof(double));
	double* shft_fft_b_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* shft_fft_b_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* T_M_Lpre_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* T_M_Lpre_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* ST_T_M_Lpre_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* ST_T_M_Lpre_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* X2_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* X2_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* x_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* x_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* fft_b_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* fft_b_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* resS_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* resS_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* LPS_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* LPS_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp_LPS_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp_LPS_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* LPSa_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* LPSa_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* b_x1_real = (double*)malloc(nx * ny * sizeof(double));
	double* b_x1_imag = (double*)malloc(nx * ny * sizeof(double));
	double* a_b1_x1_real = (double*)malloc(nx * ny * sizeof(double));
	double* a_b1_x1_imag = (double*)malloc(nx * ny * sizeof(double));
	double* X1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* X1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* x1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* x1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* Xk_real = (double*)malloc(nx * ny * sizeof(double));
	double* Xk_imag = (double*)malloc(nx * ny * sizeof(double));
	double* k_real = (double*)malloc(nx * sizeof(double));
	double* k_imag = (double*)malloc(nx * sizeof(double));
	double* out_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* out_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* resk_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* resk_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* b_real = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* b_imag = (double*)malloc(nx * ny * nt * nc * sizeof(double));
	double* b1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* b1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* x_array_1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* conj_a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* conj_a_b1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* x_array_a_b1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* abs_a_b1_real = (double*)malloc(nx * ny * nc * sizeof(double));
	double* abs_a_b1_imag = (double*)malloc(nx * ny * nc * sizeof(double));
	double* res_2_real = (double*)malloc(nx * ny * nc * nt * sizeof(double));
	double* res_2_imag = (double*)malloc(nx * ny * nc * nt * sizeof(double));
	double* E_resk_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* E_resk_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* E_resk_rshp_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* E_resk_rshp_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* M_M0_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* M_M0_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* yt_real = (double*)malloc(nt * nt * sizeof(double));
	double* yt_imag = (double*)malloc(nt * nt * sizeof(double));
	double* y_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y1_real = (double*)malloc(nx * ny * nt * sizeof(double));
	double* y1_imag = (double*)malloc(nx * ny * nt * sizeof(double));
	double* temp_V_real = (double*)malloc(nt * nt * sizeof(double));
	double* temp_V_imag = (double*)malloc(nt * nt * sizeof(double));
	double* St_Vt_real = (double*)malloc(nx * ny * K * sizeof(double));
	double* St_Vt_imag = (double*)malloc(nx * ny * K * sizeof(double));
	double* xn_real = (double*)malloc(nt * sizeof(double));
	double* xn_imag = (double*)malloc(nt * sizeof(double));
	double* n_real = (double*)malloc(nt * sizeof(double));
	double* n_imag = (double*)malloc(nt * sizeof(double));
	double* nk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	double* nk_imag = (double*)malloc(nx * nx * nx * sizeof(double));
	double* WNnk_real = (double*)malloc(nx * nx * nx * sizeof(double));
	double* WNnk_imag = (double*)malloc(nx * nx * nx * sizeof(double));
	double* k0_real = (double*)malloc(nt * sizeof(double));
	double* k0_imag = (double*)malloc(nt * sizeof(double));
	double WN_real, WN_imag;
	// cSVD Memory allocation//////////////////////////////////////////
	double* TRTM_real = (double*)malloc(nt * kn * num * sizeof(double));
	double* RTM_real = (double*)malloc(nt * kn * sizeof(double));
	double* Y_real = (double*)malloc(nx * ny * kn * sizeof(double));
	double* Y_imag = (double*)malloc(nx * ny * kn * sizeof(double));
	// Transpose 
	double* YT_real = (double*)malloc(kn * nx * ny * sizeof(double));
	double* YT_imag = (double*)malloc(kn * nx * ny * sizeof(double));
	// RIMP Processing 
	double* B_real = (double*)malloc(kn * kn * sizeof(double));
	double* B_imag = (double*)malloc(kn * kn * sizeof(double));
	double* BT_real = (double*)malloc(kn * kn * sizeof(double));
	double* BT_imag = (double*)malloc(kn * kn * sizeof(double));
	double* BA_real = (double*)malloc(kn * kn * sizeof(double));
	double* BA_imag = (double*)malloc(kn * kn * sizeof(double));
	double* BR_real = (double*)malloc(kn * kn * sizeof(double));
	double* BR_imag = (double*)malloc(kn * kn * sizeof(double));
	// Diagonal
	double* d_d = (double*)malloc(kn * sizeof(double));
	double* t1_real = (double*)malloc(1 * sizeof(double));
	double* t1_imag = (double*)malloc(1 * sizeof(double));
	double* sum_real = (double*)malloc(1 * sizeof(double));
	double* sum_imag = (double*)malloc(1 * sizeof(double));
	double* Z_real = (double*)malloc(nx * ny * kn * sizeof(double));
	double* Z_imag = (double*)malloc(nx * ny * kn * sizeof(double));
	double* ZT_real = (double*)malloc(kn * nx * ny * sizeof(double));
	double* ZT_imag = (double*)malloc(kn * nx * ny * sizeof(double));
	double* S_inv = (double*)malloc(kn * kn * sizeof(double));
	double* T_real = (double*)malloc(nt * kn * sizeof(double));
	double* T_imag = (double*)malloc(nt * kn * sizeof(double));
	double* Us_real = (double*)malloc(kn * nx * ny * sizeof(double));
	double* Us_imag = (double*)malloc(kn * nx * ny * sizeof(double));
	double* Utr_real = (double*)malloc(kn * K * sizeof(double));
	double* Utr_imag = (double*)malloc(kn * K * sizeof(double));
	double* Ut_real = (double*)malloc(nx * ny * K * sizeof(double));
	double* Ut_imag = (double*)malloc(nx * ny * K * sizeof(double));
	double* Sd_real = (double*)malloc(kn * kn * sizeof(double));
	double* Str_real = (double*)malloc(K * K * sizeof(double));
	double* Vtr_real = (double*)malloc(nt * K * sizeof(double));
	double* Vtr_imag = (double*)malloc(nt * K * sizeof(double));
	double* St = (double*)malloc(K * K * sizeof(double));
	double* YU_real = (double*)malloc(nx * ny * kn * sizeof(double));
	double* YU_imag = (double*)malloc(nx * ny * kn * sizeof(double));
	// cSVD memeory allocation Done 
	/********************************************Allocating memory where result will be stored****************************************************************/
	//double* L_r = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* L_i = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* S_r = (double*)malloc(nx * ny * nt * sizeof(double));
	//double* S_i = (double*)malloc(nx * ny * nt * sizeof(double));
	/********************************************Reading Data from the files**********************************************************************************/
	cout << "\t\t Opening File to Read .......................................................\n";
	ifstream M_read_real("M_real.txt");
	ifstream M_read_imag("M_imag.txt");
	ifstream a_b1_read_real("a_b1_real.txt");
	ifstream a_b1_read_imag("a_b1_imag.txt");
	ifstream d_read_real("d_real.txt");
	ifstream d_read_imag("d_imag.txt");
	ifstream a_mask_read_real("a_mask_real.txt");
	ifstream a_mask_read_imag("a_mask_imag.txt");
	ifstream TRTM_read_real("TRTM_real.txt");
	/****************************************Checking if all the files available in root directory or not****************************************************/
	if (!M_read_real || !M_read_imag || !a_b1_read_real || !a_b1_read_imag || !d_read_real || !d_read_imag || !a_mask_read_real || !a_mask_read_imag || !TRTM_read_real)
	{
		cout << "Cannot open text files \n";
		(void)getchar();
		return;
	}
	else
		cout << "All files found:- \t Reading Data In Progress \n";
	/****************************************Files reading started ******************************************************************************************/
	cout << "\t\t Files Reading Started ......................................................\n";
	double file_in = 0.0;
	cout << "\t\t Reading Input Data .........................................................\n";
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		d_read_real >> file_in;
		*(d_real + i) = file_in;
		d_read_imag >> *(d_imag + i);
	}

	double sum_d = 0;
	for (int i = 0; i < (nx * ny * nt * nc); i++)
	{
		sum_d = sum_d + pow(*(d_real + i), 2) + pow(*(d_imag + i), 2);
	}
	double norm_d = sqrt(sum_d);
	cout << "\t\t norm_d of input kdata is = " << norm_d << endl;

	cout << "\t\t Reading Initial a_b1........................................................\n";

	for (int i = 0; i < (nx * ny * nc); i++)
	{
		a_b1_read_real >> file_in;
		*(a_b1_real + i) = file_in;
		a_b1_read_imag >> *(a_b1_imag + i);
	}

	double sum_a_b1 = 0;
	for (int i = 0; i < nx * ny * nc; i++)
	{
		sum_a_b1 = sum_a_b1 + pow(*(a_b1_real + i), 2) + pow(*(a_b1_imag + i), 2);
	}
	double norm_a_b1 = sqrt(sum_a_b1);
	cout << "\t\t norm_a_b1 of input b1 is = " << norm_a_b1 << endl;


	cout << "\t\t Reading Initial M ..........................................................\n";

	for (int i = 0; i < (nx * ny * nt); i++)
	{
		M_read_real >> file_in;
		*(M_real + i) = file_in;
		M_read_imag >> *(M_imag + i);
	}
	double sum_M = 0;
	for (int i = 0; i < nx * ny * nt; i++)
	{
		sum_M = sum_M + pow(*(M_real + i), 2) + pow(*(M_imag + i), 2);
	}
	double norm_M = sqrt(sum_M);
	cout << "\t\t norm_M of input M is = " << norm_M << endl;

	cout << "\t\t Reading Initial TRTM ........................................................\n";

	for (int i = 0; i < (nt * kn * num); i++)
	{
		TRTM_read_real >> file_in;
		*(TRTM_real + i) = file_in;
	}
	// Display in array all the values 
	/*void readFile();
	{
		std::ifstream file("TRTM_real.txt");
		for (std::string word; file >> word; )
			std::cout << word << '\n';
	}
	*/
	// calculate Norm of TRTM
	double sum_TRTM = 0;
	for (int i = 0; i < nt * kn * num; i++)
	{
		sum_TRTM = sum_TRTM + pow(*(TRTM_real + i), 2);
	}
	double norm_TRTM = sqrt(sum_TRTM);
	cout << "\t\t norm_TRTM of input TRTM is = " << norm_TRTM << endl;

	cout << "\t\t Reading Sampling Mask........................................................\n";

	for (int i = 0; i < (nx * ny * nt); i++)
	{
		a_mask_read_real >> file_in;
		*(a_mask_real + i) = file_in;
		a_mask_read_imag >> *(a_mask_imag + i);
	}
	double sum_a_mask = 0;
	for (int i = 0; i < nx * ny * nt; i++)
	{
		sum_a_mask = sum_a_mask + pow(*(a_mask_real + i), 2) + pow(*(a_mask_imag + i), 2);
	}
	double norm_a_mask = sqrt(sum_a_mask);
	cout << "\t\t norm_a_mask of input mask is = " << norm_a_mask << endl;
	cout << "\t\t Reading Input Data Done .....................................................\n";
	// M (nx*ny, nt)
	// Initialize S=zeros(nx*ny,nt) and Lpre=M
	for (int i = 0; i < nx * ny * nt; i++)
	{
		*(S_real + i) = 0;
		*(S_imag + i) = 0;
		*(Lpre_real + i) = *(M_real + i);
		*(Lpre_imag + i) = *(M_imag + i);
	}
	int ite = 0;
	cout << "\t\t Start Iteration for L+S Reconstruction ......................................\n";
	/***************************************Check the data in the text file *************************************************************************************/
	/*void readFile();
	{
		std::ifstream file("M_real.txt");
		for (std::string word; file >> word; )
			std::cout << word << '\n';
	}
	*/
	/***************************************Algorithm from While loop starts from here ***** *********************************************************************/
	/***************************************Algorithm from While loop starts from here ***** *********************************************************************/
	/***************************************Algorithm from While loop starts from here ***** *********************************************************************/
	//display_3d_matrix_mem_complex(M_real, M_imag, nx,ny, nt);
	while (ite < 1)
	{
		ite = ite + 1;
		// Logic= TRTM_real(((ite - 1) * inc) + 1 :((ite - 1) * inc) + inc)

		for (int i = 0; i < inc; i++)
		{
			*(RTM_real + i) = *(TRTM_real + (ite - 1) * inc + i);
		}
		//   Selected string from RTM
		//  display_matrix_mem_real(RTM_real, kn, nt);

		for (int i = 0; i < (nx * ny * nt); i++)
		{
			*(M0_real + i) = *(M_real + i);
			*(M0_imag + i) = *(M_imag + i);
		}
		// Compute Total Time for L+S Algorithm
		int start_sstart = clock();
		// subtraction (M-S)
		int start_ss0 = clock();
		for (int i = 0; i < (nx * ny); i++)
		{
			for (int j = 0; j < nt; j++)
			{
				*(M_S_real + i * nt + j) = *(M_real + i * nt + j) - *(S_real + i * nt + j);
				*(M_S_imag + i * nt + j) = *(M_imag + i * nt + j) - *(S_imag + i * nt + j);
			}
		}
		int stop_ss0 = clock();
		cout << "\n CPU_ Low_rank_Extraction:" << (stop_ss0 - start_ss0) << endl;
		//display_matrix_mem_real(M_S_real, nx*ny, nt);
		//display_3d_matrix_mem_complex(M_S_real, M_S_imag, nx, ny,nt);
	/********************************************cSVD Start  *********************************************************************************************************/
		int start_s = clock();
		// Multiplication (Low rank* RTM) nx * ny, kn
		int start_s1 = clock();
		for (int i = 0; i < (nx * ny); i++)
		{
			for (int j = 0; j < kn; j++)
			{
				*(sum_real) = 0;
				*(sum_imag) = 0;
				for (int k = 0; k < nt; k++)
				{
					*(sum_real) = *(sum_real)+*(M_S_real + i * nt + k) * *(RTM_real + k * kn + j);
					*(sum_imag) = *(sum_imag)+*(M_S_imag + i * nt + k) * *(RTM_real + k * kn + j);
				}
				*(Y_real + i * kn + j) = *(sum_real);
				*(Y_imag + i * kn + j) = *(sum_imag);
			}
		}
		int stop_s1 = clock();
		cout << "\n CPU_cSVD Computation SIM:" << (stop_s1 - start_s1) << endl;
		// display_matrix_mem_complex(Y_real, Y_imag, kn, nx*ny);
		// Transpose YT=Y'
		int start_s2 = clock();
		for (int i = 0; i < (nx * ny); i++)
		{
			for (int j = 0; j < kn; j++)
			{
				*(YT_real + j * nx * ny + i) = *(Y_real + i * kn + j);
				*(YT_imag + j * nx * ny + i) = -1 * *(Y_imag + i * kn + j);
			}
		}
		int stop_s2 = clock();
		cout << "\n CPU_cSVD Computation Transpose_01:" << (stop_s2 - start_s2) << endl;
		// Multiplication B = Y'*Y;
		int start_s3 = clock();
		for (int i = 0; i < kn; i++)
		{
			for (int j = 0; j < kn; j++)
			{
				*(t1_real) = 0;
				*(t1_imag) = 0;
				*(sum_real) = 0;
				*(sum_imag) = 0;
				double x1, y1, u1, v2;
				for (int k = 0; k < nx * ny; k++)
				{
					x1 = *(YT_real + i * nx * ny + k);
					y1 = *(YT_imag + i * nx * ny + k);
					u1 = *(Y_real + k * kn + j);
					v2 = *(Y_imag + k * kn + j);
					*(t1_real) = x1 * u1 - y1 * v2;
					*(t1_imag) = x1 * v2 + y1 * u1;
					*(sum_real) = *(sum_real)+*(t1_real);
					*(sum_imag) = *(sum_imag)+*(t1_imag);
				}
				*(B_real + i * kn + j) = *(sum_real);
				*(B_imag + i * kn + j) = *(sum_imag);
			}
		}
		int stop_s3 = clock();
		cout << "\n CPU_cSVD Computation CM:" << (stop_s3 - start_s3) << endl;
		// Addition B = 0.5*(B+B');
		// Transpose
		int start_se = clock();
		for (int i = 0; i < kn; i++)
		{
			for (int j = 0; j < kn; j++)
			{
				*(BT_real + j * kn + i) = *(B_real + i * kn + j);
				*(BT_imag + j * kn + i) = -1 * *(B_imag + i * kn + j);
			}
		}
		// Addition
		for (int i = 0; i < (kn * kn); i++)
		{
			*(BA_real + i) = *(B_real + i) + *(BT_real + i);
			*(BA_imag + i) = *(B_imag + i) + *(BT_imag + i);
		}
		// display_matrix_mem_complex(BA_real, BA_imag, kn, kn);
		// constant multiplier
		for (int i = 0; i < (kn * kn); i++)
		{
			*(BR_real + i) = 0.5 * *(BA_real + i);
			*(BR_imag + i) = 0.5 * *(BA_imag + i);
		}
		int stop_se = clock();
		cout << "\n CPU_cSVD Computation Esym:" << (stop_se - start_se) << endl;
		//  Eigen_value_Decomposition [V, D]= eig(B)
		cout << "Matrix B must ensure RIP for Eigen Value Decompositon of Dim= Kn*Kn is: \n";
		// display_matrix_mem_complex(BR_real, BR_imag, kn, kn);
		cout << "\t Starting ESVD Block with tolerance ..........................................( " << tol << ") \n";
		int start_s4 = clock();
		SVD_Ejacobi(BR_real, BR_imag, UE_real, UE_imag, D_real, VE_real, VE_imag, kn, kn, tol);
		int stop_s4 = clock();
		cout << "\n CPU_cSVD Computation Eig Dec:" << (stop_s4 - start_s4) << endl;
		cout << "\t ESVD Done .............................................................................\n";
		cout << "Diagonal Matrix D_real after Eigen Value Decompositon of Dim= Kn*Kn is: \n";
		// display_matrix_mem_real(D_real, kn, kn);
		// cout << "Matrix V left Eigen Vectors of Dim= nt*Kn is: \n";
		// display_matrix_mem_complex(VE_real, VE_imag, nt, kn);   
		 //	   % Diagonal d = diag(D);
		int start_sp = clock();
		for (int i = 0; i < kn; i++)
		{
			for (int j = 0; j < kn; j++)
			{
				if (i == j)
					*(d_d + i) = *(D_real + j * kn + i);
			}
		}
		//e= sqrt(abs(d));
		double* e_real = (double*)malloc(kn * sizeof(double));
		for (int i = 0; i < kn; i++)
		{
			if (*(d_d + i) < 0)
				*(e_real + i) = -1 * pow(*(d_d + i), 0.5);
			else
			{
				*(e_real + i) = pow(*(d_d + i), 0.5);
			}
		}
		// S = spdiags(e, 0, kn, kn);%11x11 Sparse Matrix generation
		for (int i = 0; i < kn; i++)
		{
			for (int j = 0; j < kn; j++)
			{
				if (i == j)
					*(Sd_real + j * kn + i) = *(e_real + i);
				else
				{
					*(Sd_real + j * kn + i) = 0;
				}
			}
		}
		int stop_sp = clock();
		cout << "\n CPU_cSVD Computation Rescale Eig Values:" << (stop_sp - start_sp) << endl;
		//display_matrix_mem_real(Sd_real, kn, kn);
		//U= S\(Y*V)'; Multiplication 
		//[Z_real,Z_imag]=my_multiplication(Y_real,Y_imag,VE_real,VE_imag,nx*ny,kn,kn);
		int start_s5 = clock();
		for (int i = 0; i < nx * ny; i++)
		{
			for (int j = 0; j < kn; j++)
			{
				*(t1_real) = 0;
				*(t1_imag) = 0;
				*(sum_real) = 0;
				*(sum_imag) = 0;
				double x1, y1, u1, v2;
				for (int k = 0; k < kn; k++)
				{
					x1 = *(Y_real + i * kn + k);
					y1 = *(Y_imag + i * kn + k);
					u1 = *(VE_real + k * kn + j);
					v2 = *(VE_imag + k * kn + j);
					*(t1_real) = x1 * u1 - y1 * v2;
					*(t1_imag) = x1 * v2 + y1 * u1;
					*(sum_real) = *(sum_real)+*(t1_real);
					*(sum_imag) = *(sum_imag)+*(t1_imag);
				}
				*(Z_real + i * kn + j) = *(sum_real);
				*(Z_imag + i * kn + j) = *(sum_imag);
			}
		}
		int stop_s5 = clock();
		cout << "\n CPU_cSVD Computation approx Right singular values:" << (stop_s5 - start_s5) << endl;
		// Transpose [ZT_real,ZT_imag]=my_transpose(Z_real,Z_imag,nx*ny,kn);
		int start_s6 = clock();
		for (int i = 0; i < nx * ny; i++)
		{
			for (int j = 0; j < kn; j++)
			{
				*(ZT_real + j * nx * ny + i) = *(Z_real + i * kn + j);
				*(ZT_imag + j * nx * ny + i) = -1 * *(Z_imag + i * kn + j);
			}
		}
		int stop_s6 = clock();
		cout << "\n CPU_cSVD Computation Transpose_2:" << (stop_s6 - start_s6) << endl;
		//Inverse of S
		//S_inv=my_inverse(S_real,kn);
		int start_si = clock();
		for (int i = 0; i < kn; i++)
			for (int j = 0; j < kn; j++)
			{
				if (i == j)
					*(S_inv + i * kn + j) = 1 / *(Sd_real + i * kn + j);
				else
				{
					*(S_inv + i * kn + j) = 0;
				}
			}
		int stop_si = clock();
		cout << "\n CPU_cSVD Computation S_inv:" << (stop_si - start_si) << endl;

		// display_matrix_mem_real(S_inv, kn, kn);
	   //  Multiplication : [U_real,U_imag]=my_crmultiplication(S_inv,ZT_real,ZT_imag,kn,kn,nx*ny);//
		int start_s7 = clock();
		for (int i = 0; i < kn; i++)
		{
			for (int j = 0; j < nx * ny; j++)
			{
				*(sum_real) = 0;
				*(sum_imag) = 0;
				for (int k = 0; k < kn; k++)
				{
					*(sum_real) = *(sum_real)+*(S_inv + i * kn + k) * *(ZT_real + k * nx * ny + j);
					*(sum_imag) = *(sum_imag)+*(S_inv + i * kn + k) * *(ZT_imag + k * nx * ny + j);
				}
				*(Us_real + i * nx * ny + j) = *(sum_real);
				*(Us_imag + i * nx * ny + j) = *(sum_imag);
			}
		}
		int stop_s7 = clock();
		cout << "\n CPU_cSVD Computation Approx right Singular values:" << (stop_s7 - start_s7) << endl;
		// Transpose : Y=U';% 65536x11
		int start_s8 = clock();
		for (int i = 0; i < kn; i++)
		{
			for (int j = 0; j < nx * ny; j++)
			{
				*(YU_real + j * kn + i) = *(Us_real + i * nx * ny + j);
				*(YU_imag + j * kn + i) = -1 * *(Us_imag + i * nx * ny + j);
			}
		}
		int stop_s8 = clock();
		cout << "\n CPU_cSVD Computation Transpose_03:" << (stop_s8 - start_s8) << endl;
		// Multiplication : T=X'*Y; % 2nd overpass of the data matrix
		//[M_ST_real,M_ST_imag] = my_transpose(M_S_real,M_S_imag,nx*ny,nt);
		// Transpose 
		int start_s9 = clock();
		for (int i = 0; i < nx * ny; i++)
		{
			for (int j = 0; j < nt; j++)
			{
				*(M_ST_real + j * nx * ny + i) = *(M_S_real + i * nt + j);
				*(M_ST_imag + j * nx * ny + i) = -1 * *(M_S_imag + i * nt + j);
			}
		}
		int stop_s9 = clock();
		cout << "\n CPU_cSVD Computation Transpose_04:" << (stop_s9 - start_s9) << endl;
		//Multiplicaion
		// [T_real,T_imag] = my_multiplication(M_ST_real,M_ST_imag,YU_real,YU_imag,nt,nx*ny,kn);
		int start_s10 = clock();
		for (int i = 0; i < nt; i++)
		{
			for (int j = 0; j < kn; j++)
			{
				*(t1_real) = 0;
				*(t1_imag) = 0;
				*(sum_real) = 0;
				*(sum_imag) = 0;
				double x1, y1, u1, v2;
				for (int k = 0; k < nx * ny; k++)
				{
					x1 = *(M_ST_real + i * nx * ny + k);
					y1 = *(M_ST_imag + i * nx * ny + k);
					u1 = *(YU_real + k * kn + j);
					v2 = *(YU_imag + k * kn + j);
					*(t1_real) = x1 * u1 - y1 * v2;
					*(t1_imag) = x1 * v2 + y1 * u1;
					*(sum_real) = *(sum_real)+*(t1_real);
					*(sum_imag) = *(sum_imag)+*(t1_imag);
				}
				*(T_real + i * kn + j) = *(sum_real);
				*(T_imag + i * kn + j) = *(sum_imag);
			}
		}
		int stop_s10 = clock();
		cout << "\n CPU_cSVD Computation Multiplicaton:" << (stop_s10 - start_s10) << endl;
		//SVD :[V_real,V_imag,S_real,U_real,U_imag]=my_jacobi_svd(T_real,T_imag,nt,kn,tol);//
		cout << "\t Starting SVD Block with tolerance ............................................( " << tol << ") \n";
		int start_s11 = clock();
		SVD_jacobi(T_real, T_imag, V_real, V_imag, Sv_real, U_real, U_imag, nt, kn, tol);
		cout << "\t SVD Done ................................................................................\n";
		int stop_s11 = clock();
		cout << "\n CPU_cSVD Computation SVD:" << (stop_s11 - start_s11) << endl;
		// cout << "Matrix V after SVD of Dim= nt*Kn is: \n";
		// display_matrix_mem_complex(V_real, V_imag, nt, kn);
		cout << "Sv_real after SVD is:";
		// display_matrix_mem_real(Sv_real, kn, kn);
		// Truncate : Ut = Y*U(:, 1:k);//
		//[Utr_real,Utr_imag]=my_trun(U_real,U_imag,kn,k);
		int start_s12 = clock();
		for (int i = 0; i < kn; i++)
		{
			for (int j = 0; j < K; j++)
			{
				*(Utr_real + i * K + j) = *(U_real + i * K + j);
				*(Utr_imag + i * K + j) = *(U_imag + i * K + j);
			}
		}
		// Multiplication [Ut_real,Ut_imag]=my_multiplication(YU_real,YU_imag,Utr_real,Utr_imag,nx*ny,kn,k);
		//int start_s12 = clock();
		for (int i = 0; i < nx * ny; i++)
		{
			for (int j = 0; j < K; j++)
			{
				*(t1_real) = 0;
				*(t1_imag) = 0;
				*(sum_real) = 0;
				*(sum_imag) = 0;
				double x1, y1, u1, v2;
				for (int L = 0; L < kn; L++)
				{
					x1 = *(YU_real + i * kn + L);
					y1 = *(YU_imag + i * kn + L);
					u1 = *(Utr_real + L * K + j);
					v2 = *(Utr_imag + L * K + j);
					*(t1_real) = x1 * u1 - y1 * v2;
					*(t1_imag) = x1 * v2 + y1 * u1;
					*(sum_real) = *(sum_real)+*(t1_real);
					*(sum_imag) = *(sum_imag)+*(t1_imag);
				}
				*(Ut_real + i * K + j) = *(sum_real);
				*(Ut_imag + i * K + j) = *(sum_imag);
			}
		}
		int stop_s12 = clock();
		cout << "\n CPU_cSVD Computation multiplication:" << (stop_s12 - start_s12) << endl;
		// St truncation Str_real=my_Strunc(S_real,kn,k);
		//St = Sv_real(1:k, 1:k);
		int start_st = clock();
		for (int i = 0; i < K; i++)
		{
			for (int j = 0; j < K; j++)
			{
				*(Str_real + i * K + j) = *(Sv_real + i * kn + j); //
			}
		}
		cout << "Str_real after truncated according to target rank K is:";
		int stop_st = clock();
		cout << "\n CPU_cSVD Computation update singular values:" << (stop_st - start_st) << endl;
		// display_matrix_mem_real(Str_real, K, K);
		// Vt truncation :Vt = V(:, 1:k);
		// [Vtr_real,Vtr_imag]=my_trun(V_real,V_imag,nt,k); % Vt
		int start_sv = clock();
		for (int i = 0; i < nt; i++)
		{
			for (int j = 0; j < K; j++)
			{
				*(Vtr_real + i * K + j) = *(V_real + i * K + j);
				*(Vtr_imag + i * K + j) = *(V_imag + i * K + j);
			}
		}
		int stop_sv = clock();
		cout << "\n CPU_cSVD Computation update right singular values:" << (stop_sv - start_sv) << endl;
		//display_matrix_mem_complex(Vtr_real, Vtr_imag, nt, K);
			// Diagonal Value into single row vector 
		int start_ss = clock();
		double* Str_c = (double*)malloc(K * sizeof(double));
		for (int i = 0; i < K; i++)
		{
			for (int j = 0; j < K; j++)
			{
				if (i == j)
					*(Str_c + i) = *(Str_real + j * K + i);
			}
		}
		int stop_ss = clock();
		cout << "\n CPU_cSVD Computation update singular values:" << (stop_ss - start_ss) << endl;
		//e= sqrt(abs(d));
		int stop_s = clock();
		cout << "\n CPU_cSVD Computation Time:" << (stop_s - start_s) << endl;
		/********************************************cSVD_ End   *************************************************************************************************************/
		cout << "Starting Thresholding of Low Rank Components \n";
		/********************************************Soft Thresholding   ****************************************************************************************************/
		int start_sthr = clock();
		double max_St;

		max_St = *(Str_c + 0);

		p = max_St * lambda_L;


		double* temp = (double*)malloc(K * sizeof(double));
		double* out = (double*)malloc(K * sizeof(double));
		double max;

		for (int i = 0; i < K; i++)
		{
			abs_in = *(Str_c + i);
			if (abs_in == 0)
				*(temp + i) = 0;
			else
				*(temp + i) = *(Str_c + i) / abs_in;
			if (abs_in - p > 0)
				max = (abs_in - p);
			else
				max = 0;
			*(out + i) = *(temp + i) * max;

		}
		int stop_sthr = clock();
		cout << "\n CPU_Thresholding of low rank Computation Time:" << (stop_sthr - start_sthr) << endl;
		//St_real=my_spdiagonal(IST,k); % St after Thresholding Diagonalization
		/*for (int i = 0; i < K; i++)
		{
			*(St + i * K + i) = *(out + i);
		}
		*/

		// Place values on Diagonals
		for (int i = 0; i < K; i++)
		{
			for (int j = 0; j < K; j++)
			{
				if (i == j)
					*(St + j * K + i) = *(out + i);
				else
				{
					*(St + j * K + i) = 0;
				}
			}
		}
		cout << "St (Diagonal Values) after thresholding is: ";
		display_matrix_mem_real(St, K, K);

		/********************************************Reconstruct Matrix for low rank  *********************************************************************************************/
		int start_srec = clock();
		for (int i = 0; i < nt; i++)
		{
			for (int j = 0; j < K; j++)
			{
				*(temp_V_real + i * K + j) = *(Vtr_real + j * nt + i);
				*(temp_V_imag + i * K + j) = -1 * *(Vtr_imag + j * nt + i);
			}
		}

		for (int i = 0; i < nx * ny; i++)
		{
			for (int j = 0; j < K; j++)
			{
				*(sum_real) = 0; *(sum_imag) = 0;
				for (int k = 0; k < K; k++)
				{
					*(sum_real) = *(sum_real)+*(Ut_real + i * K + k) * *(St + k * K + j);
					*(sum_imag) = *(sum_imag)+*(Ut_imag + i * K + k) * *(St + k * K + j);
				}
				*(St_Vt_real + i * K + j) = *(sum_real);
				*(St_Vt_imag + i * K + j) = *(sum_imag);
			}
		}

		for (int i = 0; i < nx * ny; i++)
		{
			for (int j = 0; j < nt; j++)
			{
				*(sum_real) = 0;	*(sum_imag) = 0;
				for (int k = 0; k < K; k++)
				{
					*(sum_real) = *(sum_real)+(*(St_Vt_real + i * K + k) * *(temp_V_real + k * nt + j)) - (*(St_Vt_imag + i * K + k) * *(temp_V_imag + k * nt + j));
					*(sum_imag) = *(sum_imag)+(*(St_Vt_imag + i * K + k) * *(temp_V_imag + k * nt + j)) + (*(St_Vt_imag + i * K + k) * *(temp_V_real + k * nt + j));
				}
				*(L_real + i * nt + j) = *(sum_real);
				*(L_imag + i * nt + j) = *(sum_imag);
			}
		}
		cout << "L matrix After Thresholding, Done \n";
		int stop_srec = clock();
		cout << "\n CPU_low rank recovery:" << (stop_srec - start_srec) << endl;
		// cout << "L Matrix after thresholding  is:";
		// display_matrix_mem_complex (L_real, L_imag, nx*ny, nt);
		//	cout <<"M at Line 376 is : ";	
		//	display_matrix_mem_complex (M_real, M_imag, nx*ny, nt);
		//	cout <<"Lpre at Line 376 is : ";	
		//	display_matrix_mem_complex (Lpre_real, Lpre_imag, nx*ny, nt);
		double sumL = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumL = sumL + pow(*(L_real + i), 2) + pow(*(L_imag + i), 2);
		}
		double norm_L = sqrt(sumL);
		//cout << "\t\t norm_L at line 367 is = " << norm_L << endl;

		double sumLpre = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumLpre = sumLpre + pow(*(Lpre_real + i), 2) + pow(*(Lpre_imag + i), 2);
		}
		double norm_Lpre = sqrt(sumLpre);
		//cout << "\t\t norm_Lpre at line 367 is = " << norm_Lpre << endl;

		double sumM = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumM = sumM + pow(*(M_real + i), 2) + pow(*(M_imag + i), 2);
		}
		double norm_M = sqrt(sumM);
		//cout << "\t\t norm_M at line 367 is = " << norm_M << endl;

	/********************************************Extract Sparse components*******************************************************************************************/
		int start_sparse = clock();
		for (int i = 0; i < nx * ny; i++)
		{
			for (int j = 0; j < nt; j++)
			{
				*(M_Lpre_real + i * nt + j) = *(M_real + i * nt + j) - *(Lpre_real + i * nt + j);
				*(M_Lpre_imag + i * nt + j) = *(M_imag + i * nt + j) - *(Lpre_imag + i * nt + j);
			}
		}
		int stop_sparse = clock();
		cout << "\n Extract Sparse components:" << (stop_sparse - start_sparse) << endl;
		double sumM_Lpre = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumM_Lpre = sumM_Lpre + pow(*(M_Lpre_real + i), 2) + pow(*(M_Lpre_imag + i), 2);
		}
		double norm_M_Lpre = sqrt(sumM_Lpre);
		//cout << "\t\t norm_M_Lpre at line 367 is = " << norm_M_Lpre << endl;

		//	cout <<"M_Lpre at Line 376 is : ";	
		//	display_matrix_mem_complex (M_Lpre_real, M_Lpre_imag, nx*ny, nt);
	/********************************************Reeshape 2D to 3D *****************************************************************************************************/
		// Reshape 2D to 3D 
		for (int i = 0; i < ny; i++)
		{
			for (int j = 0; j < nx; j++)
			{
				for (int k = 0; k < nt; k++)
				{
					*(x_real + nx * ny * k + ny * j + i) = *(M_Lpre_real + nt * j + i * nx * nt + k);
					*(x_imag + nx * ny * k + ny * j + i) = *(M_Lpre_imag + nt * j + i * nx * nt + k);
				}
			}
		}
		double sumx = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumx = sumx + pow(*(x_real + i), 2) + pow(*(x_imag + i), 2);
		}

		double norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at line 380 is = " << norm_x << endl;

		//	cout << "x at Line 1037 is:";
		//	display_3d_matrix_mem_complex (x_real, x_imag, nx, ny, nt);
	/********************************************Temp FFT   ************************************************************************************************************/
		// int start_tempfft = clock();
		for (int i = 0; i < nt; i++)
		{
			*(yt_real + i) = 0;
			*(yt_imag + i) = 0;
		}

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(y_real + i) = 0;
			*(y_imag + i) = 0;
		}

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(y1_real + i) = *(y_real + i);
			*(y1_imag + i) = *(y_imag + i);
		}

		int start_tempfft = clock();
		for (int c = 0; c < ny; c++)
		{
			for (int r = 0; r < nx; r++)
			{
				for (int i = 0; i < nt; i++)
				{
					*(xn_real + i) = *(x_real + i * nx * ny + r * nx + c);
					*(xn_imag + i) = *(x_imag + i * nx * ny + r * nx + c);
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

				double mag = sqrt(pow(WN_real, 2) + pow(WN_imag, 2));
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
						*(y_real + i * nx * ny + r * nx + c) = sum0_real;
						*(y_imag + i * nx * ny + r * nx + c) = sum0_imag;
					}
				}
			}
		}
		int stop_tempfft = clock();
		cout << "\n CPU_Temporal FFT:" << (stop_tempfft - start_tempfft) << endl;
		//	display_3d_matrix_mem_complex (y_real, y_imag, nx, ny, nt);
		sumx = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumx = sumx + pow(*(y_real + i), 2) + pow(*(y_imag + i), 2);
		}

		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at line 435 is = " << norm_x << endl;


		int dim = 3, K = nt / 2, len = nt;

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
			int l = 0;
			for (int i = 0; i < nx * ny * nt; i++)
			{
				if (i < nx * ny * (nt - 1))
				{
					*(temp1_real + i) = *(y_real + i);
					*(temp1_imag + i) = *(y_imag + i);
				}
				else
				{
					*(temp2_real + l) = *(y_real + i);
					*(temp2_imag + l) = *(y_imag + i);
					l = l + 1;
				}
			}

			//	display_3d_matrix_mem_complex (temp1_real, temp1_imag, nx, ny, (nt-1));

			for (int i = 0; i < nx * ny * nt; i++)
			{
				if (i < nx * ny)
				{
					*(y_real + i) = *(temp2_real + i);
					*(y_imag + i) = *(temp2_imag + i);
				}
				else
				{
					//				* (X_real + nx*ny + i) = *(temp1_real - nx*ny + i);
					//				* (X_imag + nx*ny + i) = *(temp1_imag - nx*ny + i);
					*(y_real + i) = *(temp1_real - nx * ny + i);
					*(y_imag + i) = *(temp1_imag - nx * ny + i);

				}
			}
		}

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(shft_fft_b_real + i) = *(y_real + i);
			*(shft_fft_b_imag + i) = *(y_imag + i);
		}

		//	display_3d_matrix_mem_complex (shft_fft_b_real, shft_fft_b_imag, nx, ny, nt);
		double sumfft_b = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumfft_b = sumfft_b + pow(*(shft_fft_b_real + i), 2) + pow(*(shft_fft_b_imag + i), 2);
		}

		double norm_fft_b = sqrt(sumfft_b);
		//cout << "\t\t norm_fft_b at line 455 is = " << norm_fft_b << endl;

		double sqrt_nt;
		sqrt_nt = sqrt(nt_db);

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(T_M_Lpre_real + i) = *(shft_fft_b_real + i) / sqrt_nt;
			*(T_M_Lpre_imag + i) = *(shft_fft_b_imag + i) / sqrt_nt;
		}
		//	display_3d_matrix_mem_complex (T_M_Lpre_real, T_M_Lpre_imag, nx, ny, nt);
	/********************************************Soft Thresholding   *********************************************************************************************************/
		int start_softs = clock();
		p = lambda_S;
		double temp_real = 0, temp_imag = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			abs_in = sqrt(pow(*(T_M_Lpre_real + i), 2) + pow(*(T_M_Lpre_imag + i), 2));
			if (abs_in != 0)
			{
				temp_real = *(T_M_Lpre_real + i) / abs_in;
				temp_imag = *(T_M_Lpre_imag + i) / abs_in;
			}

			if (abs_in - p > 0)
				max = abs_in - p;
			else
				max = 0;

			*(ST_T_M_Lpre_real + i) = temp_real * max;
			*(ST_T_M_Lpre_imag + i) = temp_imag * max;
		}

		cout << "Soft Thresholding of Sparse Component Done\n ";
		int stop_softs = clock();
		cout << "\n Soft Thresh of Sparse components:" << (stop_softs - start_softs) << endl;
		//	display_3d_matrix_mem_complex (ST_T_M_Lpre_real, ST_T_M_Lpre_imag, nx, ny, nt);
		double sumST_T_M_Lpre = 0;

		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumST_T_M_Lpre = sumST_T_M_Lpre + pow(*(ST_T_M_Lpre_real + i), 2) + pow(*(ST_T_M_Lpre_imag + i), 2);
		}

		double norm_ST_T_M_Lpre = sqrt(sumST_T_M_Lpre);
		//cout << "\t\t norm_ST_T_M_Lpre at Line 488 is = " << norm_ST_T_M_Lpre;

		// Line 462


	/********************************************Inverse Temp FFT  *************************************************************************************************************/
		int a_adjoint = 1;
		dim = 3, K = nt / 2, len = nt;

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(X_real + i) = *(ST_T_M_Lpre_real + i);
			*(X_imag + i) = *(ST_T_M_Lpre_imag + i);
		}

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

		//	cout <<" X is: ";
		//	display_3d_matrix_mem_complex (X_real, X_imag, nx, ny, nt);

		for (int j = 0; j < K; j++)
		{
			int l = 0;
			for (int i = 0; i < nx * ny * nt; i++)
			{
				if (i < nx * ny * (nt - 1))
				{
					*(temp1_real + i) = *(X_real + i);
					*(temp1_imag + i) = *(X_imag + i);
				}
				else
				{
					*(temp2_real + l) = *(X_real + i);
					*(temp2_imag + l) = *(X_imag + i);
					l = l + 1;
				}
			}
			//		cout <<" temp1 is: ";
			//		display_3d_matrix_mem_complex (temp1_real, temp1_imag, nx, ny, (nt-1));
			//		cout <<" temp2 is: ";
			//		display_3d_matrix_mem_complex (temp2_real, temp2_imag, nx, ny, 1);


			for (int i = 0; i < nx * ny * nt; i++)
			{
				if (i < nx * ny)
				{
					*(X_real + i) = *(temp2_real + i);
					*(X_imag + i) = *(temp2_imag + i);
				}
				else
				{
					*(X_real + i) = *(temp1_real - nx * ny + i);
					*(X_imag + i) = *(temp1_imag - nx * ny + i);
				}
			}
			//		cout << " X is: ";
			//		display_3d_matrix_mem_complex (X_real, X_imag, nx, ny, nt);

		}
		//	cout << " x is: ";
		//	display_3d_matrix_mem_complex (X_real, X_imag, nx, ny, nt);

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(x_real + i) = *(X_real + i);
			*(x_imag + i) = -1 * *(X_imag + i);
		}

		//	display_3d_matrix_mem_complex (x_real, x_imag, nx, ny, nt);

		int start_itempfft = clock();

		// Line 502

		for (int c = 0; c < ny; c++)
		{
			for (int r = 0; r < nx; r++)
			{
				for (int i = 0; i < nt; i++)
				{
					*(xn_real + i) = *(x_real + i * nx * ny + r * nx + c);
					*(xn_imag + i) = *(x_imag + i * nx * ny + r * nx + c);
				}

				for (int i = 0; i < nt; i++)
				{
					*(n_real + i) = i;
					*(k0_real + i) = i;
					*(n_imag + i) = i;
					*(k0_imag + i) = i;
				}

				//				WN_real = 0.9877;		WN_imag = -0.1564;
				WN_real = cos(2 * PI / nt);	WN_imag = sin(-2 * PI / nt);
				if (WN_real < 0)
					sign_cos = -1;
				else
					sign_cos = 1;

				if (WN_imag < 0)
					sign_sin = -1;
				else
					sign_sin = 1;

				double mag = sqrt(pow(WN_real, 2) + pow(WN_imag, 2));
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
						//					*(nk_imag   + i*nt + j) = sum0_imag;
						if (*(nk_real + i * nt + j) == 0)
						{
							*(WNnk_real + i * nt + j) = 1;
							*(WNnk_imag + i * nt + j) = 0;
						}
						else
						{
							*(WNnk_real + i * nt + j) = pow(mag, *(nk_real + i * nt + j)) * cos(angle * sum0_real * PI / 180);		// WNnk (i, j) = WN ^ nk (i, j); Correct this one
							*(WNnk_imag + i * nt + j) = pow(mag, *(nk_real + i * nt + j)) * sin(angle * sum0_real * PI / 180);		// WNnk (i, j) = WN ^ nk (i, j); Correct this one
						}
					}
				}
				//			cout << "WNnk is: ";
				//			display_matrix_mem_complex (WNnk_real, WNnk_imag, nx*ny, nt);
				//			cout << "xn is: ";
				//			display_matrix_mem_complex (xn_real, xn_imag, nx*ny, nt);

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
						*(y_real + i * nx * ny + r * nx + c) = sum0_real;
						*(y_imag + i * nx * ny + r * nx + c) = sum0_imag;
					}
				}
			}
		}
		//	display_3d_matrix_mem_complex (y_real, y_imag, nx, ny, nt);
		int stop_itempfft = clock();
		cout << "\n CPU_Inverse Temp FFT:" << (stop_itempfft - start_itempfft) << endl;



		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(fft_b_real + i) = *(y_real + i);
			*(fft_b_imag + i) = -1 * *(y_imag + i);
		}


		// Line 577

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(resS_real + i) = *(fft_b_real + i) * sqrt_nt / nt;
			*(resS_imag + i) = *(fft_b_imag + i) * sqrt_nt / nt;
		}
		/********************************************Data Consistency Started   ****************************************************************************************************/
		cout << "\t Data Consistency Started .................................................................\n";
		/*		for (int i = 0; i < nx*ny; i++)
					for (int j = 0; j < nt; j++)
					{
						*(S_real + i*nx*ny + j) = *(resS_real + j*nx*ny + i);
						*(S_imag + i*nx*ny + j) = *(resS_imag + j*nx*ny + i);
					}
		*/
		// Reshape 3D to 2D 
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				for (int k = 0; k < nt; k++)
				{
					*(S_real + nt * j + nx * nt * i + k) = *(resS_real + nx * ny * k + ny * j + i);
					*(S_imag + nt * j + nx * nt * i + k) = *(resS_imag + nx * ny * k + ny * j + i);
				}
			}
		}

		double sumS = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumS = sumS + pow(*(S_real + i), 2) + pow(*(S_imag + i), 2);
		}
		double norm_S = sqrt(sumS);
		//cout << "\t\t norm_S is = " << norm_S;


		//	cout << " L is: ";
			//	display_matrix_mem_complex (L_real, L_imag, nx*ny, nt);
			//	cout << "S is: ";
			//	display_matrix_mem_complex (S_real, S_imag, nx*ny, nt);

		int start_Lps = clock();
		cout << "\n Soft Thresh of Sparse components:" << (stop_softs - start_softs) << endl;

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(LPSa_real + i) = *(L_real + i) + *(S_real + i);
			*(LPSa_imag + i) = *(L_imag + i) + *(S_imag + i);
		}
		int stop_Lps = clock();
		cout << "\n CPU_L+S time:" << (stop_Lps - start_Lps) << endl;
		double sumLPSa = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumLPSa = sumLPSa + pow(*(LPSa_real + i), 2) + pow(*(LPSa_imag + i), 2);
		}
		double norm_LPSa = sqrt(sumLPSa);
		cout << "\t\t norm_LPSa is = " << norm_LPSa;

		//	cout << " LPSa is: ";
		//	display_matrix_mem_complex (LPSa_real, LPSa_imag, nx*ny, nt);
		// Reshape 2D to 3D starts
/*			for (int i = 0; i < nx*ny; i++)
				for (int j = 0; j < nt; j++)
				{
					*(LPS_real + j*nx*ny + i) = *(LPSa_real + i*nx*ny + j);
					*(LPS_imag + j*nx*ny + i) = *(LPSa_imag + i*nx*ny + j);
				}
*/
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				for (int k = 0; k < nt; k++)
				{
					*(LPS_real + nx * ny * k + ny * j + i) = *(LPSa_real + nt * j + nx * nt * i + k);
					*(LPS_imag + nx * ny * k + ny * j + i) = *(LPSa_imag + nt * j + nx * nt * i + k);
				}
			}
		}

		double sumLPS = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumLPS = sumLPS + pow(*(LPS_real + i), 2) + pow(*(LPS_imag + i), 2);
		}
		double norm_LPS = sqrt(sumLPS);
		//cout << "\t\t norm_LPS is = " << norm_LPS;

/*				for (int k = 0; k < nt; k++)
				{
					for (int i = 0; i < nx; i++)
					{
						for (int j = 0; j < ny; j++)
						{
							*(temp_LPS_real + nx*ny*k + i*ny + j) = *(LPS_real + nx*ny*k + j*nx + i) ;
							*(temp_LPS_imag + nx*ny*k + i*ny + j) = *(LPS_imag + nx*ny*k + j*nx + i) ;
						}
					}
				}
				// Reshape 2D to 3D Ends
			   */
			   //	cout << " a_b1 is: ";
			 //	display_3d_matrix_mem_complex (a_b1_real, a_b1_imag, nx, ny, nc);
	/*******************************************************E(L+S) **********************************************************************************************************/

		int start_enc = clock();
		a_adjoint = 1;
		int start_b1 = clock();
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
				/*						double sumb_x1 = 0, suma_b1_x1 = 0;
										for (int i = 0; i < nx*ny; i++)
										{
											sumb_x1 = sumb_x1 + pow (*(b_x1_real + i), 2) + pow (*(b_x1_imag + i), 2);
											suma_b1_x1 = suma_b1_x1 + pow (*(a_b1_x1_real + i), 2) + pow (*(a_b1_x1_imag + i), 2);
										}
										double norm_b_x1 = sqrt (sumb_x1), norma_b_x1 = sqrt (suma_b1_x1);
										cout << "\t\t norm_b_x1 is = " << norm_b_x1 << endl;
										cout << "\t\t norma_b_x1 is = " << norma_b_x1 << endl;
				*/
				//			cout << " b_x1 is: ";
				//			display_matrix_mem_complex (b_x1_real, b_x1_imag, nx, ny);
				//			cout << " a_b1_x1 is: ";
				//			display_matrix_mem_complex (a_b1_x1_real, a_b1_x1_imag, nx, ny);

				for (int i = 0; i < nx * ny; i++)
				{
					*(X2_real + ch * nx * ny * nt + tt * nx * ny + i) = (*(b_x1_real + i) * *(a_b1_x1_real + i)) - (*(b_x1_imag + i) * *(a_b1_x1_imag + i));
					*(X2_imag + ch * nx * ny * nt + tt * nx * ny + i) = (*(b_x1_real + i) * *(a_b1_x1_imag + i)) + (*(b_x1_imag + i) * *(a_b1_x1_real + i));
				}
			}
		}
		int stop_b1 = clock();
		cout << "\n LSxb1:" << (stop_b1 - start_b1) << endl;
		//	cout << "X2 is: ";
		//	display_4d_matrix_mem_complex (X2_real, X2_imag, nx, ny, nt, nc);
		// Line 600
		sumx = 0;
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			sumx = sumx + pow(*(X2_real + i), 2) + pow(*(X2_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at Line 627 is = " << norm_x;

		dim = 1;
		K = nx / 2;
		int start_x1 = clock();
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
		int stop_x1 = clock();
		cout << "\n x1:" << (stop_x1 - start_x1) << endl;
		//	cout << "X2 is : 2";
		//	display_4d_matrix_mem_complex (X2_real, X2_imag, nx, ny, nt, nc);
		sumx = 0;
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			sumx = sumx + pow(*(X2_real + i), 2) + pow(*(X2_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at Line 646 is = " << norm_x;

		int nr = nc;
		int start_x2 = clock();

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

					double mag = sqrt(pow(WN_real, 2) + pow(WN_imag, 2));
					double angle = atan(WN_imag / WN_real) * 180 / PI;
					//N = nx; 
					for (int ii = 0; ii < nx; ii++)
					{
						for (int jj = 0; jj < nx; jj++)
						{
							double sum0_real = 0;	double sum0_imag = 0;
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
		int stop_x2 = clock();
		cout << "\n x2:" << (stop_x2 - start_x2) << endl;



		sumx = 0;
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			sumx = sumx + pow(*(X2_real + i), 2) + pow(*(X2_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at Line 678 is = " << norm_x;

				//	cout << "X2 is : 2";
				//	display_4d_matrix_mem_complex (X2_real, X2_imag, nx, ny, nt, nc);

				// Line 650 
		int start_x3 = clock();
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
		int stop_x3 = clock();
		cout << "\n x3:" << (stop_x3 - start_x3) << endl;
		sumx = 0;
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			sumx = sumx + pow(*(X2_real + i), 2) + pow(*(X2_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at Line 693 is = " << norm_x << endl;

				//	cout << "X2 is after rotate is :";
				//	display_4d_matrix_mem_complex (X2_real, X2_imag, nx, ny, nt, nc);

		double sqrt_nx = sqrt(nx_db);

		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			*(X2_real + i) = *(X2_real + i) * sqrt_nx;
			*(X2_imag + i) = *(X2_imag + i) * sqrt_nx;
		}

		//	cout << "X2 is after rotate is :";
		//	display_4d_matrix_mem_complex (X2_real, X2_imag, nx, ny, nt, nc);
		sumx = 0;
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			sumx = sumx + pow(*(X2_real + i), 2) + pow(*(X2_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at Line 775 is = " << norm_x << endl;
		dim = 2;
		K = ny / 2;
		int start_x4 = clock();
		for (int jj = 0; jj < nc; jj++)
		{
			int l = 0;

			for (int k = 0; k < nx * ny * nt; k++)
			{

				*(X1_real + l) = *(X2_real + nx * ny * nt * jj + k);
				*(X1_imag + l) = *(X2_imag + nx * ny * nt * jj + k);
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
					*(X2_real + nx * ny * nt * jj + k) = *(X1_real + l);
					*(X2_imag + nx * ny * nt * jj + k) = *(X1_imag + l);
					l = l + 1;
				}
			}
		}
		int stop_x4 = clock();
		cout << "\n x4:" << (stop_x4 - start_x4) << endl;
		//	cout << "X2 after rotate is :";
		//	display_4d_matrix_mem_complex (X2_real, X2_imag, nx, ny, nt, nc);
		sumx = 0;
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			sumx = sumx + pow(*(X2_real + i), 2) + pow(*(X2_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at Line 791 is = " << norm_x << endl;
				// Line 670

				// Line 690

				// Line 725
		int start_x5 = clock();
		for (int kk = 0; kk < nc; kk++)
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
					for (int jj = 0; jj < ny; jj++)
					{
						*(Xk_real + jj * nx) = *(x_real + ny * c + jj);
						*(Xk_imag + jj * nx) = *(x_imag + ny * c + jj);
					}
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

					//							WN_real = 0.9988;		WN_imag = -0.0491;
					WN_real = cos(2 * PI / nx);	WN_imag = sin(-2 * PI / nx);
					if (WN_real < 0)
						sign_cos = -1;
					else
						sign_cos = 1;

					if (WN_imag < 0)
						sign_sin = -1;
					else
						sign_sin = 1;

					double mag = sqrt(pow(WN_real, 2) + pow(WN_imag, 2));
					double angle = atan(WN_imag / WN_real) * 180 / PI;
					//N = nx; 
					for (int ii = 0; ii < nx; ii++)
					{
						for (int jj = 0; jj < nx; jj++)
						{
							double sum0_real = 0;	double sum0_imag = 0;
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
					//				cout << "nk is :";
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

							*(y_real + ny * c + ii) = sum0_real / N;
							*(y_imag + ny * c + ii) = sum0_imag / N;
						}
					}
				}
				//			cout << "y is :";
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
		int stop_x5 = clock();
		cout << "\n x5:" << (stop_x5 - start_x5) << endl;
		sumx = 0;
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			sumx = sumx + pow(*(X2_real + i), 2) + pow(*(X2_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at Line 827 is = " << norm_x << endl;
				//	cout << "X2 after rotate is :";
				//	display_4d_matrix_mem_complex (X2_real, X2_imag, nx, ny, nt, nc);

		int start_y = clock();
		for (int jj = 0; jj < nc; jj++)
		{
			int l = 0;

			for (int k = 0; k < nx * ny * nt; k++)
			{

				*(X1_real + l) = *(X2_real + nx * ny * nt * jj + k);
				*(X1_imag + l) = *(X2_imag + nx * ny * nt * jj + k);
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
					*(X2_real + nx * ny * nt * jj + k) = *(X1_real + l);
					*(X2_imag + nx * ny * nt * jj + k) = *(X1_imag + l);
					l = l + 1;
				}
			}
		}
		int stop_y = clock();
		cout << "\n y:" << (stop_y - start_y) << endl;
		sumx = 0;
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			sumx = sumx + pow(*(X2_real + i), 2) + pow(*(X2_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at Line 842 is = " << norm_x << endl;

				//	cout << "X2 after rotate at Line 770 is:";
				//	display_4d_matrix_mem_complex (X2_real, X2_imag, nx, ny, nt, nc);
				// Line 740

		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			*(res_real + i) = *(X2_real + i) * sqrt_ny;
			*(res_imag + i) = *(X2_imag + i) * sqrt_ny;
		}
		cout << "\t Applying Sampling Mask....................................\n\n";
		//	cout << "res after sqrt_ny mult is:";
		//	display_4d_matrix_mem_complex (res_real, res_imag, nx, ny, nt, nc);
		sumx = 0;
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			sumx = sumx + pow(*(res_real + i), 2) + pow(*(res_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at Line 693 is = " << norm_x << endl;
		int start_A = clock();
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
				*(res_real + nx * ny * nt * ch + i) = *(res_x_real + i) * *(a_mask_real + i);
				*(res_imag + nx * ny * nt * ch + i) = *(res_x_imag + i) * *(a_mask_real + i);
			}

		}
		int stop_A = clock();
		cout << "\n A:" << (stop_A - start_A) << endl;
		//	cout << "res at Line 797 is:";
		//	display_4d_matrix_mem_complex (res_real, res_imag, nx, ny, nt, nc);
		sumx = 0;
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			sumx = sumx + pow(*(res_real + i), 2) + pow(*(res_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		int stop_enc = clock();
		cout << "\n Data Consistency E(L+S):" << (stop_enc - start_enc) << endl;
		/*******************************************************E(L+S)-d**********************************************************************************************************/
		//cout << "\t\t norm_x at Line 869 is = " << norm_x << endl;
		int start_subd = clock();
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			*(resk_real + i) = *(res_real + i) - *(d_real + i);
			*(resk_imag + i) = *(res_imag + i) - *(d_imag + i);
			//*(b_real + i) = *(resk_real + i);
			//*(b_imag + i) = *(resk_imag + i);
		}
		int stop_subd = clock();
		cout << "\n B:" << (stop_subd - start_subd) << endl;


		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			*(b_real + i) = *(resk_real + i);
			*(b_imag + i) = *(resk_imag + i);
		}

		//  cout << "Line 778 ....... ";
		//	cout << "resk at Line 812 is:";
		//	display_4d_matrix_mem_complex (resk_real, resk_imag, nx, ny, nt, nc);

		// Line 778
		sumx = 0;
		for (int i = 0; i < nx * ny * nt * nc; i++)
		{
			sumx = sumx + pow(*(resk_real + i), 2) + pow(*(resk_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at Line 883 is = " << norm_x << endl;
	/******************************************************* Inverse Encoding Operator  *****************************************************************************************/
		int start_invenc = clock();
		int start_X1 = clock();
		a_adjoint = 1;
		for (int ch = 0; ch < nc; ch++)
		{
			for (int i = 0; i < nx * ny * nt; i++)
			{
				*(b1_real + i) = *(b_real + nx * ny * nt * ch + i);
				*(b1_imag + i) = *(b_imag + nx * ny * nt * ch + i);
			}

			//		cout << "b1 at 818 is:";
			//		display_3d_matrix_mem_complex (b1_real, b1_imag, nx, ny, nt);

			for (int i = 0; i < nx * ny * nt; i++)
			{
				//			*(X2_real + nx*ny*nt*ch + i) = *(b1_real + i) * *(a_mask_real + i);
				//			*(X2_imag + nx*ny*nt*ch + i) = *(b1_imag + i) * *(a_mask_real + i);
				*(out_real + i) = *(b1_real + i) * *(a_mask_real + i);
				*(out_imag + i) = *(b1_imag + i) * *(a_mask_real + i);
			}

			//		cout << "X2 at 827 is:";
			//		display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);

			dim = 1;
			K = ny / 2;

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

			//		cout << "X at Line 844 is:";
			//		display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);

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

					double mag = sqrt(pow(WN_real, 2) + pow(WN_imag, 2));
					double angle = atan(WN_imag / WN_real) * 180 / PI;
					//N = nx; 
					for (int ii = 0; ii < nx; ii++)
					{
						for (int jj = 0; jj < nx; jj++)
						{
							double sum0_real = 0;	double sum0_imag = 0;
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

			dim = 2;
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

					//							WN_real = 0.9988;		WN_imag = -0.0491;
					WN_real = cos(2 * PI / nx);	WN_imag = sin(-2 * PI / nx);
					if (WN_real < 0)
						sign_cos = -1;
					else
						sign_cos = 1;

					if (WN_imag < 0)
						sign_sin = -1;
					else
						sign_sin = 1;

					double mag = sqrt(pow(WN_real, 2) + pow(WN_imag, 2));
					double angle = atan(WN_imag / WN_real) * 180 / PI;
					//N = nx; 
					for (int ii = 0; ii < nx; ii++)
					{
						for (int jj = 0; jj < nx; jj++)
						{
							double sum0_real = 0;	double sum0_imag = 0;
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

			//		cout << "out at Line 975 is:";
			//		display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);


			/*
			for (int ii = 0; ii < nx*ny*nt*nc; ii++)
			{
			*(X2_real + ii) = *(out_real + ii);
			*(X2_imag + ii) = *(out_imag + ii);
			}
			// Line 941
			*/

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


			//	cout << "out at Line 986 is:";
			//	display_3d_matrix_mem_complex (out_real, out_imag, nx, ny, nt);


			// Line 955

			for (int i = 0; i < nx * ny * nt; i++)
			{
				*(x_array_real + nx * ny * nt * ch + i) = *(out_real + i) / sqrt_ny;
				*(x_array_imag + nx * ny * nt * ch + i) = *(out_imag + i) / sqrt_ny;
			}

			//	cout << "x_array at Line 994 is:";
			//	display_4d_matrix_mem_complex (x_array_real, x_array_imag, nx, ny, nt, nc);
			sumx = 0;
			for (int i = 0; i < nx * ny * nt * nc; i++)
			{
				sumx = sumx + pow(*(x_array_real + i), 2) + pow(*(x_array_imag + i), 2);
			}
			norm_x = sqrt(sumx);
			//cout << "\t\t norm_x at Line 1067 is = " << norm_x << endl;


		} // end for ch 0 -> nc
		int stop_X1 = clock();
		cout << "\n X1:" << (stop_X1 - start_X1) << endl;
		//cout << "\t Line 962 \n";

		// Line 966
		for (int i = 0; i < nx * ny * nc; i++)
		{
			*(conj_a_b1_imag + i) = *(a_b1_imag + i) * -1;
			*(conj_a_b1_real + i) = *(a_b1_real + i);
		}

		int start_C = clock();
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
				*(abs_a_b1_real + i) = pow(*(a_b1_real + i), 2) + pow(*(a_b1_imag + i), 2);
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



		}// end for tt = 0 -> nt
		int stop_C = clock();
		cout << "\n C:" << (stop_C - start_C) << endl;
		int stop_invenc = clock();
		cout << "\n Data Consistency Inverse E:" << (stop_invenc - start_invenc) << endl;


		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(E_resk_real + i) = *(res_2_real + i);
			*(E_resk_imag + i) = *(res_2_imag + i);
		}

		cout << "\t\t Data Consistency Done ............................................................\n";
		/******************************************************* Reshape and Norms  *****************************************************************************************/

		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				for (int k = 0; k < nt; k++)
				{
					*(E_resk_rshp_real + nt * j + nx * nt * i + k) = *(E_resk_real + nx * ny * k + ny * j + i);
					*(E_resk_rshp_imag + nt * j + nx * nt * i + k) = *(E_resk_imag + nx * ny * k + ny * j + i);
				}
			}
		}
		//	cout << "E_resk_rshp at Line 1037 is:";
		//	display_matrix_mem_complex (E_resk_rshp_real, E_resk_rshp_imag, nx*ny, nt);

		sumx = 0;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumx = sumx + pow(*(E_resk_rshp_real + i), 2) + pow(*(E_resk_rshp_imag + i), 2);
		}
		norm_x = sqrt(sumx);
		//cout << "\t\t norm_x at Line 1113 is = " << norm_x << endl;

	/******************************************************* Update values for next iteration  *********************************************************************************/
				// Line 1000
		int start_lpse = clock();
		cout << "\n Data Consistency Inverse E:" << (stop_invenc - start_invenc) << endl;
		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(M_real + i) = *(L_real + i) + *(S_real + i) - *(E_resk_rshp_real + i);
			*(M_imag + i) = *(L_imag + i) + *(S_imag + i) - *(E_resk_rshp_imag + i);

			//
			//*(Lpre_real + i) = *(L_real + i);
			//*(Lpre_imag + i) = *(L_imag + i);
			//*(M_M0_real + i) = *(M_real + i) - *(M0_real + i);
			//*(M_M0_imag + i) = *(M_imag + i) - *(M0_imag + i);

		}
		int stop_lpse = clock();
		cout << "\n Data Consistency M(LS-C):" << (stop_lpse - start_lpse) << endl;

		for (int i = 0; i < nx * ny * nt; i++)
		{
			*(Lpre_real + i) = *(L_real + i);
			*(Lpre_imag + i) = *(L_imag + i);
			*(M_M0_real + i) = *(M_real + i) - *(M0_real + i);
			*(M_M0_imag + i) = *(M_imag + i) - *(M0_imag + i);

		}




		//	cout << "M at Line 1045 is:";
		//	display_matrix_mem_complex (M_real, M_imag, nx*ny, nt);
		//	cout << "Lpre at Line 1045 is:";
		//	display_matrix_mem_complex (Lpre_real, Lpre_imag, nx*ny, nt);
		//	cout << "M-M0 at Line 1045 is:";
		//	display_matrix_mem_complex (M_M0_real, M_M0_imag, nx*ny, nt);



		// Check code for Line number 1013 - 1015

		// Line 1023
		double sumM_M0 = 0, sumM0 = 0;

		for (int i = 0; i < nx * ny * nt; i++)
		{
			sumM_M0 = sumM_M0 + pow(*(M_M0_real + i), 2) + pow(*(M_M0_imag + i), 2);
			sumM0 = sumM0 + pow(*(M0_real + i), 2) + pow(*(M0_imag + i), 2);
		}

		double norm_M_M0 = sqrt(sumM_M0);
		double norm_M0 = sqrt(sumM0);

		cout << "\t\t norm_M_M0 = " << norm_M_M0;
		cout << "\tnorm_M0 = " << norm_M0 << endl;

		cout << "\t\t Iteration No : " << ite << "done \n";
		int stop_sstop = clock();
		cout << "\n CPU_ Total L+S Computation time:" << (stop_sstop - start_sstart) << endl;
	}
}

/******************************************************* The End Below functions *****************************************************************************************/


/********************************************Function Jacobi SVD_for Eigen_Decomposition *******************************************************************************/


void SVD_Ejacobi(double* A_real, double* A_imag, double* U_real, double* U_imag, double* S, double* V_real, double* V_imag, int M, int N, double tol)
{
	double K = N, On = 0, Previous_Off = 99, Off = 9999999999999999;

	//cout <<"V(eye)";

	//	double *I    = (double *) malloc(N*N * sizeof(double));
	//	double *Z_nxn    = (double *) malloc(N*N * sizeof(double));
	//	double *Z_mxn    = (double *) malloc(M*N * sizeof(double));
	double* G_real = (double*)malloc(N * N * sizeof(double));
	double* G_imag = (double*)malloc(N * N * sizeof(double));
	double* temp_V_real = (double*)malloc(N * N * sizeof(double));
	double* temp_V_imag = (double*)malloc(N * N * sizeof(double));
	double* temp_A_real = (double*)malloc(M * N * sizeof(double));
	double* temp_A_imag = (double*)malloc(M * N * sizeof(double));
	double* trans_A_real = (double*)malloc(M * N * sizeof(double));
	double* trans_A_imag = (double*)malloc(M * N * sizeof(double));
	double* B_real = (double*)malloc(N * N * sizeof(double));
	double* B_imag = (double*)malloc(N * N * sizeof(double));
	double* temp_U_real = (double*)malloc(M * N * sizeof(double));
	double* temp_U_imag = (double*)malloc(M * N * sizeof(double));
	double* V_S_real = (double*)malloc(N * N * sizeof(double));
	double* V_S_imag = (double*)malloc(N * N * sizeof(double));
	double* inv_A_real = (double*)malloc(M * N * sizeof(double));
	double* inv_A_imag = (double*)malloc(M * N * sizeof(double));



	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (i == j)
			{
				*(V_real + i * N + j) = 1;
				*(V_imag + i * N + j) = 0;
			}
			else
			{
				*(V_real + i * N + j) = 0;
				*(V_imag + i * N + j) = 0;
			}
		}
	}

	//	cout <<" \n V(eye) Done ! \n";

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			On = On + (*(A_real + i * N + j) * *(A_real + i * N + j)) + (*(A_imag + i * N + j) * *(A_imag + i * N + j));
		}
	}

	//	cout <<" On = " << On <<endl;

	On = On / N;

	//	cout <<" On / N = " << On <<endl;

	double R, b_rr = 0, b_cc = 0, b_rc_real = 0, b_rc_imag = 0, m = 0, tau, t, sign_tau, C, S_real, S_imag, sum0_real, sum0_imag;
	int ite = 0;
	while (Off / On > tol)
		//	int x = 1;
		//	for (int g = 0; g<1; g++)
	{

		cout << "\t\t Off/On = " << Off / On << " in Iteration Number = " << " is " << ite << endl;
		ite = ite + 1;
		R = 0;

		for (int r = 0; r < N - 1; r++)
		{
			for (int c = r + 1; c < N; c++)
			{
				b_rr = 0; b_cc = 0; b_rc_real = 0; b_rc_imag = 0;
				for (int i = 0; i < M; i++)
				{
					b_rr = b_rr + (*(A_real + i * N + r) * *(A_real + i * N + r)) + (*(A_imag + i * N + r) * *(A_imag + i * N + r));
					b_cc = b_cc + (*(A_real + i * N + c) * *(A_real + i * N + c)) + (*(A_imag + i * N + c) * *(A_imag + i * N + c));
					b_rc_real = b_rc_real + ((*(A_real + i * N + r) * *(A_real + i * N + c)) + (*(A_imag + i * N + r) * *(A_imag + i * N + c)));
					b_rc_imag = b_rc_imag + ((*(A_real + i * N + r) * *(A_imag + i * N + c)) - (*(A_imag + i * N + r) * *(A_real + i * N + c)));
				}

				//cout << "b_rr = " << b_rr << "\t"; 
				//cout << "b_cc = " << b_cc << "\t"; 
				//cout << "b_rc = " << b_rc_real  << " + " << b_rc_imag << endl;

				m = pow((b_rc_real * b_rc_real) + (b_rc_imag * b_rc_imag), 0.5);

				if (m != 0)
				{
					tau = (b_cc - b_rr) / (2 * m);

					if (tau != 0)
					{
						R = R + 1;

						if (tau < 0)
							sign_tau = -1;
						else
							sign_tau = 1;

						t = sign_tau / ((sign_tau * tau) + pow((1 + tau * tau), 0.5));	// t = sign (tau) ./ (abs (tau) + sqrt (1+tau .^ 2));
						C = 1 / (pow((1 + t * t), 0.5));
						S_real = (b_rc_real * t * C) / m;
						S_imag = (b_rc_imag * t * C) / m;

						for (int i = 0; i < N; i++)
						{
							for (int j = 0; j < N; j++)
							{
								if (i == j)
								{
									*(G_real + i * N + j) = 1;
									*(G_imag + i * N + j) = 0;
								}
								else
								{
									*(G_real + i * N + j) = 0;
									*(G_imag + i * N + j) = 0;
								}
							}
						}


						*(G_real + r * N + r) = C;	*(G_imag + r * N + r) = 0;
						*(G_real + c * N + c) = C;	*(G_imag + c * N + c) = 0;
						*(G_real + r * N + c) = S_real;
						*(G_imag + r * N + c) = S_imag;
						*(G_real + c * N + r) = -1 * S_real;
						*(G_imag + c * N + r) = S_imag;


						for (int i = 0; i < M; i++)
						{
							for (int j = 0; j < N; j++)
							{
								*(temp_A_real + i * N + j) = 0;
								*(temp_A_imag + i * N + j) = 0;
							}
						}

						for (int i = 0; i < N; i++)
						{
							for (int j = 0; j < N; j++)
							{
								*(temp_V_real + i * N + j) = 0;
								*(temp_V_imag + i * N + j) = 0;
							}
						}

						//*temp_A_real = *Z_mxn;	*temp_A_imag = *Z_mxn;
						//*temp_V_real = *Z_nxn;	*temp_V_imag = *Z_nxn;

						for (int i = 0; i < M; i++)
						{
							for (int j = 0; j < N; j++)
							{
								sum0_real = 0;	sum0_imag = 0;

								for (int k = 0; k < N; k++)
								{
									sum0_real = sum0_real + (*(A_real + i * N + k) * *(G_real + k * N + j)) - (*(A_imag + i * N + k) * *(G_imag + k * N + j));
									sum0_imag = sum0_imag + (*(A_real + i * N + k) * *(G_imag + k * N + j)) + (*(A_imag + i * N + k) * *(G_real + k * N + j));
								}
								*(temp_A_real + i * N + j) = sum0_real;
								*(temp_A_imag + i * N + j) = sum0_imag;
							}
						}

						for (int i = 0; i < N; i++)
						{
							for (int j = 0; j < N; j++)
							{
								sum0_real = 0;	sum0_imag = 0;

								for (int k = 0; k < N; k++)
								{
									sum0_real = sum0_real + (*(V_real + i * N + k) * *(G_real + k * N + j)) - (*(V_imag + i * N + k) * *(G_imag + k * N + j));
									sum0_imag = sum0_imag + (*(V_real + i * N + k) * *(G_imag + k * N + j)) + (*(V_imag + i * N + k) * *(G_real + k * N + j));
									//								cout << "V_real = " << *(V_real + i*N + k) << "\tG_real = " << *(G_real + i*N + k) << "\tV_imag = " << *(V_imag + i*N + k) << "\tG_imag = " << *(G_imag + i*N + k) << endl; fgetc(stdin);
								}

								*(temp_V_real + i * N + j) = sum0_real;
								*(temp_V_imag + i * N + j) = sum0_imag;
							}
						}


						for (int i = 0; i < M; i++)
						{
							for (int j = 0; j < N; j++)
							{
								*(A_real + i * N + j) = *(temp_A_real + i * N + j);
								*(A_imag + i * N + j) = *(temp_A_imag + i * N + j);
							}
						}

						for (int i = 0; i < N; i++)
						{
							for (int j = 0; j < N; j++)
							{
								*(V_real + i * N + j) = *(temp_V_real + i * N + j);
								*(V_imag + i * N + j) = *(temp_V_imag + i * N + j);
							}
						}

					}

				}


			}

		}

		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				*(trans_A_real + j * M + i) = *(A_real + i * N + j);
				*(trans_A_imag + j * M + i) = -1 * *(A_imag + i * N + j);
			}
		}

		for (int i = 0; i < N; i++)		// transA (2, 8) and A = (8, 2)
		{
			for (int j = 0; j < N; j++)
			{
				sum0_real = 0;	sum0_imag = 0;

				for (int k = 0; k < M; k++)
				{
					sum0_real = sum0_real + (*(trans_A_real + i * M + k) * *(A_real + k * N + j)) - (*(trans_A_imag + i * M + k) * *(A_imag + k * N + j));
					sum0_imag = sum0_imag + (*(trans_A_real + i * M + k) * *(A_imag + k * N + j)) + (*(trans_A_imag + i * M + k) * *(A_real + k * N + j));
				}
				*(B_real + i * N + j) = sum0_real;
				*(B_imag + i * N + j) = sum0_imag;
			}
		}



		sum0_real = 0;

		for (int i = 0; i < N; i++)
		{
			for (int j = i + 1; j < N; j++)
			{
				sum0_real = sum0_real + pow(((*(B_real + i * N + j) * *(B_real + i * N + j)) + (*(B_imag + i * N + j) * *(B_imag + i * N + j))), 0.5);

			}
		}

		Off = sum0_real / (N * N);

		Previous_Off = Off;
	}

	double B_diag[col], IX[col], temp, tempX;

	for (int i = 0; i < N; i++)
	{
		IX[i] = i;
		if (*(B_real + i * N + i) < 0)
			B_diag[i] = pow(*(B_real + i * N + i) * -1, 0.5);
		else
			B_diag[i] = pow(*(B_real + i * N + i), 0.5);
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = N - 1; j >= i + 1; j--)
		{
			if (B_diag[j - 1] > B_diag[j])
			{
				temp = B_diag[j - 1];
				tempX = IX[j - 1];
				B_diag[j - 1] = B_diag[j];
				B_diag[j] = temp;
				IX[j - 1] = IX[j];
				IX[j] = tempX;
			}
		}
	}
	int IXint = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			IXint = IX[i];
			*(temp_A_real + j * N + i) = *(A_real + j * N + IXint);
			*(temp_A_imag + j * N + i) = *(A_imag + j * N + IXint);
		}
	}

	IXint = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			IXint = IX[i];
			*(temp_V_real + j * N + i) = *(V_real + j * N + IXint);
			*(temp_V_imag + j * N + i) = *(V_imag + j * N + IXint);
		}
	}

	//V_real = temp_V_real;	V_imag = temp_V_imag;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			*(U_real + j * N + i) = *(temp_A_real + j * N + i) / B_diag[i];
			*(U_imag + j * N + i) = *(temp_A_imag + j * N + i) / B_diag[i];
		}
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (i == j)
				*(S + j * N + i) = B_diag[i];
			else
				*(S + j * N + i) = 0;

		}
	}

	/*	cout << "S : \n ";
	for (int i = 0; i < N; i++)
	{
	for (int j = 0; j < N; j++)
	{
	cout << *(S + i*N + j) << "\t";

	}
	cout << endl;
	}
	*/
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			*(V_real + i * N + j) = *(temp_V_real + i * N + j);
			*(V_imag + i * N + j) = *(temp_V_imag + i * N + j);
		}
	}
}
/********************************************Function Jacobi SVD _for Eigen_decompostion End**************************************************************************/
/********************************************Function Jacobi SVD *****************************************************************************************************/
void SVD_jacobi(double* A_real, double* A_imag, double* U_real, double* U_imag, double* S, double* V_real, double* V_imag, int M, int N, double tol)
{
	double K = N, On = 0, Previous_Off = 99, Off = 9999999999999999;

	//cout <<"V(eye)";

	//	double *I    = (double *) malloc(N*N * sizeof(double));
	//	double *Z_nxn    = (double *) malloc(N*N * sizeof(double));
	//	double *Z_mxn    = (double *) malloc(M*N * sizeof(double));
	double* G_real = (double*)malloc(N * N * sizeof(double));
	double* G_imag = (double*)malloc(N * N * sizeof(double));
	double* temp_V_real = (double*)malloc(N * N * sizeof(double));
	double* temp_V_imag = (double*)malloc(N * N * sizeof(double));
	double* temp_A_real = (double*)malloc(M * N * sizeof(double));
	double* temp_A_imag = (double*)malloc(M * N * sizeof(double));
	double* trans_A_real = (double*)malloc(M * N * sizeof(double));
	double* trans_A_imag = (double*)malloc(M * N * sizeof(double));
	double* B_real = (double*)malloc(N * N * sizeof(double));
	double* B_imag = (double*)malloc(N * N * sizeof(double));
	double* temp_U_real = (double*)malloc(M * N * sizeof(double));
	double* temp_U_imag = (double*)malloc(M * N * sizeof(double));
	double* V_S_real = (double*)malloc(N * N * sizeof(double));
	double* V_S_imag = (double*)malloc(N * N * sizeof(double));
	double* inv_A_real = (double*)malloc(M * N * sizeof(double));
	double* inv_A_imag = (double*)malloc(M * N * sizeof(double));



	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (i == j)
			{
				*(V_real + i * N + j) = 1;
				*(V_imag + i * N + j) = 0;
			}
			else
			{
				*(V_real + i * N + j) = 0;
				*(V_imag + i * N + j) = 0;
			}
		}
	}

	//	cout <<" \n V(eye) Done ! \n";

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			On = On + (*(A_real + i * N + j) * *(A_real + i * N + j)) + (*(A_imag + i * N + j) * *(A_imag + i * N + j));
		}
	}

	//	cout <<" On = " << On <<endl;

	On = On / N;

	//	cout <<" On / N = " << On <<endl;

	double R, b_rr = 0, b_cc = 0, b_rc_real = 0, b_rc_imag = 0, m = 0, tau, t, sign_tau, C, S_real, S_imag, sum0_real, sum0_imag;
	int ite = 0;
	while (Off / On > tol)
		//	int x = 1;
		//	for (int g = 0; g<1; g++)
	{

		cout << "\t\t Off/On = " << Off / On << " in Iteration Number = " << " is " << ite << endl;
		ite = ite + 1;
		R = 0;

		for (int r = 0; r < N - 1; r++)
		{
			for (int c = r + 1; c < N; c++)
			{
				b_rr = 0; b_cc = 0; b_rc_real = 0; b_rc_imag = 0;
				for (int i = 0; i < M; i++)
				{
					b_rr = b_rr + (*(A_real + i * N + r) * *(A_real + i * N + r)) + (*(A_imag + i * N + r) * *(A_imag + i * N + r));
					b_cc = b_cc + (*(A_real + i * N + c) * *(A_real + i * N + c)) + (*(A_imag + i * N + c) * *(A_imag + i * N + c));
					b_rc_real = b_rc_real + ((*(A_real + i * N + r) * *(A_real + i * N + c)) + (*(A_imag + i * N + r) * *(A_imag + i * N + c)));
					b_rc_imag = b_rc_imag + ((*(A_real + i * N + r) * *(A_imag + i * N + c)) - (*(A_imag + i * N + r) * *(A_real + i * N + c)));
				}

				//cout << "b_rr = " << b_rr << "\t"; 
				//cout << "b_cc = " << b_cc << "\t"; 
				//cout << "b_rc = " << b_rc_real  << " + " << b_rc_imag << endl;

				m = pow((b_rc_real * b_rc_real) + (b_rc_imag * b_rc_imag), 0.5);

				if (m != 0)
				{
					tau = (b_cc - b_rr) / (2 * m);

					if (tau != 0)
					{
						R = R + 1;

						if (tau < 0)
							sign_tau = -1;
						else
							sign_tau = 1;

						t = sign_tau / ((sign_tau * tau) + pow((1 + tau * tau), 0.5));	// t = sign (tau) ./ (abs (tau) + sqrt (1+tau .^ 2));
						C = 1 / (pow((1 + t * t), 0.5));
						S_real = (b_rc_real * t * C) / m;
						S_imag = (b_rc_imag * t * C) / m;

						for (int i = 0; i < N; i++)
						{
							for (int j = 0; j < N; j++)
							{
								if (i == j)
								{
									*(G_real + i * N + j) = 1;
									*(G_imag + i * N + j) = 0;
								}
								else
								{
									*(G_real + i * N + j) = 0;
									*(G_imag + i * N + j) = 0;
								}
							}
						}


						*(G_real + r * N + r) = C;	*(G_imag + r * N + r) = 0;
						*(G_real + c * N + c) = C;	*(G_imag + c * N + c) = 0;
						*(G_real + r * N + c) = S_real;
						*(G_imag + r * N + c) = S_imag;
						*(G_real + c * N + r) = -1 * S_real;
						*(G_imag + c * N + r) = S_imag;


						for (int i = 0; i < M; i++)
						{
							for (int j = 0; j < N; j++)
							{
								*(temp_A_real + i * N + j) = 0;
								*(temp_A_imag + i * N + j) = 0;
							}
						}

						for (int i = 0; i < N; i++)
						{
							for (int j = 0; j < N; j++)
							{
								*(temp_V_real + i * N + j) = 0;
								*(temp_V_imag + i * N + j) = 0;
							}
						}

						//*temp_A_real = *Z_mxn;	*temp_A_imag = *Z_mxn;
						//*temp_V_real = *Z_nxn;	*temp_V_imag = *Z_nxn;

						for (int i = 0; i < M; i++)
						{
							for (int j = 0; j < N; j++)
							{
								sum0_real = 0;	sum0_imag = 0;

								for (int k = 0; k < N; k++)
								{
									sum0_real = sum0_real + (*(A_real + i * N + k) * *(G_real + k * N + j)) - (*(A_imag + i * N + k) * *(G_imag + k * N + j));
									sum0_imag = sum0_imag + (*(A_real + i * N + k) * *(G_imag + k * N + j)) + (*(A_imag + i * N + k) * *(G_real + k * N + j));
								}
								*(temp_A_real + i * N + j) = sum0_real;
								*(temp_A_imag + i * N + j) = sum0_imag;
							}
						}

						for (int i = 0; i < N; i++)
						{
							for (int j = 0; j < N; j++)
							{
								sum0_real = 0;	sum0_imag = 0;

								for (int k = 0; k < N; k++)
								{
									sum0_real = sum0_real + (*(V_real + i * N + k) * *(G_real + k * N + j)) - (*(V_imag + i * N + k) * *(G_imag + k * N + j));
									sum0_imag = sum0_imag + (*(V_real + i * N + k) * *(G_imag + k * N + j)) + (*(V_imag + i * N + k) * *(G_real + k * N + j));
									//								cout << "V_real = " << *(V_real + i*N + k) << "\tG_real = " << *(G_real + i*N + k) << "\tV_imag = " << *(V_imag + i*N + k) << "\tG_imag = " << *(G_imag + i*N + k) << endl; fgetc(stdin);
								}

								*(temp_V_real + i * N + j) = sum0_real;
								*(temp_V_imag + i * N + j) = sum0_imag;
							}
						}


						for (int i = 0; i < M; i++)
						{
							for (int j = 0; j < N; j++)
							{
								*(A_real + i * N + j) = *(temp_A_real + i * N + j);
								*(A_imag + i * N + j) = *(temp_A_imag + i * N + j);
							}
						}

						for (int i = 0; i < N; i++)
						{
							for (int j = 0; j < N; j++)
							{
								*(V_real + i * N + j) = *(temp_V_real + i * N + j);
								*(V_imag + i * N + j) = *(temp_V_imag + i * N + j);
							}
						}

					}

				}


			}

		}

		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				*(trans_A_real + j * M + i) = *(A_real + i * N + j);
				*(trans_A_imag + j * M + i) = -1 * *(A_imag + i * N + j);
			}
		}

		for (int i = 0; i < N; i++)		// transA (2, 8) and A = (8, 2)
		{
			for (int j = 0; j < N; j++)
			{
				sum0_real = 0;	sum0_imag = 0;

				for (int k = 0; k < M; k++)
				{
					sum0_real = sum0_real + (*(trans_A_real + i * M + k) * *(A_real + k * N + j)) - (*(trans_A_imag + i * M + k) * *(A_imag + k * N + j));
					sum0_imag = sum0_imag + (*(trans_A_real + i * M + k) * *(A_imag + k * N + j)) + (*(trans_A_imag + i * M + k) * *(A_real + k * N + j));
				}
				*(B_real + i * N + j) = sum0_real;
				*(B_imag + i * N + j) = sum0_imag;
			}
		}



		sum0_real = 0;

		for (int i = 0; i < N; i++)
		{
			for (int j = i + 1; j < N; j++)
			{
				sum0_real = sum0_real + pow(((*(B_real + i * N + j) * *(B_real + i * N + j)) + (*(B_imag + i * N + j) * *(B_imag + i * N + j))), 0.5);

			}
		}

		Off = sum0_real / (N * N);

		Previous_Off = Off;
	}

	double B_diag[col], IX[col], temp, tempX;

	for (int i = 0; i < N; i++)
	{
		IX[i] = i;
		if (*(B_real + i * N + i) < 0)
			B_diag[i] = pow(*(B_real + i * N + i) * -1, 0.5);
		else
			B_diag[i] = pow(*(B_real + i * N + i), 0.5);
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = N - 1; j >= i + 1; j--)
		{
			if (B_diag[j - 1] < B_diag[j])
			{
				temp = B_diag[j];
				tempX = IX[j];
				B_diag[j] = B_diag[j - 1];
				B_diag[j - 1] = temp;
				IX[j] = IX[j - 1];
				IX[j - 1] = tempX;
			}
		}
	}
	int IXint = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			IXint = IX[i];
			*(temp_A_real + j * N + i) = *(A_real + j * N + IXint);
			*(temp_A_imag + j * N + i) = *(A_imag + j * N + IXint);
		}
	}

	IXint = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			IXint = IX[i];
			*(temp_V_real + j * N + i) = *(V_real + j * N + IXint);
			*(temp_V_imag + j * N + i) = *(V_imag + j * N + IXint);
		}
	}

	//V_real = temp_V_real;	V_imag = temp_V_imag;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			*(U_real + j * N + i) = *(temp_A_real + j * N + i) / B_diag[i];
			*(U_imag + j * N + i) = *(temp_A_imag + j * N + i) / B_diag[i];
		}
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (i == j)
				*(S + j * N + i) = B_diag[i];
			else
				*(S + j * N + i) = 0;

		}
	}

	/*	cout << "S : \n ";
	for (int i = 0; i < N; i++)
	{
	for (int j = 0; j < N; j++)
	{
	cout << *(S + i*N + j) << "\t";

	}
	cout << endl;
	}
	*/
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			*(V_real + i * N + j) = *(temp_V_real + i * N + j);
			*(V_imag + i * N + j) = *(temp_V_imag + i * N + j);
		}
	}
}


void display_matrix_mem_real(double* A, int row, int col)
{
	cout << "\n\n";
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << *(A + i * row + j) << "\t";
		}
		cout << "\n";
	}
}



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

/********************************************Function Jacobi SVD _End*****************************************************************************************************/