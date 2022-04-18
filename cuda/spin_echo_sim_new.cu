#define _USE_MATH_DEFINES
#include <iostream>
#include <complex>
#include <cuda.h>
#include <cuComplex.h>
#include <stdlib.h>
#include <ctime>
#include <time.h>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <assert.h>
#include <typeinfo>
#include "device_launch_parameters.h"
#include <math.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <chrono>

const int DIM_L = 4;    // dimension of matrices to exponentiate (aka size of the Liouville space)
const int DIM_H = 2;    // dimension of hamiltonian and operators (aka size of Hilbert space)
const float v0 = 10.0;  // larmor freq
const float bw = 0.25;  // half bandwidth

struct timespec begin, end;
struct timespec t_begin, t_end;
struct dataStruct
{
    // physical host parameters
    cuFloatComplex* h_H;            // hamiltonian in hilbert space
    cuFloatComplex* h_H_L;          // hamiltonian in liouville space
    cuFloatComplex* h_U;            // propagator
    cuFloatComplex* h_rho;          // density matrix
    cuFloatComplex* h_M;            // signal (sum <I+>)
    cuFloatComplex* h_Mz;           // z magnetization
    cuFloatComplex* h_M_stencil;    // localization stencil
    cuFloatComplex* h_M_loc;        // local mag, for error checking
    cuFloatComplex* h_id;           // identity matrix
    cuFloatComplex* h_U90;          // 90 pulse operator
    cuFloatComplex* h_U180;         // 180 pulse operator
    float* h_v;                     // frequency distribution

    // pade approximant coefficients
    cuFloatComplex* h_CC;

    // device parameters
    cuFloatComplex* H;              // hamiltonian in hilbert space
    cuFloatComplex* H_L;            // hamiltonian in liouville space
    cuFloatComplex* U;              // propagator
    cuFloatComplex* rho;            // density matrix
    cuFloatComplex* rho_T;          // intermediate density matrix (for propagation function)
    cuFloatComplex* M_loc;          // local planar M
    cuFloatComplex* M_eval;         // <I+>
    cuFloatComplex* Mz_eval;        // <Iz>
    cuFloatComplex* M_loc_z;        // local Mz
    cuFloatComplex* M;              // pointer to M on device
    cuFloatComplex* Mz;             // Mz
    cuFloatComplex* M_stencil;      // stencil
    cuFloatComplex* U90;            // 90 pulse
    cuFloatComplex* U180;           // 180 pulse
    float* v;                       // frequencies

    // device intermediates
    cuFloatComplex* CC;             // pade approximant coefficients
    cuFloatComplex* X;              // intermediate matrix
    cuFloatComplex* V;              // intermediate matrix
    cuFloatComplex* VpX;            // used for GESV
    cuFloatComplex* VmX;            // used for GESV
    cuFloatComplex* prodA;          // products of A from A^2 thru A^13
    cuFloatComplex* d_sa;           // intermediate storage for A
    cuFloatComplex* id;             // identity matrix
    cuFloatComplex* d_ab;           // for gaussian elim
    cuFloatComplex* d_x;            // for gaussian elim
    cuFloatComplex* ed_a_T;         // for scaling exponent
    int* scale_int;                 // scale factor

    // simulation constants
    int nx;
    int ny;
    int nf;
    int nt;
    float tau;
    float dt;
    float lw;

};

// dataStruct preparation prototype
dataStruct prepare_dataStruct(int t_nx, int t_ny, float t_dt, float t_tau, float t_lw, int resample_freqs);
void generate_new_freqs(dataStruct x);

// basic tool prototypes
void ens_print_matrix(cuFloatComplex* h_a, int idx, int dim);
__device__ void ens_multiply_matrix(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex* d_c, int idx, int dim);
__device__ void ens_multiply_matrix_single(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex* d_c, int idx, int dim);
__device__ void ens_scalar_matrix_mult(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex a, int idx, int dim);
__device__ void ens_scalar_matrix_div(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex a, int idx, int dim);
__device__ void ens_swap_row(cuFloatComplex* d_a, int i, int j, int idx, int dim);

// solver prototypes
__device__ int ens_forward_elim(cuFloatComplex* d_ab, int dim, int idx);
__device__ void ens_back_sub(cuFloatComplex* d_ab, cuFloatComplex* d_x, int dim, int idx);
__global__ void ens_my_gesv(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex* d_c, cuFloatComplex* d_ab, cuFloatComplex* d_x, int nf);

// matrix exponential prototypes
__device__ float ens_opnorm(cuFloatComplex* d_a, int idx);
__global__ void scale_and_prep(cuFloatComplex* d_a, cuFloatComplex* d_sa, int* scale_int, int nf);
__global__ void calc_prod_mat(cuFloatComplex* d_sa, cuFloatComplex* prodA, int nf);
__global__ void calc_XV(cuFloatComplex* X, cuFloatComplex* V, cuFloatComplex* VpX, cuFloatComplex* VmX, cuFloatComplex* d_sa, cuFloatComplex* prodA, cuFloatComplex* CC, cuFloatComplex* id, int nf);
__global__ void rev_scaling(cuFloatComplex* ed_a, cuFloatComplex* ed_a_T, int* scale_int, int nf);

// stencil prototypes
__global__ void calc_local_M_2D(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex* d_c, int nrow, int ncol);
void calc_stencil_2D(cuFloatComplex* h_M_stencil, float xi, float p, int func, float s_w, float p_w, float d_w, int nx, int ny);

// physics prototypes
__global__ void calc_H(float alpha_x, float alpha_y, float alpha_z, float t, float* v, float dt, int nf, cuFloatComplex* M_loc, cuFloatComplex* M_loc_z, cuFloatComplex* H);
__global__ void calc_M_eval_2D(cuFloatComplex* rho, cuFloatComplex* M_eval, cuFloatComplex* Mz_eval, int nf);
__global__ void calc_M(cuFloatComplex* M_eval, cuFloatComplex* Mz_eval, cuFloatComplex* M, cuFloatComplex* Mz, int t_idx, int nf);
__global__ void time_propagate_liouville(cuFloatComplex* rho, cuFloatComplex* rho_T, cuFloatComplex* U, int nf);
__global__ void time_propagate_hilbert(cuFloatComplex* rho, cuFloatComplex* rho_T, cuFloatComplex* U, int nf);
__global__ void pulse(cuFloatComplex* rho, cuFloatComplex* rho_T, cuFloatComplex* pulse_op, int nf);
__global__ void ham_hilbert_to_liouville(cuFloatComplex* H, cuFloatComplex* H_L, float dt, float gamma1, float gamma2, float gamma3, int nf, float alpha_x, float alpha_y, cuFloatComplex* M_loc);

// custom operations for accuracy
__device__ static __inline__ cuFloatComplex my_cuCaddf(cuFloatComplex x, cuFloatComplex y);
__device__ static __inline__ cuFloatComplex my_cuCsubf(cuFloatComplex x, cuFloatComplex y);
__device__ static __inline__ cuFloatComplex my_cuCmulf(cuFloatComplex x, cuFloatComplex y);
__device__ static __inline__ cuFloatComplex my_cuCdivf(cuFloatComplex x, cuFloatComplex y);

// for error checking
void write_to_file_c(std::ofstream &output_name, cuFloatComplex* A, cuFloatComplex* h_A, int dim, int e_check, int t_idx, int num_steps);
void write_to_file_f(std::ofstream &output_name, float* A, float* h_A, int dim, int e_check, int t_idx, int num_steps);  

// for timing
void start_clock(timespec begin, int time_exec);
void end_clock(timespec begin, timespec end, std::ofstream &t_output, int time_exec);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                 //
//                              #### ##    ## ########    ##     ##    ###    #### ##    ##                                        //
//                               ##  ###   ##    ##       ###   ###   ## ##    ##  ###   ##                                        //
//                               ##  ####  ##    ##       #### ####  ##   ##   ##  ####  ##                                        //
//                               ##  ## ## ##    ##       ## ### ## ##     ##  ##  ## ## ##                                        //
//                               ##  ##  ####    ##       ##     ## #########  ##  ##  ####                                        //
//                               ##  ##   ###    ##       ##     ## ##     ##  ##  ##   ###                                        //
//                              #### ##    ##    ##       ##     ## ##     ## #### ##    ##                                        //
//                                                                                                                                 //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{

    // load input and output files & timing bool
    const char* trial_num = argv[1];
    int time_exec = atof(argv[2]);
    int error_check = atof(argv[3]);
    int resample_freqs = atof(argv[4]);
    int num_err = atof(argv[5]);

    // .txt
    std::string txt = ".txt";

    // base file names
    std::string s_sim_params_name = "sim_params";
    std::string s_echo_params_name = "echo_params";
    std::string s_real_output_name = "real_output";
    std::string s_imag_output_name = "imag_output";
    std::string s_z_real_output_name = "z_real_output";
    std::string s_z_imag_output_name = "z_imag_output";
    std::string s_time_output_name = "time_output";
    std::string s_freqs_output_name = "freqs_output";
    std::string s_pulse_op_output_name = "pulse_op_output";
    std::string s_rho0_output_name = "rho0_output";
    std::string s_stencil_output_name = "stencil_output";
    std::string s_m_eval_output_name = "m_eval_output";
    std::string s_mz_eval_output_name = "mz_eval_output";
    std::string s_m_local_output_name = "m_local_output";
    std::string s_mz_local_output_name = "mz_local_output";
    std::string s_h_output_name = "h_output";
    std::string s_hl_output_name = "hl_output";
    std::string s_u_output_name = "u_output";

    // add the number & .txt
    std::string t_sim_params_name = s_sim_params_name + trial_num + txt;
    std::string t_echo_params_name = s_echo_params_name + trial_num + txt;
    std::string t_r_output_name = s_real_output_name + trial_num + txt;
    std::string t_i_output_name = s_imag_output_name + trial_num + txt;
    std::string t_zr_output_name = s_z_real_output_name + trial_num + txt;
    std::string t_zi_output_name = s_z_imag_output_name + trial_num + txt;
    std::string t_t_output_name = s_time_output_name + trial_num + txt;
    std::string t_freqs_output_name = s_freqs_output_name + trial_num + txt;
    std::string t_pulse_op_output_name = s_pulse_op_output_name + trial_num + txt;
    std::string t_rho0_output_name = s_rho0_output_name + trial_num + txt;
    std::string t_stencil_output_name = s_stencil_output_name + trial_num + txt;
    std::string t_m_eval_output_name = s_m_eval_output_name + trial_num + txt;
    std::string t_mz_eval_output_name = s_mz_eval_output_name + trial_num + txt;
    std::string t_m_local_output_name = s_m_local_output_name + trial_num + txt;
    std::string t_mz_local_output_name = s_mz_local_output_name + trial_num + txt;
    std::string t_h_output_name = s_h_output_name + trial_num + txt;
    std::string t_hl_output_name = s_hl_output_name + trial_num + txt;
    std::string t_u_output_name = s_u_output_name + trial_num + txt;

    // convert to const char*
    const char* sim_params_name = t_sim_params_name.c_str();
    const char* echo_params_name = t_echo_params_name.c_str();
    const char* r_output_name = t_r_output_name.c_str();
    const char* i_output_name = t_i_output_name.c_str();
    const char* zr_output_name = t_zr_output_name.c_str();
    const char* zi_output_name = t_zi_output_name.c_str();
    const char* t_output_name = t_t_output_name.c_str();
    const char* freqs_output_name = t_freqs_output_name.c_str();
    const char* pulse_op_output_name = t_pulse_op_output_name.c_str();
    const char* rho0_output_name = t_rho0_output_name.c_str();
    const char* stencil_output_name = t_stencil_output_name.c_str();
    const char* m_eval_output_name = t_m_eval_output_name.c_str();
    const char* mz_eval_output_name = t_mz_eval_output_name.c_str();
    const char* m_local_output_name = t_m_local_output_name.c_str();
    const char* mz_local_output_name = t_mz_local_output_name.c_str();
    const char* h_output_name = t_h_output_name.c_str();
    const char* hl_output_name = t_hl_output_name.c_str();
    const char* u_output_name = t_u_output_name.c_str();

    // open sim params file
    std::ifstream sim_params;
    sim_params.open(sim_params_name);

    // open output files
    std::ofstream r_output;
    std::ofstream i_output;
    std::ofstream zr_output;
    std::ofstream zi_output;
    r_output.open(r_output_name);
    i_output.open(i_output_name);
    zr_output.open(zr_output_name);
    zi_output.open(zi_output_name);

    // only open if timing
    std::ofstream t_output;
    if (time_exec == 1)
    {
        t_output.open(t_output_name);
    }

    // error checking: creat outputs
    std::ofstream freqs_output;
    std::ofstream pulse_op_output;
    std::ofstream rho0_output;
    std::ofstream stencil_output;
    std::ofstream m_eval_output;
    std::ofstream mz_eval_output;
    std::ofstream m_local_output;
    std::ofstream mz_local_output;
    std::ofstream h_output;
    std::ofstream hl_output;
    std::ofstream u_output;

    // only open if checking for errors
    if (error_check == 1)
    {
        // error checking: open files
        freqs_output.open(freqs_output_name);
        pulse_op_output.open(pulse_op_output_name);
        rho0_output.open(rho0_output_name);
        stencil_output.open(stencil_output_name);
        m_eval_output.open(m_eval_output_name);
        mz_eval_output.open(mz_eval_output_name);
        m_local_output.open(m_local_output_name);
        mz_local_output.open(mz_local_output_name);
        h_output.open(h_output_name);
        hl_output.open(hl_output_name);
        u_output.open(u_output_name);
    }

    // simulation parameters
    int t_nx;
    int t_ny;
    float t_dt;
    float t_tau;
    float t_lw;

    while (sim_params >> t_nx >> t_ny >> t_dt >> t_tau >> t_lw)
    {
        // create data structure
        start_clock(begin, time_exec);
        dataStruct x;
        x = prepare_dataStruct(t_nx, t_ny, t_dt, t_tau, t_lw, resample_freqs);
        cudaDeviceSynchronize();
        end_clock(begin, end, t_output, time_exec);

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                                                                                                          //
        //            ########  ########   #######  ########     ###     ######      ###    ######## ########                       //
        //            ##     ## ##     ## ##     ## ##     ##   ## ##   ##    ##    ## ##      ##    ##                             //
        //            ##     ## ##     ## ##     ## ##     ##  ##   ##  ##         ##   ##     ##    ##                             //
        //            ########  ########  ##     ## ########  ##     ## ##   #### ##     ##    ##    ######                         //
        //            ##        ##   ##   ##     ## ##        ######### ##    ##  #########    ##    ##                             //
        //            ##        ##    ##  ##     ## ##        ##     ## ##    ##  ##     ##    ##    ##                             //
        //            ##        ##     ##  #######  ##        ##     ##  ######   ##     ##    ##    ########                       //
        //                                                                                                                          //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // open echo params again
        std::ifstream echo_params;
        echo_params.open(echo_params_name);

        // input arguments
        float alpha_x; float alpha_y; float alpha_z; float xi; float p;
        float gamma1; float gamma2; float gamma3;
        int stencil_form; float s_w; float p_w; float d_w;
        float angle90; float angle180; int phase90; int phase180;

        // big block of memory for copying stuff over: size = size of biggest possible thing to save = U = 16*nf
        cuFloatComplex* err_check = new cuFloatComplex[DIM_L*DIM_L*x.nf];
        float* err_check_f = new float[DIM_L*DIM_L*x.nf];

        while (echo_params >> alpha_x >> alpha_y >> alpha_z >> xi >> p >> gamma1 >> gamma2 >> gamma3 >> stencil_form >> s_w >> p_w >> d_w >> angle90 >> angle180 >> phase90 >> phase180)
        {
            // set time and time index to zero
            int t_idx = 0;
            float t = 0;

            if (resample_freqs == 1)
            {
                // resample frequencies
                start_clock(begin, time_exec);
                generate_new_freqs(x);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_f(freqs_output, x.v, err_check_f, x.nf, error_check, t_idx, num_err);
            }

            start_clock(begin, time_exec);
            // create the pulse operators and move to device
                
                // phase 0 = along x
                if (phase90 == 0)
                {
                    x.h_U90[0] = make_cuFloatComplex(cosf(angle90/2), 0);
                    x.h_U90[1] = make_cuFloatComplex(0, sinf(angle90/2));
                    x.h_U90[2] = make_cuFloatComplex(0, sinf(angle90/2));
                    x.h_U90[3] = make_cuFloatComplex(cosf(angle90/2), 0);
                }

                // phase 1 = along y
                if (phase90 == 1)
                {
                    x.h_U90[0] = make_cuFloatComplex(cosf(angle90/2), 0);
                    x.h_U90[1] = make_cuFloatComplex(sinf(angle90/2), 0);
                    x.h_U90[2] = make_cuFloatComplex(-sinf(angle90/2), 0);
                    x.h_U90[3] = make_cuFloatComplex(cosf(angle90/2), 0);
                }

                // phase 2 = along -x
                if (phase90 == 2)
                {
                    x.h_U90[0] = make_cuFloatComplex(cosf(angle90/2), 0);
                    x.h_U90[1] = make_cuFloatComplex(0, -sinf(angle90/2));
                    x.h_U90[2] = make_cuFloatComplex(0, -sinf(angle90/2));
                    x.h_U90[3] = make_cuFloatComplex(cosf(angle90/2), 0);
                }

                // phase 3 = along -y
                if (phase90 == 3)
                {
                    x.h_U90[0] = make_cuFloatComplex(cosf(angle90/2), 0);
                    x.h_U90[1] = make_cuFloatComplex(-sinf(angle90/2), 0);
                    x.h_U90[2] = make_cuFloatComplex(sinf(angle90/2), 0);
                    x.h_U90[3] = make_cuFloatComplex(cosf(angle90/2), 0);
                }

                // phase 0 = along x
                if (phase180 == 0)
                {
                    x.h_U180[0] = make_cuFloatComplex(cosf(angle180/2), 0);
                    x.h_U180[1] = make_cuFloatComplex(0, sinf(angle180/2));
                    x.h_U180[2] = make_cuFloatComplex(0, sinf(angle180/2));
                    x.h_U180[3] = make_cuFloatComplex(cosf(angle180/2), 0);
                }

                // phase 1 = along y
                if (phase180 == 1)
                {
                    x.h_U180[0] = make_cuFloatComplex(cosf(angle180/2), 0);
                    x.h_U180[1] = make_cuFloatComplex(sinf(angle180/2), 0);
                    x.h_U180[2] = make_cuFloatComplex(-sinf(angle180/2), 0);
                    x.h_U180[3] = make_cuFloatComplex(cosf(angle180/2), 0);
                }

                // phase 2 = along -x
                if (phase180 == 2)
                {
                    x.h_U180[0] = make_cuFloatComplex(cosf(angle180/2), 0);
                    x.h_U180[1] = make_cuFloatComplex(0, -sinf(angle180/2));
                    x.h_U180[2] = make_cuFloatComplex(0, -sinf(angle180/2));
                    x.h_U180[3] = make_cuFloatComplex(cosf(angle180/2), 0);
                }

                // phase 3 = along -y
                if (phase180 == 3)
                {
                    x.h_U180[0] = make_cuFloatComplex(cosf(angle180/2), 0);
                    x.h_U180[1] = make_cuFloatComplex(-sinf(angle180/2), 0);
                    x.h_U180[2] = make_cuFloatComplex(sinf(angle180/2), 0);
                    x.h_U180[3] = make_cuFloatComplex(cosf(angle180/2), 0);
                }

            //
            end_clock(begin, end, t_output, time_exec);

            // copy pulse operators to device
            start_clock(begin, time_exec);
            cudaMemcpy(x.U90, x.h_U90, DIM_H * DIM_H * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
            cudaMemcpy(x.U180, x.h_U180, DIM_H * DIM_H * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            end_clock(begin, end, t_output, time_exec);
            write_to_file_c(pulse_op_output, x.U90, err_check, DIM_H * DIM_H, error_check, t_idx, num_err);
            write_to_file_c(pulse_op_output, x.U180, err_check, DIM_H * DIM_H, error_check, t_idx, num_err);

            // start timing whole trial
            start_clock(t_begin, time_exec);

            // reset initial condition
            start_clock(begin, time_exec);
            cudaMemcpy(x.rho, x.h_rho, x.nf * DIM_H * DIM_H * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            end_clock(begin, end, t_output, time_exec);
            write_to_file_c(rho0_output, x.rho, err_check, x.nf * DIM_H * DIM_H, error_check, t_idx, num_err);

            // make the stencil
            start_clock(begin, time_exec);
            calc_stencil_2D(x.h_M_stencil, xi, p, stencil_form, s_w, p_w, d_w, x.nx, x.ny);
            end_clock(begin, end, t_output, time_exec);

            // copy stencils to device
            start_clock(begin, time_exec);
            cudaMemcpy(x.M_stencil, x.h_M_stencil, x.nf * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            end_clock(begin, end, t_output, time_exec);
            write_to_file_c(stencil_output, x.M_stencil, err_check, x.nf, error_check, t_idx, num_err);

            // pulse
            start_clock(begin, time_exec);
            pulse <<< x.nf / 350 + 1, 350 >>> (x.rho, x.rho_T, x.U90, x.nf);
            cudaDeviceSynchronize();
            end_clock(begin, end, t_output, time_exec);

            for (int i = 0; i < x.nt; i++)
            {
                // calculate M_eval
                start_clock(begin, time_exec);
                calc_M_eval_2D <<< x.nf / 1000 + 1, 1000 >>> (x.rho, x.M_eval, x.Mz_eval, x.nf);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(m_eval_output, x.M_eval, err_check, x.nf, error_check, t_idx, num_err);
                write_to_file_c(mz_eval_output, x.Mz_eval, err_check, x.nf, error_check, t_idx, num_err);

                // calc and save M
                start_clock(begin, time_exec);
                calc_M <<< 1, 1 >>> (x.M_eval, x.Mz_eval, x.M, x.Mz, t_idx, x.nf);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);

                // calculate M_loc
                start_clock(begin, time_exec);
                calc_local_M_2D <<< x.nf / 1000 + 1, 1000 >>> (x.M_eval, x.M_stencil, x.M_loc, x.nx, x.ny);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(m_local_output, x.M_loc, err_check, x.nf, error_check, t_idx, num_err);

                // calculate M_loc_z
                start_clock(begin, time_exec);
                calc_local_M_2D <<< x.nf / 1000 + 1, 1000 >>> (x.Mz_eval, x.M_stencil, x.M_loc_z, x.nx, x.ny);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(mz_local_output, x.M_loc_z, err_check, x.nf, error_check, t_idx, num_err);

                // calculate hamiltonian
                start_clock(begin, time_exec);
                calc_H <<< x.nf / 1000 + 1, 1000 >>> (alpha_x, alpha_y, alpha_z, t, x.v, x.dt, x.nf, x.M_loc, x.M_loc_z, x.H);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(h_output, x.H, err_check, x.nf * DIM_H * DIM_H, error_check, t_idx, num_err);

                // convert to liouville space
                start_clock(begin, time_exec);
                ham_hilbert_to_liouville <<< x.nf / 1000 + 1, 1000 >>> (x.H, x.H_L, x.dt, gamma1, gamma2, gamma3, x.nf, alpha_x, alpha_y, x.M_loc);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(hl_output, x.H_L, err_check, x.nf * DIM_L * DIM_L, error_check, t_idx, num_err);

                // CALCULATE MATRIX EXPONENT

                    // scale and prep
                    start_clock(begin, time_exec);
                    scale_and_prep <<< x.nf / 1000 + 1, 1000 >>> (x.H_L, x.d_sa, x.scale_int, x.nf);
                    cudaDeviceSynchronize();
                    end_clock(begin, end, t_output, time_exec);

                    // calculate matrix products
                    start_clock(begin, time_exec);
                    calc_prod_mat <<< x.nf / 1000 + 1, 1000 >>> (x.d_sa, x.prodA, x.nf);
                    cudaDeviceSynchronize();
                    end_clock(begin, end, t_output, time_exec);

                    // calculate V, X, V+X, V-X
                    start_clock(begin, time_exec);
                    calc_XV <<< x.nf / 1000 + 1, 1000 >>> (x.X, x.V, x.VpX, x.VmX, x.d_sa, x.prodA, x.CC, x.id, x.nf);
                    cudaDeviceSynchronize();
                    end_clock(begin, end, t_output, time_exec);

                    // gesv
                    start_clock(begin, time_exec);
                    ens_my_gesv <<< x.nf / 1000 + 1, 1000 >>> (x.VmX, x.VpX, x.U, x.d_ab, x.d_x, x.nf);
                    cudaDeviceSynchronize();
                    end_clock(begin, end, t_output, time_exec);

                    // undo scaling
                    start_clock(begin, time_exec);
                    rev_scaling <<< x.nf / 1000 + 1, 1000 >>> (x.U, x.ed_a_T, x.scale_int, x.nf);
                    cudaDeviceSynchronize();
                    end_clock(begin, end, t_output, time_exec);

                //

                // time propagate
                start_clock(begin, time_exec);
                time_propagate_liouville <<< x.nf / 1000 + 1, 1000 >>> (x.rho, x.rho_T, x.U, x.nf);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(u_output, x.U, err_check, x.nf * DIM_L * DIM_L, error_check, t_idx, num_err);

                // incrememt t, t_idx
                t_idx += 1;
                t += x.dt;
            }

            // calculate M_eval
            start_clock(begin, time_exec);
            calc_M_eval_2D <<< x.nf / 1000 + 1, 1000 >>> (x.rho, x.M_eval, x.Mz_eval, x.nf);
            cudaDeviceSynchronize();
            end_clock(begin, end, t_output, time_exec);
            write_to_file_c(m_eval_output, x.M_eval, err_check, x.nf, error_check, t_idx, num_err);
            write_to_file_c(mz_eval_output, x.Mz_eval, err_check, x.nf, error_check, t_idx, num_err);

            // calc and save M
            start_clock(begin, time_exec);
            calc_M <<< 1, 1 >>> (x.M_eval, x.Mz_eval, x.M, x.Mz, t_idx, x.nf);
            cudaDeviceSynchronize();
            end_clock(begin, end, t_output, time_exec);

            // incrememt t_idx
            t_idx += 1;

            // pulse
            start_clock(begin, time_exec);
            pulse <<< x.nf / 350 + 1, 350 >>> (x.rho, x.rho_T, x.U180, x.nf);
            cudaDeviceSynchronize();
            end_clock(begin, end, t_output, time_exec);

            for (int i = 0; i < 2 * x.nt; i++)
            {
                // calculate M_eval
                start_clock(begin, time_exec);
                calc_M_eval_2D <<< x.nf / 1000 + 1, 1000 >>> (x.rho, x.M_eval, x.Mz_eval, x.nf);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(m_eval_output, x.M_eval, err_check, x.nf, error_check, t_idx, num_err);
                write_to_file_c(mz_eval_output, x.Mz_eval, err_check, x.nf, error_check, t_idx, num_err);

                // calc and save M
                start_clock(begin, time_exec);
                calc_M <<< 1, 1 >>> (x.M_eval, x.Mz_eval, x.M, x.Mz, t_idx, x.nf);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);

                // calculate M_loc
                start_clock(begin, time_exec);
                calc_local_M_2D <<< x.nf / 1000 + 1, 1000 >>> (x.M_eval, x.M_stencil, x.M_loc, x.nx, x.ny);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(m_local_output, x.M_loc, err_check, x.nf, error_check, t_idx, num_err);

                // calculate M_loc_z
                start_clock(begin, time_exec);
                calc_local_M_2D <<< x.nf / 1000 + 1, 1000 >>> (x.Mz_eval, x.M_stencil, x.M_loc_z, x.nx, x.ny);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(mz_local_output, x.M_loc_z, err_check, x.nf, error_check, t_idx, num_err);

                // calculate hamiltonian
                start_clock(begin, time_exec);
                calc_H <<< x.nf / 1000 + 1, 1000 >>> (alpha_x, alpha_y, alpha_z, t, x.v, x.dt, x.nf, x.M_loc, x.M_loc_z, x.H);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(h_output, x.H, err_check, x.nf * DIM_H * DIM_H, error_check, t_idx, num_err);

                // convert to liouville space
                start_clock(begin, time_exec);
                ham_hilbert_to_liouville <<< x.nf / 1000 + 1, 1000 >>> (x.H, x.H_L, x.dt, gamma1, gamma2, gamma3, x.nf, alpha_x, alpha_y, x.M_loc);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(hl_output, x.H_L, err_check, x.nf * DIM_L * DIM_L, error_check, t_idx, num_err);

                // CALCULATE MATRIX EXPONENT

                    // scale and prep
                    start_clock(begin, time_exec);
                    scale_and_prep <<< x.nf / 1000 + 1, 1000 >>> (x.H_L, x.d_sa, x.scale_int, x.nf);
                    cudaDeviceSynchronize();
                    end_clock(begin, end, t_output, time_exec);

                    // calculate matrix products
                    start_clock(begin, time_exec);
                    calc_prod_mat <<< x.nf / 1000 + 1, 1000 >>> (x.d_sa, x.prodA, x.nf);
                    cudaDeviceSynchronize();
                    end_clock(begin, end, t_output, time_exec);

                    // calculate V, X, V+X, V-X
                    start_clock(begin, time_exec);
                    calc_XV <<< x.nf / 1000 + 1, 1000 >>> (x.X, x.V, x.VpX, x.VmX, x.d_sa, x.prodA, x.CC, x.id, x.nf);
                    cudaDeviceSynchronize();
                    end_clock(begin, end, t_output, time_exec);

                    // gesv
                    start_clock(begin, time_exec);
                    ens_my_gesv <<< x.nf / 1000 + 1, 1000 >>> (x.VmX, x.VpX, x.U, x.d_ab, x.d_x, x.nf);
                    cudaDeviceSynchronize();
                    end_clock(begin, end, t_output, time_exec);

                    // undo scaling
                    start_clock(begin, time_exec);
                    rev_scaling <<< x.nf / 1000 + 1, 1000 >>> (x.U, x.ed_a_T, x.scale_int, x.nf);
                    cudaDeviceSynchronize();
                    end_clock(begin, end, t_output, time_exec);

                //

                // time propagate
                start_clock(begin, time_exec);
                time_propagate_liouville <<< x.nf / 1000 + 1, 1000 >>> (x.rho, x.rho_T, x.U, x.nf);
                cudaDeviceSynchronize();
                end_clock(begin, end, t_output, time_exec);
                write_to_file_c(u_output, x.U, err_check, x.nf * DIM_L * DIM_L, error_check, t_idx, num_err);

                // incrememt t, t_idx
                t_idx += 1;
                t += x.dt;
            }

            // calculate M_eval
            start_clock(begin, time_exec);
            calc_M_eval_2D <<< x.nf / 1000 + 1, 1000 >>> (x.rho, x.M_eval, x.Mz_eval, x.nf);
            cudaDeviceSynchronize();
            end_clock(begin, end, t_output, time_exec);
            write_to_file_c(m_eval_output, x.M_eval, err_check, x.nf, error_check, t_idx, num_err);
            write_to_file_c(mz_eval_output, x.Mz_eval, err_check, x.nf, error_check, t_idx, num_err);

            // calc and save M
            start_clock(begin, time_exec);
            calc_M <<< 1, 1 >>> (x.M_eval, x.Mz_eval, x.M, x.Mz, t_idx, x.nf);
            cudaDeviceSynchronize();
            end_clock(begin, end, t_output, time_exec);

            // copy magnetization over
            start_clock(begin, time_exec);
            cudaMemcpy(x.h_M, x.M, (3 * x.nt + 2) * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
            cudaMemcpy(x.h_Mz, x.Mz, (3 * x.nt + 2) * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
            end_clock(begin, end, t_output, time_exec);

            // write to output file
            start_clock(begin, time_exec);
            for (int i = 0; i < 3 * x.nt + 2; i++)
            {
                r_output << cuCrealf(x.h_M[i]) << " ";
                i_output << cuCimagf(x.h_M[i]) << " ";
                zr_output << cuCrealf(x.h_Mz[i]) << " "; 
                zi_output << cuCimagf(x.h_Mz[i]) << " "; 
            }
            r_output << "\n";
            i_output << "\n";
            zr_output << "\n";
            zi_output << "\n";
            end_clock(begin, end, t_output, time_exec);

            // time for total execution of trial
            end_clock(t_begin, t_end, t_output, time_exec);

            // add a linebreak to timing output
            if (time_exec == 1)
            {
                t_output << "\n";

            }
        }

        // close echo params file
        echo_params.close();

        // free all memory before next iteration
        delete [] x.h_H;
        delete [] x.h_H_L;
        delete [] x.h_U;
        delete [] x.h_rho;
        delete [] x.h_M;
        delete [] x.h_Mz;
        delete [] x.h_M_stencil;
        delete [] x.h_M_loc;
        delete [] x.h_id;
        delete [] x.h_U90;
        delete [] x.h_U180;
        delete [] x.h_v;
        delete [] x.h_CC;
        delete [] err_check;
        delete [] err_check_f;

        cudaFree(x.H); 
        cudaFree(x.H_L);
        cudaFree(x.U);    
        cudaFree(x.rho);    
        cudaFree(x.rho_T);
        cudaFree(x.M_loc);
        cudaFree(x.M_eval);
        cudaFree(x.Mz_eval);
        cudaFree(x.M_loc_z);     
        cudaFree(x.M); 
        cudaFree(x.Mz);    
        cudaFree(x.M_stencil);       
        cudaFree(x.U90);
        cudaFree(x.U180);
        cudaFree(x.v);     

        cudaFree(x.CC);    
        cudaFree(x.X);
        cudaFree(x.V);
        cudaFree(x.VpX);
        cudaFree(x.VmX);
        cudaFree(x.prodA);
        cudaFree(x.d_sa);
        cudaFree(x.id);
        cudaFree(x.d_ab);
        cudaFree(x.d_x);
        cudaFree(x.ed_a_T);
        cudaFree(x.scale_int);
    }

    r_output.close();
    i_output.close();
    zr_output.close();
    zi_output.close();
    t_output.close();
    sim_params.close();

    return 0;
}

dataStruct prepare_dataStruct(int t_nx, int t_ny, float t_dt, float t_tau, float t_lw, int resample_freqs)
{
    // create the struct
    dataStruct x;

    // simulation constants
    x.nx = t_nx;
    x.ny = t_ny;
    x.nf = t_nx * t_ny;
    x.dt = t_dt;
    x.tau = t_tau;
    x.lw = t_lw;
    float gamma = 2 * M_PI * 1000000;
    x.nt = (int)(gamma * t_tau / t_dt);

    // allocate host memory (parameters)
    x.h_H         = new cuFloatComplex[x.nf * DIM_H * DIM_H];
    x.h_H_L       = new cuFloatComplex[x.nf * DIM_L * DIM_L];
    x.h_U         = new cuFloatComplex[x.nf * DIM_L * DIM_L];
    x.h_rho       = new cuFloatComplex[x.nf * DIM_H * DIM_H];
    x.h_M         = new cuFloatComplex[3*x.nt + 2];
    x.h_Mz        = new cuFloatComplex[3*x.nt + 2];
    x.h_M_stencil = new cuFloatComplex[x.nf];
    x.h_M_loc     = new cuFloatComplex[x.nf];
    x.h_id        = new cuFloatComplex[DIM_L * DIM_L];
    x.h_U90       = new cuFloatComplex[DIM_H * DIM_H];
    x.h_U180      = new cuFloatComplex[DIM_H * DIM_H];
    x.h_v         = new float[x.nf];

    // allocate host memory (temporary variables)
    x.h_CC = new cuFloatComplex[14];

    // allocate device memory (parameters)
    cudaMalloc(&x.H, x.nf * DIM_H * DIM_H * sizeof(cuFloatComplex)); 
    cudaMalloc(&x.H_L, x.nf * DIM_L * DIM_L * sizeof(cuFloatComplex));
    cudaMalloc(&x.U, x.nf * DIM_L * DIM_L * sizeof(cuFloatComplex));    
    cudaMalloc(&x.rho, x.nf * DIM_H * DIM_H * sizeof(cuFloatComplex));    
    cudaMalloc(&x.rho_T, x.nf * DIM_H * DIM_H * sizeof(cuFloatComplex));  
    cudaMalloc(&x.M_loc, x.nf * sizeof(cuFloatComplex));         
    cudaMalloc(&x.M_eval, x.nf * sizeof(cuFloatComplex));          
    cudaMalloc(&x.Mz_eval, x.nf * sizeof(cuFloatComplex));          
    cudaMalloc(&x.M_loc_z, x.nf * sizeof(cuFloatComplex));         
    cudaMalloc(&x.M, (3 * x.nt + 2) * sizeof(cuFloatComplex));
    cudaMalloc(&x.Mz, (3 * x.nt + 2) * sizeof(cuFloatComplex));          
    cudaMalloc(&x.M_stencil, x.nf * sizeof(cuFloatComplex));         
    cudaMalloc(&x.U90, DIM_H * DIM_H * sizeof(cuFloatComplex));  
    cudaMalloc(&x.U180, DIM_H * DIM_H * sizeof(cuFloatComplex));  
    cudaMalloc(&x.v, x.nf * sizeof(float));       

    // allocate device memory (intermediate variables)
    cudaMalloc(&x.CC, 14 * sizeof(cuFloatComplex));           
    cudaMalloc(&x.X, x.nf * DIM_L * DIM_L * sizeof(cuFloatComplex));
    cudaMalloc(&x.V, x.nf * DIM_L * DIM_L * sizeof(cuFloatComplex));
    cudaMalloc(&x.VpX, x.nf * DIM_L * DIM_L * sizeof(cuFloatComplex));
    cudaMalloc(&x.VmX, x.nf * DIM_L * DIM_L * sizeof(cuFloatComplex));
    cudaMalloc(&x.prodA, x.nf * 12 * DIM_L * DIM_L * sizeof(cuFloatComplex));
    cudaMalloc(&x.d_sa, x.nf * DIM_L * DIM_L * sizeof(cuFloatComplex));
    cudaMalloc(&x.id, DIM_L * DIM_L * sizeof(cuFloatComplex));
    cudaMalloc(&x.d_ab, x.nf * DIM_L * (DIM_L + 1) * sizeof(cuFloatComplex));
    cudaMalloc(&x.d_x, x.nf * DIM_L * sizeof(cuFloatComplex));
    cudaMalloc(&x.ed_a_T, x.nf * DIM_L * DIM_L * sizeof(cuFloatComplex));
    cudaMalloc(&x.scale_int, x.nf * sizeof(int));

    // fill rho with zeros
    for (int i = 0; i < x.nf; i++)
    {
        for (int j = 0; j < DIM_H * DIM_H; j++)
        {
            // initialize rho as all in up state
            x.h_rho[DIM_H * DIM_H * i + j] = make_cuFloatComplex(0, 0);
        }
    }

    // set initial condition
    for (int i = 0; i < x.nf; i++)
    {
        // initialize rho as all in up state
        x.h_rho[DIM_H * DIM_H * i] = make_cuFloatComplex(1, 0);
    }

    // load pre-gen freq distribution
    std::ifstream freqs;
    freqs.open("freqs.txt");
    float v_i;
    int idx = 0;
    while (freqs >> v_i)
    {
        x.h_v[idx] = v_i;
        if (idx == x.nf - 1)
        {
            break;
        }
        idx += 1;
    }
    freqs.close();

    if (resample_freqs == 1)
    {
        // create distribution of frequencies
        unsigned freq_seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(freq_seed);
        std::cauchy_distribution<float> distribution(v0, x.lw);
    
        // sample
        int idx = 0;
        while (idx < x.nf) 
        {
            float v_i = distribution(generator);
            if ( (v_i >= (v0 - bw)) && (v_i < (v0 + bw)) ) // enforce bandwidth of 0.5MHz
            {
                x.h_v[idx] = v_i;
                idx += 1;
            }
        }
    }

    // make an identity matrix
    for (int i = 0; i < DIM_L; i++)
    {
        for (int j = 0; j < DIM_L; j++)
        {
            if (i == j)
            {
                x.h_id[DIM_L * i + j] = make_cuFloatComplex(1, 0);
            }
            else
            {
                x.h_id[DIM_L * i + j] = make_cuFloatComplex(0, 0);
            }
        }
    }

    // assign pade coefficient values
    x.h_CC[0] = make_cuFloatComplex(64764752532480000, 0);
    x.h_CC[1] = make_cuFloatComplex(32382376266240000, 0);
    x.h_CC[2] = make_cuFloatComplex(7771770303897600, 0);
    x.h_CC[3] = make_cuFloatComplex(1187353796428800, 0);
    x.h_CC[4] = make_cuFloatComplex(129060195264000, 0);
    x.h_CC[5] = make_cuFloatComplex(10559470521600, 0);
    x.h_CC[6] = make_cuFloatComplex(670442572800, 0);
    x.h_CC[7] = make_cuFloatComplex(33522128640, 0);
    x.h_CC[8] = make_cuFloatComplex(1323241920, 0);
    x.h_CC[9] = make_cuFloatComplex(40840800, 0);
    x.h_CC[10] = make_cuFloatComplex(960960, 0);
    x.h_CC[11] = make_cuFloatComplex(16380, 0);
    x.h_CC[12] = make_cuFloatComplex(182, 0);
    x.h_CC[13] = make_cuFloatComplex(1, 0);

    // copy the frequencies to the device
    cudaMemcpy(x.v, x.h_v, x.nf * sizeof(float), cudaMemcpyHostToDevice);

    // copy over pade approximant coefficients
    cudaMemcpy(x.CC, x.h_CC, 14 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(x.id, x.h_id, DIM_L * DIM_L * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    return x;
}

void generate_new_freqs(dataStruct x)
{

    // re-create distribution & random seed
    unsigned freq_seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(freq_seed);
    std::cauchy_distribution<float> distribution(v0, x.lw);
    
    // re-sample
    int idx = 0;
    while (idx < x.nf) 
    {
        float v_i = distribution(generator);
        if ( (v_i >= (v0 - bw)) && (v_i < (v0 + bw)) )
        {
            x.h_v[idx] = v_i;
            idx += 1;
        }
    }

    // copy to device
    cudaMemcpy(x.v, x.h_v, x.nf * sizeof(float), cudaMemcpyHostToDevice);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ########     ###     ######  ####  ######     ########  #######   #######  ##        ######             //
// ##     ##   ## ##   ##    ##  ##  ##    ##       ##    ##     ## ##     ## ##       ##    ##            //
// ##     ##  ##   ##  ##        ##  ##             ##    ##     ## ##     ## ##       ##                  //
// ########  ##     ##  ######   ##  ##             ##    ##     ## ##     ## ##        ######             //
// ##     ## #########       ##  ##  ##             ##    ##     ## ##     ## ##             ##            //
// ##     ## ##     ## ##    ##  ##  ##    ##       ##    ##     ## ##     ## ##       ##    ##            //
// ########  ##     ##  ######  ####  ######        ##     #######   #######  ########  ######             //
//                                                                                                         //
//                         Basic Utilities (matrix multiplication, printing, etc)                          //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// round function
__global__ void round_to_digit(cuFloatComplex *d_a, int decimal_places, int nf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        for (int i = 0; i < DIM_L; i++)
        {
            for (int j = 0; j < DIM_L; j++)
            {

                // value to scale by: 10^digits
                float base = 10.0;
                float s = powf(base, decimal_places);

                // current matrix element
                cuFloatComplex dij = d_a[DIM_L*DIM_L*idx + DIM_L*i + j];
                float r_dij = cuCrealf(dij);
                float i_dij = cuCimagf(dij);
                
                // scale up
                float r1_dij = (float) s*r_dij;
                float i1_dij = (float) s*i_dij;

                // round off
                float r2_dij = rintf(r1_dij);
                float i2_dij = rintf(i1_dij);

                // scale back down
                float r3_dij = (float) r2_dij/s;
                float i3_dij = (float) i2_dij/s;

                // replace the value
                d_a[DIM_L*DIM_L*idx + DIM_L*i + j] = make_cuFloatComplex(r3_dij, i3_dij);
            }
        }
    }
}

// print matrix
void ens_print_matrix(cuFloatComplex* h_a, int idx, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            std::cout << cuCrealf(h_a[dim * dim * idx + dim * i + j]) << " + " << cuCimagf(h_a[dim * dim * idx + dim * i + j]) << "i";
            if (j == dim - 1)
            {
                std::cout << ";";
            }
            else
            {
                std::cout << ",   ";
            }
        }
        std::cout << std::endl;
    }
}

// multiply matrices, product of subsets of an ensemble saved to the subset of an ensemble
__device__ void ens_multiply_matrix(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex* d_c, int idx, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            // initialize to zero
            d_c[dim * dim * idx + dim * i + j] = make_cuFloatComplex(0, 0);

            // do the multiplication
            for (int k = 0; k < dim; k++)
            {
                cuFloatComplex temp = my_cuCmulf(d_a[dim * dim * idx + dim * i + k], d_b[dim * dim * idx + dim * k + j]);
                d_c[dim * dim * idx + dim * i + j] = my_cuCaddf(d_c[dim * dim * idx + dim * i + j], temp);
            }
        }
    }
}

// same as above, product of subsets of an ensemble saved to a single destination
__device__ void ens_multiply_matrix_single(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex* d_c, int idx, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            // initialize to zero
            d_c[dim * i + j] = make_cuFloatComplex(0, 0);

            // do the multiplication
            for (int k = 0; k < dim; k++)
            {
                cuFloatComplex temp = my_cuCmulf(d_a[dim * dim * idx + dim * i + k], d_b[dim * dim * idx + dim * k + j]);
                d_c[dim * i + j] = my_cuCaddf(d_c[dim * i + j], temp);
            }
        }
    }
}

// multiply matrix by a scalar
__device__ void ens_scalar_matrix_mult(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex a, int idx, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            d_b[dim * dim * idx + dim * i + j] = my_cuCmulf(d_a[dim * dim * idx + dim * i + j], a);
        }
    }
}

// divide matrix by a scalar
__device__ void ens_scalar_matrix_div(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex a, int idx, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            d_b[dim * dim * idx + dim * i + j] = my_cuCdivf(d_a[dim * dim * idx + dim * i + j], a);
        }
    }
}

// swap two rows of a matrix for gaussian elim (input = augmented matrix)
__device__ void ens_swap_row(cuFloatComplex* d_a, int i, int j, int idx, int dim)
{
    for (int k = 0; k <= dim; k++)
    {
        cuFloatComplex temp = d_a[(dim + 1) * dim * idx + (dim + 1) * i + k];
        d_a[(dim + 1) * dim * idx + (dim + 1) * i + k] = d_a[(dim + 1) * dim * idx + (dim + 1) * j + k];
        d_a[(dim + 1) * dim * idx + (dim + 1) * j + k] = temp;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  ######      ###    ##     ##  ######   ######  ####    ###    ##    ##    ######## ##       #### ##     ##           //
// ##    ##    ## ##   ##     ## ##    ## ##    ##  ##    ## ##   ###   ##    ##       ##        ##  ###   ###           //
// ##         ##   ##  ##     ## ##       ##        ##   ##   ##  ####  ##    ##       ##        ##  #### ####           //
// ##   #### ##     ## ##     ##  ######   ######   ##  ##     ## ## ## ##    ######   ##        ##  ## ### ##           //
// ##    ##  ######### ##     ##       ##       ##  ##  ######### ##  ####    ##       ##        ##  ##     ##           //
// ##    ##  ##     ## ##     ## ##    ## ##    ##  ##  ##     ## ##   ###    ##       ##        ##  ##     ##           //
//  ######   ##     ##  #######   ######   ######  #### ##     ## ##    ##    ######## ######## #### ##     ##           //
//                                                                                                                       //
//                                 Gaussian Elimination Algorithm to solve Ax = B for x                                  //
//                            forward_elim puts the augmented matrix A|B into echelon form                               //
//                           back_sub calculates the values of x using A|B from forward elim                             //
//             inputs d_ab = pointer to matrix A|B in device memory, d_x = pointer to memory for solution x              //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int ens_forward_elim(cuFloatComplex* d_ab, int dim, int idx)
{
    // loop over each row
    for (int i = 0; i < dim - 1; i++)
    {

        // pre-sort matrix (bad sort alg, need better for larger matrices)
        for (int j = i; j < dim - 1; j++)
        {
            for (int k = j + 1; k < dim; k++)
            {
                float max_val = cuCabsf(d_ab[(dim + 1) * dim * idx + (dim + 1) * j + i]);
                if (max_val < cuCabsf(d_ab[(dim + 1) * dim * idx + (dim + 1) * k + i]))
                {
                    ens_swap_row(d_ab, j, k, idx, dim);
                }
            }
        }

        // scale and subtract it from each row below it
        for (int j = i + 1; j < dim; j++)
        {
            // check for singularity before dividing: if so, swap rows with the row with the largest value in that column
            if (cuCabsf(d_ab[(dim + 1) * i + i]) == 0)
            {
                continue;
            }

            // calculate the scale factor
            cuFloatComplex scale = my_cuCdivf(d_ab[(dim + 1) * dim * idx + (dim + 1) * j + i], d_ab[(dim + 1) * dim * idx + (dim + 1) * i + i]);

            // do the subtraction
            for (int k = 0; k < dim + 1; k++)
            {
                d_ab[(dim + 1) * dim * idx + (dim + 1) * j + k] = my_cuCsubf(d_ab[(dim + 1) * dim * idx + (dim + 1) * j + k], my_cuCmulf(scale, d_ab[(dim + 1) * dim * idx + (dim + 1) * i + k]));
            }

        }
    }

    return 0;
}

// backward substitution
__device__ void ens_back_sub(cuFloatComplex* d_ab, cuFloatComplex* d_x, int dim, int idx)
{
    // the final element
    d_x[dim * idx + dim - 1] = my_cuCdivf(d_ab[(dim + 1) * dim * idx + (dim + 1) * (dim - 1) + dim], d_ab[(dim + 1) * dim * idx + (dim + 1) * (dim - 1) + dim - 1]);

    // recursively calculate the rest
    for (int i = dim - 2; i >= 0; i--)
    {
        // to keep track of the sum
        cuFloatComplex s = my_cuCdivf(d_ab[(dim + 1) * dim * idx + (dim + 1) * i + dim], d_ab[(dim + 1) * dim * idx + (dim + 1) * i + i]);
        for (int j = i + 1; j < dim; j++)
        {
            s = my_cuCsubf(s, my_cuCdivf(my_cuCmulf(d_x[dim * idx + j], d_ab[(dim + 1) * dim * idx + (dim + 1) * i + j]), d_ab[(dim + 1) * dim * idx + (dim + 1) * i + i]));
        }
        d_x[dim * idx + i] = s;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                      
//                    ##     ## ##    ##     ######   ########  ######  ##     ##                                       //
//                    ###   ###  ##  ##     ##    ##  ##       ##    ## ##     ##                                       //
//                    #### ####   ####      ##        ##       ##       ##     ##                                       //
//                    ## ### ##    ##       ##   #### ######    ######  ##     ##                                       //
//                    ##     ##    ##       ##    ##  ##             ##  ##   ##                                        //
//                    ##     ##    ##       ##    ##  ##       ##    ##   ## ##                                         //
//                    ##     ##    ##        ######   ########  ######     ###                                          //
//                                                                                                                      //
//                 solves the system of equations AC = B for C, where A and B are complex NxN matrices                  //
//              d_a, d_b = pointers to matrices A and B in device memory, d_c = pointer to memory for C                 //
//       uses Gaussian Elimination to calculate one column of C using one column of B at a time to solve Ax = B         //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void ens_my_gesv(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex* d_c, cuFloatComplex* d_ab, cuFloatComplex* d_x, int nf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        // loop over each column of B, attach to A, and solve (d_b)
        for (int col = 0; col < DIM_L; col++)
        {
            // make the augmented matrix A + one col of B
            for (int row_h = 0; row_h < DIM_L; row_h++)
            {
                for (int col_h = 0; col_h < DIM_L; col_h++)
                {
                    d_ab[(DIM_L + 1) * DIM_L * idx + (DIM_L + 1) * row_h + col_h] = d_a[DIM_L * DIM_L * idx + DIM_L * row_h + col_h];
                }
                d_ab[(DIM_L + 1) * DIM_L * idx + (DIM_L + 1) * row_h + DIM_L] = d_b[DIM_L * DIM_L * idx + DIM_L * row_h + col];
            }

            // do the forward elimination to put d_ab in reduced echelon form
            ens_forward_elim(d_ab, DIM_L, idx);

            // do the backward substitution to get the solution
            ens_back_sub(d_ab, d_x, DIM_L, idx);

            // assign d_x to d_c
            for (int row = 0; row < DIM_L; row++)
            {
                d_c[DIM_L * DIM_L * idx + DIM_L * row + col] = d_x[DIM_L * idx + row];
            }
        }
    }
}

// finding the operator norm of a matrix
__device__ float ens_opnorm(cuFloatComplex* d_a, int idx)
{
    // compute the column sums
    float col_sum[DIM_L];
    for (int col = 0; col < DIM_L; col++)
    {
        // initialize to zero
        col_sum[col] = 0;
        for (int row = 0; row < DIM_L; row++)
        {
            col_sum[col] += cuCabsf(d_a[DIM_L * DIM_L * idx + DIM_L * row + col]);
        }
    }

    // find the maximum value of the column sum
    float max_col_sum = 0;
    for (int col = 0; col < DIM_L; col++)
    {
        if (col_sum[col] > max_col_sum)
        {
            max_col_sum = col_sum[col];
        }
    }
    return max_col_sum;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ##     ##    ###    ######## ########  #### ##     ##    ######## ##     ## ########   #######  ##    ## ######## ##    ## ########   //
// ###   ###   ## ##      ##    ##     ##  ##   ##   ##     ##        ##   ##  ##     ## ##     ## ###   ## ##       ###   ##    ##      //
// #### ####  ##   ##     ##    ##     ##  ##    ## ##      ##         ## ##   ##     ## ##     ## ####  ## ##       ####  ##    ##      //
// ## ### ## ##     ##    ##    ########   ##     ###       ######      ###    ########  ##     ## ## ## ## ######   ## ## ##    ##      //
// ##     ## #########    ##    ##   ##    ##    ## ##      ##         ## ##   ##        ##     ## ##  #### ##       ##  ####    ##      //
// ##     ## ##     ##    ##    ##    ##   ##   ##   ##     ##        ##   ##  ##        ##     ## ##   ### ##       ##   ###    ##      //
// ##     ## ##     ##    ##    ##     ## #### ##     ##    ######## ##     ## ##         #######  ##    ## ######## ##    ##    ##      //
//                                                                                                                                       //
//    calculate a matrix exponent, courtesy of Julia source code (skipping the scaling/permuting steps in LAPACK.gebal!)                 //
//       inputs = d_a, device pointer to the matrix A to exponentiate, ed_a, pointer to memory to store exp(A)                           //
//                          also a bunch of pointers to preallocated memory for speed                                                    //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// scaling the matrix
__global__ void scale_and_prep(cuFloatComplex* d_a, cuFloatComplex* d_sa, int* scale_int, int nf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        // calculate the operator norm
        float nA = ens_opnorm(d_a, idx);

        // calculate the appropriate scaling based on the operator norm, and then scale the matrix
        float s = log2(nA / 5.4);
        int si_int = __float2int_ru(s);
        scale_int[idx] = si_int;
        if (si_int > 0)
        {
            cuFloatComplex si = make_cuFloatComplex(powf(2, si_int), 0);
            ens_scalar_matrix_div(d_a, d_sa, si, idx, DIM_L);
        }
        else
        {
            cuFloatComplex si = make_cuFloatComplex(1, 0);
            ens_scalar_matrix_div(d_a, d_sa, si, idx, DIM_L);
        }
    }
}

// calculating product matrices
__global__ void calc_prod_mat(cuFloatComplex* d_sa, cuFloatComplex* prodA, int nf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        // products of the matrix: first iteration = square
        for (int i = 0; i < DIM_L; i++)
        {
            for (int j = 0; j < DIM_L; j++)
            {
                prodA[12 * DIM_L * DIM_L * idx + DIM_L * i + j] = make_cuFloatComplex(0, 0);
                for (int k = 0; k < DIM_L; k++)
                {
                    cuFloatComplex temp = my_cuCmulf(d_sa[DIM_L * DIM_L * idx + DIM_L * i + k], d_sa[DIM_L * DIM_L * idx + DIM_L * k + j]);
                    prodA[12 * DIM_L * DIM_L * idx + DIM_L * i + j] = my_cuCaddf(prodA[12 * DIM_L * DIM_L * idx + DIM_L * i + j], temp);
                }
            }
        }
        // recursively fill in the rest
        for (int pow_idx = 1; pow_idx < 12; pow_idx++)
        {
            for (int i = 0; i < DIM_L; i++)
            {
                for (int j = 0; j < DIM_L; j++)
                {
                    prodA[12 * DIM_L * DIM_L * idx + DIM_L * DIM_L * pow_idx + DIM_L * i + j] = make_cuFloatComplex(0, 0);
                    for (int k = 0; k < DIM_L; k++)
                    {
                        cuFloatComplex temp = my_cuCmulf(prodA[12 * DIM_L * DIM_L * idx + DIM_L * DIM_L * (pow_idx - 1) + DIM_L * i + k], // ...
                            d_sa[DIM_L * DIM_L * idx + DIM_L * k + j]);
                        prodA[12 * DIM_L * DIM_L * idx + DIM_L * DIM_L * pow_idx + DIM_L * i + j] = // ...
                            my_cuCaddf(prodA[12 * DIM_L * DIM_L * idx + DIM_L * DIM_L * pow_idx + DIM_L * i + j], temp);
                    }
                }
            }
        }
    }
}

// calculating X and V matrices, and sum and diff
__global__ void calc_XV(cuFloatComplex* X, cuFloatComplex* V, cuFloatComplex* VpX, cuFloatComplex* VmX, cuFloatComplex* d_sa, cuFloatComplex* prodA, cuFloatComplex* CC, cuFloatComplex* id, int nf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        // calculate matrices X and V
        for (int i = 0; i < DIM_L; i++)
        {
            // V[DIM_L*DIM_L*idx + DIM_L*i + i] = make_cuFloatComplex(0,0);

            for (int j = 0; j < DIM_L; j++)
            {

                // set to zero or lose your mind
                V[DIM_L * DIM_L * idx + DIM_L * i + j] = make_cuFloatComplex(0, 0);
                X[DIM_L * DIM_L * idx + DIM_L * i + j] = make_cuFloatComplex(0, 0);

                // calculate
                V[DIM_L * DIM_L * idx + DIM_L * i + j] = my_cuCaddf(V[DIM_L * DIM_L * idx + DIM_L * i + j], my_cuCmulf(CC[0], id[DIM_L * i + j]));
                X[DIM_L * DIM_L * idx + DIM_L * i + j] = my_cuCaddf(X[DIM_L * DIM_L * idx + DIM_L * i + j], my_cuCmulf(CC[1], d_sa[DIM_L * DIM_L * idx + DIM_L * i + j]));
                for (int pow_idx = 0; pow_idx < 6; pow_idx++)
                {
                    int v_idx = 2 * pow_idx;
                    int u_idx = 2 * pow_idx + 1;

                    cuFloatComplex temp_v = my_cuCaddf(V[DIM_L * DIM_L * idx + DIM_L * i + j], // ...
                        my_cuCmulf(CC[v_idx + 2], prodA[12 * DIM_L * DIM_L * idx + DIM_L * DIM_L * v_idx + DIM_L * i + j]));
                    V[DIM_L * DIM_L * idx + DIM_L * i + j] = temp_v;

                    cuFloatComplex temp_u = my_cuCaddf(X[DIM_L * DIM_L * idx + DIM_L * i + j], // ...
                        my_cuCmulf(CC[u_idx + 2], prodA[12 * DIM_L * DIM_L * idx + DIM_L * DIM_L * u_idx + DIM_L * i + j]));
                    X[DIM_L * DIM_L * idx + DIM_L * i + j] = temp_u;
                }
            }
        }

        // calculate the numerator and denominator of the pade approximant (V + X and V - X)
        for (int i = 0; i < DIM_L; i++)
        {
            for (int j = 0; j < DIM_L; j++)
            {
                VpX[DIM_L * DIM_L * idx + DIM_L * i + j] = my_cuCaddf(V[DIM_L * DIM_L * idx + DIM_L * i + j], X[DIM_L * DIM_L * idx + DIM_L * i + j]);
                VmX[DIM_L * DIM_L * idx + DIM_L * i + j] = my_cuCsubf(V[DIM_L * DIM_L * idx + DIM_L * i + j], X[DIM_L * DIM_L * idx + DIM_L * i + j]);
            }
        }
    }
}

// reversing the scaling
__global__ void rev_scaling(cuFloatComplex* ed_a, cuFloatComplex* ed_a_T, int* scale_int, int nf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        // squaring X = exp(A/2^n) to finish
        int si_int = scale_int[idx];
        if (si_int > 0)
        {
            for (int i = 0; i < si_int; i++)
            {
                // square
                ens_multiply_matrix(ed_a, ed_a, ed_a_T, idx, DIM_L);

                // reassign ed_a => ed_a * ed_a
                for (int j = 0; j < DIM_L; j++)
                {
                    for (int k = 0; k < DIM_L; k++)
                    {
                        ed_a[DIM_L * DIM_L * idx + DIM_L * j + k] = ed_a_T[DIM_L * DIM_L * idx + DIM_L * j + k];
                    }
                }
            }
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  ######  ######## ######## ##    ##  ######  #### ##          ########  #######   #######  ##        ######          //
// ##    ##    ##    ##       ###   ## ##    ##  ##  ##             ##    ##     ## ##     ## ##       ##    ##         //
// ##          ##    ##       ####  ## ##        ##  ##             ##    ##     ## ##     ## ##       ##               //
//  ######     ##    ######   ## ## ## ##        ##  ##             ##    ##     ## ##     ## ##        ######          //
//       ##    ##    ##       ##  #### ##        ##  ##             ##    ##     ## ##     ## ##             ##         //
// ##    ##    ##    ##       ##   ### ##    ##  ##  ##             ##    ##     ## ##     ## ##       ##    ##         //
//  ######     ##    ######## ##    ##  ######  #### ########       ##     #######   #######  ########  ######          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void calc_local_M_2D(cuFloatComplex* d_a, cuFloatComplex* d_b, cuFloatComplex* d_c, int nrow, int ncol)
{
    // d_a = M_eval, d_b = stencil, d_c = M_local
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nrow * ncol)
    {
        // coordinates of the spin
        int col_idx = idx % ncol;
        int row_idx = (idx - col_idx) / ncol;
        d_c[ncol * row_idx + col_idx] = make_cuFloatComplex(0, 0);

        // elementwise multiply
        for (int i = 0; i < nrow; i++)
        {
            for (int j = 0; j < ncol; j++)
            {
                int i_a = (i + row_idx) % nrow;
                int j_a = (j + col_idx) % ncol;
                cuFloatComplex temp = my_cuCmulf(d_a[ncol * i_a + j_a], d_b[ncol * i + j]);
                d_c[ncol * row_idx + col_idx] = my_cuCaddf(d_c[ncol * row_idx + col_idx], temp);
            }
        }
    }
}

void calc_stencil_2D(cuFloatComplex* h_M_stencil, float xi, float p, int func, float s_w, float p_w, float d_w, int nx, int ny)
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {

            int mod_i = ((i + nx / 2) % nx) - nx / 2;
            int mod_j = ((j + ny / 2) % ny) - ny / 2;
            float x = (float)mod_i;
            float y = (float)mod_j;
            float theta = atan2f(y, x);
            float ang_fac = s_w + p_w * cosf(theta) + d_w * cosf(2.0 * theta);
            float r = sqrt(powf(x, 2) + powf(y, 2));

            if (func == 0 || func == 3)
            {
                float sten = expf(-powf(r/xi, p));
                h_M_stencil[ny * i + j] = make_cuFloatComplex(sten, 0);
            }
            else if (func == 1 || func == 4)
            {
                float sten = ang_fac / powf(r, p);
                h_M_stencil[ny * i + j] = make_cuFloatComplex(sten, 0);
            }
            else if (func == 2 || func == 5)
            {
                float xh = 2.0 * (r / xi);
                float sten = (ang_fac / powf(xh, 4)) * (xh * cosf(xh) - sin(xh));
                h_M_stencil[ny * i + j] = make_cuFloatComplex(sten, 0);
            }
            else
            {
                float sten = 1.0 / (nx * ny);
                h_M_stencil[ny * i + j] = make_cuFloatComplex(sten, 0);
            }
        }
    }

    // no self coupling
    h_M_stencil[0] = make_cuFloatComplex(0, 0);

    /*
    // if scaled global, adjust
    if (func == 3 || func == 4 || func == 5)
    {
        float sten = 0.0;
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                sten = sten + cuCrealf(h_M_stencil[ny * i + j]);
            }
        }

        // divide out
        sten = 1.0 / sten;

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                h_M_stencil[ny * i + j] = make_cuFloatComplex(sten, 0.0f);
            }
        } 

    }
    */
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                        ########  ##     ## ##    ##  ######  ####  ######   ######                                   //
//                        ##     ## ##     ##  ##  ##  ##    ##  ##  ##    ## ##    ##                                  //
//                        ##     ## ##     ##   ####   ##        ##  ##       ##                                        //    
//                        ########  #########    ##     ######   ##  ##        ######                                   //
//                        ##        ##     ##    ##          ##  ##  ##             ##                                  //
//                        ##        ##     ##    ##    ##    ##  ##  ##    ## ##    ##                                  //
//                        ##        ##     ##    ##     ######  ####  ######   ######                                   //
//                                                                                                                      //
//                ######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######                         //
//                ##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##                        //
//                ##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##                              //
//                ######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######                         //
//                ##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##                        //
//                ##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##                        //
//                ##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######                         //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// EXACT HAMILTONIAN WITH ALL TIME DEPENDENCE

/*
__global__ void calc_H(float alpha_x, float alpha_y, float alpha_z, float t, float* v, float dt, int nf, cuFloatComplex* M_loc, cuFloatComplex* M_loc_z, cuFloatComplex* H)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        // H = -(v - v0)Iz - alpha_x Ix<Mx> - alpha_y Iy<My> - alpha_z Iz<Mz>

        // some terms we need
        float v0t = __fmul_rn(v0, t);
        cuFloatComplex exp_p = make_cuFloatComplex(cosf(__fmul_rn(2.0f, v0t)), sinf(__fmul_rn(2.0f, v0t))); // e^(2iv0t)
        cuFloatComplex exp_m = make_cuFloatComplex(cosf(__fmul_rn(2.0f, v0t)), __fmul_rn(-1.0f, sinf(__fmul_rn(2.0f, v0t)))); // e^(-2iv0t)
        cuFloatComplex aMx = my_cuCmulf(make_cuFloatComplex(__fmul_rn(0.25f, alpha_x), 0.0f), M_loc[idx]); // (1/4) * M * alpha_x
        cuFloatComplex aMy = my_cuCmulf(make_cuFloatComplex(__fmul_rn(0.25f, alpha_y), 0.0f), M_loc[idx]); // (1/4) * M * alpha_y
        cuFloatComplex aMz = my_cuCmulf(make_cuFloatComplex(__fmul_rn(0.5f, alpha_z), 0.0f), M_loc_z[idx]); // (1/2) * Mz * alpha_z
        cuFloatComplex m1 = make_cuFloatComplex(-1.0f, 0.0f); // -1 as complex float

        // initialize with zeros
        H[DIM_H * DIM_H * idx] = make_cuFloatComplex(0.0f, 0.0f); 
        H[DIM_H * DIM_H * idx + 1] = make_cuFloatComplex(0.0f, 0.0f);  
        H[DIM_H * DIM_H * idx + 2] = make_cuFloatComplex(0.0f, 0.0f); 
        H[DIM_H * DIM_H * idx + 3] = make_cuFloatComplex(0.0f, 0.0f);       

        // -vIz - <My>Iy
        H[DIM_H * DIM_H * idx] = my_cuCaddf(H[DIM_H * DIM_H * idx], 
                                 make_cuFloatComplex(__fmul_rn(0.5f, (v0 - v[idx])), 0.0f));        // (1/2)(v0 - v)
        H[DIM_H * DIM_H * idx + 1] = my_cuCaddf(H[DIM_H * DIM_H * idx + 1], 
                                     my_cuCsubf(my_cuCmulf(aMy, exp_p), cuConjf(aMy)));             // -alpha_y * (conjM - M * e^(2iv0t)) / 4 
        H[DIM_H * DIM_H * idx + 2] = my_cuCaddf(H[DIM_H * DIM_H * idx + 2], 
                                     my_cuCsubf(my_cuCmulf(cuConjf(aMy), exp_m), aMy));             // -alpha_y * (M - conjM * e^(-2iv0t)) / 4
        H[DIM_H * DIM_H * idx + 3] = my_cuCaddf(H[DIM_H * DIM_H * idx + 3], 
                                     make_cuFloatComplex(__fmul_rn(0.5f, (v[idx] - v0)), 0.0f));    // (1/2)(v - v0)

        // -<Mx>Ix (only off diagonal terms)
        H[DIM_H * DIM_H * idx + 1] = my_cuCsubf(H[DIM_H * DIM_H * idx + 1], // ...
                                     my_cuCaddf(my_cuCmulf(aMx, exp_p), cuConjf(aMx)));  // -alpha_x * (conjM + M * e^(2iv0t)) / 4 
        H[DIM_H * DIM_H * idx + 2] = my_cuCsubf(H[DIM_H * DIM_H * idx + 2], // ...
                                     my_cuCaddf(my_cuCmulf(cuConjf(aMx), exp_m), aMx));  // -alpha_x * (M + conjM * e^(-2iv0t)) / 4

        // -<Mz>Iz (only diagonal terms)
        H[DIM_H * DIM_H * idx] = my_cuCsubf(H[DIM_H * DIM_H * idx], aMz);          // -(1/2) * alpha_z * Mz
        H[DIM_H * DIM_H * idx + 3] = my_cuCaddf(H[DIM_H * DIM_H * idx + 3], aMz);  // (1/2) * alpha_z * Mz

    }
}
*/

// MAGNUS HAMILTONIAN
__global__ void calc_H(float alpha_x, float alpha_y, float alpha_z, float t, float* v, float dt, int nf, cuFloatComplex* M_loc, cuFloatComplex* M_loc_z, cuFloatComplex* H)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        
        // DOES NOT MULTIPLY BY THE PERIOD T OR BY FACTOR i BECAUSE THIS HAPPENS IN THE LIOUVILLE CONVERSION
        // DISSIPATION TERM IS ADDED IN LATER, DURING LIOUVILLE TRANSITION
        // THESE TERMS ARE STILL ACCURATE WITH DISSIPATION BUT THE DISS TERM GETS MIXED UP WITH ALPHA, OMEGA, ETC

        // initialize to zero
        H[DIM_H * DIM_H * idx] = make_cuFloatComplex(0,0);
        H[DIM_H * DIM_H * idx + 1] = make_cuFloatComplex(0,0);
        H[DIM_H * DIM_H * idx + 2] = make_cuFloatComplex(0,0);
        H[DIM_H * DIM_H * idx + 3] = make_cuFloatComplex(0,0);

        // FIRST ORDER TERM //

            // terms we need, and ignoring gammas because they all cancel out since everything is in units 1/gamma
            float dv = __fsub_rn(v[idx], v0);
            float wx = cuCrealf(M_loc[idx]);
            float wy = cuCimagf(M_loc[idx]);
            float wz = cuCrealf(M_loc_z[idx]); // should be real regardless
            float alpha = __fadd_rn(alpha_x, alpha_y);
            float dv_az = __fadd_rn(dv, __fmul_rn(alpha_z, wz));
            cuFloatComplex wp = M_loc[idx];
            cuFloatComplex rp5 = make_cuFloatComplex(0.5f, 0.0f); // one half as complex (real)
            cuFloatComplex ip5 = make_cuFloatComplex(0.0f, 0.5f); // one half as complex (imag)   

            // the coefficients
            cuFloatComplex Az = make_cuFloatComplex(__fmul_rn(-1.0f, dv_az), 0.0f);
            cuFloatComplex Ax = make_cuFloatComplex(__fmul_rn(alpha, __fmul_rn(-0.5f, wx)), 0.0f);
            cuFloatComplex Ay = make_cuFloatComplex(__fmul_rn(alpha, __fmul_rn(-0.5f, wy)), 0.0f);

            // multiply z term by 1/2 and add (11), subtract (22)
            H[DIM_H * DIM_H * idx]     = my_cuCaddf(H[DIM_H * DIM_H * idx], my_cuCmulf(rp5, Az)); // (1/2)*(dv + Az)
            H[DIM_H * DIM_H * idx + 3] = my_cuCsubf(H[DIM_H * DIM_H * idx + 3], my_cuCmulf(rp5, Az)); // -(1/2)*(dv + Az)
            
            // multiply x term by 1/2 and add (12), (21)
            H[DIM_H * DIM_H * idx + 1] = my_cuCaddf(H[DIM_H * DIM_H * idx + 1], my_cuCmulf(rp5, Ax)); // (1/2)*Ax
            H[DIM_H * DIM_H * idx + 2] = my_cuCaddf(H[DIM_H * DIM_H * idx + 2], my_cuCmulf(rp5, Ax)); // (1/2)*Ax

            // multiply y term by i/2 and sub (12), add (21)
            H[DIM_H * DIM_H * idx + 1] = my_cuCsubf(H[DIM_H * DIM_H * idx + 1], my_cuCmulf(ip5, Ay)); // -(i/2)*Ay
            H[DIM_H * DIM_H * idx + 2] = my_cuCaddf(H[DIM_H * DIM_H * idx + 2], my_cuCmulf(ip5, Ay)); // (i/2)*Ay

        // SECOND ORDER TERM //

            // terms we need
            float eta = 0;
            if (alpha != 0)
            {
                eta = __fdiv_rn(__fsub_rn(alpha_x, alpha_y), alpha); // anisotropy factor
            }
            float alpha2 = __fmul_rn(alpha, alpha); // alpha squared
            cuFloatComplex wp2 = my_cuCmulf(wp, wp); // wp * wp
            cuFloatComplex wp2bar = my_cuCmulf(cuConjf(wp), cuConjf(wp)); // conj(wp) * conj(wp)
            cuFloatComplex wpwpbar = my_cuCmulf(wp, cuConjf(wp)); // wp * conj(wp)
            cuFloatComplex Pw = my_cuCaddf(wp2, my_cuCsubf(wp2bar, my_cuCmulf(make_cuFloatComplex(eta, 0.0f), wpwpbar))); // polynomial in z term

            // the coefficients
            cuFloatComplex Bz = my_cuCdivf(my_cuCmulf(Pw, make_cuFloatComplex(__fmul_rn(alpha2, eta), 0.0f)), make_cuFloatComplex(__fmul_rn(16.0f, v0), 0.0f));
            cuFloatComplex Bx = make_cuFloatComplex(__fmul_rn(__fdiv_rn(__fmul_rn(eta, __fmul_rn(alpha, dv_az)), __fmul_rn(4.0f, v0)), wx), 0.0f);
            cuFloatComplex By = make_cuFloatComplex(__fmul_rn(-1.0f, __fmul_rn(__fdiv_rn(__fmul_rn(eta, __fmul_rn(alpha, dv_az)), __fmul_rn(4.0f, v0)), wy)), 0.0f);

            // multiply z term by 1/2 and add (11), subtract (22)
            H[DIM_H * DIM_H * idx]     = my_cuCaddf(H[DIM_H * DIM_H * idx], my_cuCmulf(rp5, Bz)); // (1/2)*Bz
            H[DIM_H * DIM_H * idx + 3] = my_cuCsubf(H[DIM_H * DIM_H * idx + 3], my_cuCmulf(rp5, Bz)); // -(1/2)*Bz
            
            // multiply x term by 1/2 and add (12), (21)
            H[DIM_H * DIM_H * idx + 1] = my_cuCaddf(H[DIM_H * DIM_H * idx + 1], my_cuCmulf(rp5, Bx)); // (1/2)*Bx
            H[DIM_H * DIM_H * idx + 2] = my_cuCaddf(H[DIM_H * DIM_H * idx + 2], my_cuCmulf(rp5, Bx)); // (1/2)*Bx

            // multiply y term by i/2 and sub (12), add (21)
            H[DIM_H * DIM_H * idx + 1] = my_cuCsubf(H[DIM_H * DIM_H * idx + 1], my_cuCmulf(ip5, By)); // -(i/2)*By
            H[DIM_H * DIM_H * idx + 2] = my_cuCaddf(H[DIM_H * DIM_H * idx + 2], my_cuCmulf(ip5, By)); // (i/2)*By

        // THIRD ORDER TERM //

            // terms we need, broken up
            cuFloatComplex Pw2 = my_cuCsubf(my_cuCmulf(cuConjf(wp), cuConjf(wp)), my_cuCaddf(my_cuCmulf(wp, wp), my_cuCmulf(my_cuCmulf(wp, cuConjf(wp)), make_cuFloatComplex(eta, 0.0f))));
            cuFloatComplex Qw = my_cuCsubf(my_cuCmulf(wp, wp), my_cuCmulf(my_cuCmulf(wp, cuConjf(wp)), make_cuFloatComplex(__fmul_rn(2.0f, eta), 0.0f)));
            cuFloatComplex d_p = make_cuFloatComplex(__fmul_rn(__fmul_rn(v0, v0), 128.0f), 0.0f);
            cuFloatComplex d_z = make_cuFloatComplex(__fmul_rn(__fmul_rn(v0, v0), 32.0f), 0.0f);
            cuFloatComplex n_p1 = my_cuCmulf(make_cuFloatComplex(__fmul_rn(alpha, eta), 0.0f), make_cuFloatComplex(__fmul_rn(16.0f, __fmul_rn(dv_az, dv_az)), 0.0f));
            cuFloatComplex n_p2 = my_cuCmulf(make_cuFloatComplex(__fmul_rn(alpha, eta), 0.0f), my_cuCmulf(make_cuFloatComplex(__fmul_rn(2.0f, alpha2), 0.0f), Pw2));
            cuFloatComplex n_p3 = my_cuCmulf(make_cuFloatComplex(__fmul_rn(alpha, eta), 0.0f), my_cuCmulf(make_cuFloatComplex(__fmul_rn(eta, alpha2), 0.0f), Qw));
            cuFloatComplex n_z1 = make_cuFloatComplex(__fmul_rn(-1.0f, __fmul_rn(alpha2, dv_az)), 0.0f);
            cuFloatComplex n_z2 = my_cuCaddf(my_cuCmulf(cuConjf(wp), cuConjf(wp)), Qw);

            // actual terms
            cuFloatComplex Cz = my_cuCdivf(my_cuCmulf(n_z1, n_z2), d_z);
            cuFloatComplex Cx = my_cuCdivf(my_cuCaddf(n_p1, my_cuCsubf(n_p3, n_p2)), d_p);
            cuFloatComplex Cy = my_cuCmulf(make_cuFloatComplex(-1.0f, 0.0), my_cuCdivf(my_cuCaddf(n_p1, my_cuCaddf(n_p3, n_p2)), d_p));

            // multiply z term by 1/2 and add (11), subtract (22)
            H[DIM_H * DIM_H * idx]     = my_cuCaddf(H[DIM_H * DIM_H * idx], my_cuCmulf(rp5, Cz)); // (1/2)*Cz
            H[DIM_H * DIM_H * idx + 3] = my_cuCsubf(H[DIM_H * DIM_H * idx + 3], my_cuCmulf(rp5, Cz)); // -(1/2)*Cz
            
            // multiply x term by 1/2 and add (12), (21)
            H[DIM_H * DIM_H * idx + 1] = my_cuCaddf(H[DIM_H * DIM_H * idx + 1], my_cuCmulf(rp5, Cx)); // (1/2)*Cx
            H[DIM_H * DIM_H * idx + 2] = my_cuCaddf(H[DIM_H * DIM_H * idx + 2], my_cuCmulf(rp5, Cx)); // (1/2)*Cx

            // multiply y term by i/2 and sub (12), add (21)
            H[DIM_H * DIM_H * idx + 1] = my_cuCsubf(H[DIM_H * DIM_H * idx + 1], my_cuCmulf(ip5, Cy)); // -(i/2)*Cy
            H[DIM_H * DIM_H * idx + 2] = my_cuCaddf(H[DIM_H * DIM_H * idx + 2], my_cuCmulf(ip5, Cy)); // (i/2)*Cy

    }
}

__global__ void calc_M_eval_2D(cuFloatComplex* rho, cuFloatComplex* M_eval, cuFloatComplex* Mz_eval, int nf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        M_eval[idx] = rho[DIM_H * DIM_H * idx + 2]; // tr[(0 1; 0 0)(a11 a12; a21 a22)] = a21, aka element 3 in the array or index 2
        Mz_eval[idx] = my_cuCmulf(make_cuFloatComplex(0.5, 0), my_cuCsubf(rho[DIM_H * DIM_H * idx], rho[DIM_H * DIM_H * idx + 3])); // tr[(1/2)(1 0; 0 -1)(a11 a12; a21 a22)] = (1/2)(a11 - a22)
    }
}

__global__ void calc_M(cuFloatComplex* M_eval, cuFloatComplex* Mz_eval, cuFloatComplex* M, cuFloatComplex* Mz, int t_idx, int nf)
{
    M[t_idx] = make_cuFloatComplex(0, 0);
    Mz[t_idx] = make_cuFloatComplex(0, 0);
    float P = 1.0 / nf;

    for (int m_idx = 0; m_idx < nf; m_idx++)
    {
    
        cuFloatComplex cP = make_cuFloatComplex(P, 0);
        M[t_idx] = my_cuCaddf(M[t_idx], my_cuCmulf(cP, M_eval[m_idx]));
        Mz[t_idx] = my_cuCaddf(Mz[t_idx], my_cuCmulf(cP, Mz_eval[m_idx]));

    }
}

__global__ void time_propagate_liouville(cuFloatComplex* rho, cuFloatComplex* rho_T, cuFloatComplex* U, int nf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        // propagate
        for (int i = 0; i < DIM_L; i++)
        {
            rho_T[DIM_L * idx + i] = make_cuFloatComplex(0, 0);
            for (int j = 0; j < DIM_L; j++)
            {
                rho_T[DIM_L * idx + i] = my_cuCaddf(rho_T[DIM_L * idx + i], my_cuCmulf(U[DIM_L * DIM_L * idx + DIM_L * i + j], rho[DIM_L * idx + j]));
            }
        }

        // overwrite rho
        for (int i = 0; i < DIM_L; i++)
        {
            rho[DIM_L * idx + i] = rho_T[DIM_L * idx + i];
        }
    }
}

__global__ void time_propagate_hilbert(cuFloatComplex* rho, cuFloatComplex* rho_T, cuFloatComplex* U, int nf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        for (int i = 0; i < DIM_H; i++)
        {
            for (int j = 0; j < DIM_H; j++)
            {
                rho_T[DIM_H * DIM_H * idx + DIM_H * i + j] = make_cuFloatComplex(0, 0);
                for (int k = 0; k < DIM_H; k++)
                {
                    for (int l = 0; l < DIM_H; l++)
                    {
                        // rho_ij = U_ik x rho_kl x conj(U_lj)
                        rho_T[DIM_H * DIM_H * idx + DIM_H * i + j] = my_cuCaddf(rho_T[DIM_H * DIM_H * idx + DIM_H * i + j], // ...
                            my_cuCmulf(U[DIM_H * DIM_H * idx + DIM_H * i + k], my_cuCmulf( // ...
                                rho[DIM_H * DIM_H * idx + DIM_H * k + l], cuConjf(U[DIM_H * DIM_H * idx + DIM_H * l + j]))));
                    }
                }
            }
        }
        for (int i = 0; i < DIM_H; i++)
        {
            for (int j = 0; j < DIM_H; j++)
            {
                rho[DIM_H * DIM_H * idx + DIM_H * i + j] = rho_T[DIM_H * DIM_H * idx + DIM_H * i + j];
            }
        }
    }
}

__global__ void pulse(cuFloatComplex* rho, cuFloatComplex* rho_T, cuFloatComplex* pulse_op, int nf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        for (int i = 0; i < DIM_H; i++)
        {
            for (int j = 0; j < DIM_H; j++)
            {
                rho_T[DIM_H * DIM_H * idx + DIM_H * i + j] = make_cuFloatComplex(0, 0);
                for (int k = 0; k < DIM_H; k++)
                {
                    for (int l = 0; l < DIM_H; l++)
                    {
                        // rho_ij = U_ik x rho_kl x conj(U_jl) <-- conj transpose so U*_lj = conj(U_jl)
                        rho_T[DIM_H * DIM_H * idx + DIM_H * i + j] = my_cuCaddf(rho_T[DIM_H * DIM_H * idx + DIM_H * i + j], // ...
                            my_cuCmulf(pulse_op[DIM_H * i + k], my_cuCmulf(rho[DIM_H * DIM_H * idx + DIM_H * k + l], cuConjf(pulse_op[DIM_H * j + l]))));
                    }
                }
            }
        }

        for (int i = 0; i < DIM_H; i++)
        {
            for (int j = 0; j < DIM_H; j++)
            {
                rho[DIM_H * DIM_H * idx + DIM_H * i + j] = rho_T[DIM_H * DIM_H * idx + DIM_H * i + j];
            }
        }
    }
}

__global__ void ham_hilbert_to_liouville(cuFloatComplex* H, cuFloatComplex* H_L, float dt, float gamma1, float gamma2, float gamma3, int nf, float alpha_x, float alpha_y, cuFloatComplex* M_loc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nf)
    {
        // some values i'll need1
        cuFloatComplex idt = make_cuFloatComplex(0, -dt);
        cuFloatComplex temp_L;
        cuFloatComplex temp_R;

        // H x I, I x H
        for (int i = 0; i < DIM_H; i++)
        {
            for (int j = 0; j < DIM_H; j++)
            {
                for (int k = 0; k < DIM_H; k++)
                {
                    for (int l = 0; l < DIM_H; l++)
                    {
                        int block_idx = DIM_L * DIM_H * i + DIM_H * j;
                        int sub_idx = block_idx + DIM_L * k + l;

                        // H x I
                        if (k == l)
                        {
                            temp_R = H[DIM_H * DIM_H * idx + DIM_H * i + j];   
                        }
                        else
                        {
                            temp_R = make_cuFloatComplex(0, 0);
                        }

                        // I x conj(H)
                        if (i == j)
                        {
                            temp_L = cuConjf(H[DIM_H * DIM_H * idx + DIM_H * k + l]);
                        }
                        else
                        {
                            temp_L = make_cuFloatComplex(0, 0);
                        }

                        H_L[DIM_L * DIM_L * idx + sub_idx] = my_cuCmulf(idt, my_cuCsubf(temp_R, temp_L));
                    }
                }
            }
        }

        // FIRST ORDER DISSIPATION

            // add in dissipation (gamma1)
            H_L[DIM_L * DIM_L * idx] = my_cuCsubf(H_L[DIM_L * DIM_L * idx], make_cuFloatComplex(__fmul_rn(gamma1, dt), 0));
            H_L[DIM_L * DIM_L * idx + 3] = my_cuCaddf(H_L[DIM_L * DIM_L * idx + 3], make_cuFloatComplex(__fmul_rn(gamma1, dt), 0));
            H_L[DIM_L * DIM_L * idx + 5] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 5], make_cuFloatComplex(__fdiv_rn(__fmul_rn(gamma1, dt), 2.0f), 0));
            H_L[DIM_L * DIM_L * idx + 10] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 10], make_cuFloatComplex(__fdiv_rn(__fmul_rn(gamma1, dt), 2.0f), 0));

            // add in dissipation (gamma2)
            H_L[DIM_L * DIM_L * idx + 5] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 5], make_cuFloatComplex(__fdiv_rn(__fmul_rn(gamma2, dt), 2.0f), 0));
            H_L[DIM_L * DIM_L * idx + 10] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 10], make_cuFloatComplex(__fdiv_rn(__fmul_rn(gamma2, dt), 2.0f), 0));
            H_L[DIM_L * DIM_L * idx + 12] = my_cuCaddf(H_L[DIM_L * DIM_L * idx + 12], make_cuFloatComplex(__fmul_rn(gamma2, dt), 0));
            H_L[DIM_L * DIM_L * idx + 15] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 15], make_cuFloatComplex(__fmul_rn(gamma2, dt), 0));

            // add in the dissipation (gamma3)
            H_L[DIM_L * DIM_L * idx + 5] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 5], make_cuFloatComplex(__fdiv_rn(__fmul_rn(gamma3, dt), 2.0f), 0));
            H_L[DIM_L * DIM_L * idx + 10] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 10], make_cuFloatComplex(__fdiv_rn(__fmul_rn(gamma3, dt), 2.0f), 0));

        // SECOND ORDER DISSIPATION

            // terms we need
            float alpha = __fadd_rn(alpha_x, alpha_y);
            float eta = 0;
            if (alpha != 0)
            {
                eta = __fdiv_rn(__fsub_rn(alpha_x, alpha_y), alpha); // anisotropy factor
            }
            cuFloatComplex w = M_loc[idx];
            cuFloatComplex wbar = cuConjf(w);
            cuFloatComplex G1 = make_cuFloatComplex(__fadd_rn(gamma1, __fadd_rn(gamma2, gamma3)), 0.0f); // gamma1 + gamma2 + gamma3 (elements 21, 31, 24, 34)
            cuFloatComplex G2 = make_cuFloatComplex(__fadd_rn(__fmul_rn(3.0f, gamma1), __fsub_rn(gamma3, gamma2)), 0.0f); // 3gamma1 + (gamma3 - gamma2) (elements 12, 13)
            cuFloatComplex G3 = make_cuFloatComplex(__fsub_rn(gamma1, __fadd_rn(__fmul_rn(3.0f, gamma2), gamma3)), 0.0f); // gamma1 - (3gamma2 + gamma3) (elements 42, 43)
            cuFloatComplex coef = make_cuFloatComplex(__fmul_rn(dt, __fdiv_rn(__fmul_rn(alpha, eta), __fmul_rn(16.0f, v0))), 0.0f); // dt * alpha eta / 16 B0

            // matrix elements
            cuFloatComplex M12 = my_cuCmulf(coef, my_cuCmulf(G2, my_cuCmulf(make_cuFloatComplex(-1.0f, 0.0f), wbar)));
            cuFloatComplex M13 = cuConjf(M12);
            cuFloatComplex M21 = my_cuCmulf(coef, my_cuCmulf(G1, my_cuCmulf(make_cuFloatComplex(-1.0f, 0.0f), w)));
            cuFloatComplex M31 = cuConjf(M21);
            cuFloatComplex M24 = my_cuCmulf(coef, my_cuCmulf(G1, my_cuCmulf(make_cuFloatComplex(1.0f, 0.0f), w)));
            cuFloatComplex M34 = cuConjf(M24);
            cuFloatComplex M42 = my_cuCmulf(coef, my_cuCmulf(G3, my_cuCmulf(make_cuFloatComplex(-1.0f, 0.0f), wbar)));
            cuFloatComplex M43 = cuConjf(M42);

            // assign
            H_L[DIM_L * DIM_L * idx + 1] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 1], M12);
            H_L[DIM_L * DIM_L * idx + 2] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 2], M13);
            H_L[DIM_L * DIM_L * idx + 4] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 4], M21);
            H_L[DIM_L * DIM_L * idx + 7] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 7], M24);
            H_L[DIM_L * DIM_L * idx + 8] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 8], M31);
            H_L[DIM_L * DIM_L * idx + 11] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 11], M34);
            H_L[DIM_L * DIM_L * idx + 13] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 13], M42);
            H_L[DIM_L * DIM_L * idx + 14] = my_cuCsubf(H_L[DIM_L * DIM_L * idx + 14], M43);

    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// special addition
__device__ static __inline__ cuFloatComplex my_cuCaddf(cuFloatComplex x, cuFloatComplex y)
{
    float re_z = __fadd_rn(cuCrealf(x), cuCrealf(y));
    float im_z = __fadd_rn(cuCimagf(x), cuCimagf(y));

    cuFloatComplex z = make_cuFloatComplex(re_z, im_z);
    return z;
}

// special subtraction
__device__ static __inline__ cuFloatComplex my_cuCsubf(cuFloatComplex x, cuFloatComplex y)
{
    float re_z = __fsub_rn(cuCrealf(x), cuCrealf(y));
    float im_z = __fsub_rn(cuCimagf(x), cuCimagf(y));

    cuFloatComplex z = make_cuFloatComplex(re_z, im_z);
    return z;
}

// special multiplication
__device__ static __inline__ cuFloatComplex my_cuCmulf(cuFloatComplex x, cuFloatComplex y)
{
    float re_z_L = __fmul_rn(cuCrealf(x), cuCrealf(y));
    float re_z_R = __fmul_rn(cuCimagf(x), cuCimagf(y));
    float re_z = __fsub_rn(re_z_L, re_z_R);

    float im_z_L = __fmul_rn(cuCrealf(x), cuCimagf(y));
    float im_z_R = __fmul_rn(cuCimagf(x), cuCrealf(y));
    float im_z = __fadd_rn(im_z_L, im_z_R);

    cuFloatComplex z = make_cuFloatComplex(re_z, im_z);
    return z;
}

// special multiplication
__device__ static __inline__ cuFloatComplex my_cuCdivf(cuFloatComplex x, cuFloatComplex y)
{
    // scaling
    float re_s = (float) fabs((double) cuCrealf(y));
    float im_s = (float) fabs((double) cuCimagf(y));
    float s = __fadd_rn(re_s, im_s);
    float q = __fdiv_rn(1.0f, s);

    // terms
    float ars = __fmul_rn(cuCrealf(x), q);
    float ais = __fmul_rn(cuCimagf(x), q);
    float brs = __fmul_rn(cuCrealf(y), q);
    float bis = __fmul_rn(cuCimagf(y), q);

    // second scaling
    float s_L = __fmul_rn(brs, brs);
    float s_R = __fmul_rn(bis, bis);
    s = __fadd_rn(s_L, s_R);
    q = __fdiv_rn(1.0f, s);

    // terms for final result
    float re_z_L = __fmul_rn(ars, brs);
    float re_z_R = __fmul_rn(ais, bis);
    float re_z = __fmul_rn(__fadd_rn(re_z_L, re_z_R), q);
    float im_z_L = __fmul_rn(ais, brs);
    float im_z_R = __fmul_rn(ars, bis);
    float im_z = __fmul_rn(__fsub_rn(im_z_L, im_z_R), q);

    // result
    cuFloatComplex z = make_cuFloatComplex(re_z, im_z);
    return z;

}

// WRITE TO FILE FUNCTION
void write_to_file_c(std::ofstream &output_name, cuFloatComplex* A, cuFloatComplex* h_A, int dim, int e_check, int t_idx, int num_steps)
{
    if (e_check == 1)
    {
        if (t_idx < num_steps)
        {
            // copy to host
            cudaMemcpy(h_A, A, dim*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
            for (int i = 0; i < dim; i++)
            {
                output_name << cuCrealf(h_A[i]) << " " << cuCimagf(h_A[i]) << " ";
            }
            output_name << "\n";
        }
    }
}

void write_to_file_f(std::ofstream &output_name, float* A, float* h_A, int dim, int e_check, int t_idx, int num_steps)
{
    if (e_check == 1)
    {
        if (t_idx < num_steps)
        {
            cudaMemcpy(h_A, A, dim*sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < dim; i++)
            {
                output_name << h_A[i] << " ";
            }
            output_name << "\n";
        }
    }
}

void start_clock(timespec begin, int time_exec)
{
    if (time_exec == 1)
    {
        clock_gettime (CLOCK_PROCESS_CPUTIME_ID, &begin);
    }
}
void end_clock(timespec begin, timespec end, std::ofstream &t_output, int time_exec)
{
    if (time_exec == 1)
    {
        clock_gettime (CLOCK_PROCESS_CPUTIME_ID, &end);
        int time = 1e9 * (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec);
        t_output << time << " ";
    }
}