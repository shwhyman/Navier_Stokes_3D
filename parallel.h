#ifndef PARALLEL_H_INCLUDED
#define PARALLEL_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

extern int N_i, N_j, N_k;

extern int * proc;			                    //Process indexed by vertex
extern int * i_min, * i_max;		            //Min, Max vertex indices of processes
extern int * left_proc, * right_proc;	        //Processes to left and right

extern int my_rank;

int INDEX(int i, int j, int k);

void make_domains(int num_procs);

void SOR(int num_procs, double * R, double * P, double * P_new, double dx, double dy, double dz, double omega);

void set_F_G_H_and_R(int num_procs, double * F, double * G, double * H, double * R, double * U, double * V, double * W,
                    double dx, double dy, double dz, double Re, double dt, double g_x, double g_y, double g_z);

void timestamp();

#endif // PARALLEL_H_INCLUDED
