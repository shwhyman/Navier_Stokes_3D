#ifndef ALGORITHM_H_INCLUDED
#define ALGORITHM_H_INCLUDED

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

int initialise_theta_function_field(double * U, double * V, double * W, double dx, double dy, double dz,
                                    int x_nodes, int y_nodes, int z_nodes, double A, double a, double b, double c);

int get_gaussian_nodes(double * U, double * V, double * W, double x_pos, int i_array, double y_pos, int j_array,
                       double z_pos, int k_array, double i_unit, double j_unit, double k_unit, int x_nodes, int y_nodes,
                       int z_nodes, double A, double a, double b, double c);

void init_pressure(double * P);

double * allocate_NULL();

int set_F_G_and_H(double * U, double * V, double * W, double * F, double * G, double * H,
                  double dx, double dy, double dz, double Re, double dt, double g_x, double g_y, double g_z);

int set_RHS(double * R, double * F, double * G, double * H, double dx, double dy, double dz, double dt);

int updateField(double * U, double * V, double * W, double * F, double * G, double * H,
                double * P_new, double dx, double dy, double dz, double dt);
#endif // ALGORITHM_H_INCLUDED
