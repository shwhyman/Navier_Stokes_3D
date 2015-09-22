#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "algorithm.h"
#include "derivatives.h"

/***************************************************************************************************************************************/
int initialise_theta_function_field(double * U, double * V, double * W, double dx, double dy, double dz,
                                    int x_nodes, int y_nodes, int z_nodes, double A, double a, double b, double c)
/*
  Purpose:
    Sum the contribution of a given number of 'Gaussian Fields' located at the centre of neighbouring unit cels
  Modified:
    August 2014
  Notes: x, y, z nodes are the maximum node at which a gaussian will be placed.
*/
/***************************************************************************************************************************************/
{
    int i, j, k;

    int i_cent = (int)(0.5*(N_i + 1));
    int j_cent = (int)(0.5*(N_j + 1));
    int k_cent = (int)(0.5*(N_k + 1));

    double i_unit = ((double)N_i)*dx;
    double j_unit = ((double)N_j)*dy;
    double k_unit = ((double)N_k)*dz;

    for(i = 1; i < N_i + 1; i++)
    {
        for(j = 1; j < N_j + 1; j++)
        {
            for(k = 1; k < N_k + 1; k++)
            {

                double x_pos = ((double)(i - i_cent))*dx;
                double y_pos = ((double)(j - j_cent))*dy;
                double z_pos = ((double)(k - k_cent))*dz;

                get_gaussian_nodes(U, V, W, x_pos, i, y_pos, j, z_pos, k, i_unit, j_unit, k_unit,
                                   x_nodes, y_nodes, z_nodes, A, a, b, c);
            }
        }
    }

    return 0;
}

/******************************************************************************************************************************************************************************************/
int get_gaussian_nodes(double * U, double * V, double * W, double x_pos, int i_array, double y_pos, int j_array,
                       double z_pos, int k_array, double i_unit, double j_unit, double k_unit, int x_nodes, int y_nodes,
                       int z_nodes, double A, double a, double b, double c)
/*
  Purpose:
    Performs the summation described in the function above
  Modified:
    May 2015
  Notes:
    The commented code is for a few different fields which were tested before setling of the current one.
*/
/******************************************************************************************************************************************************************************************/
{
    int i, j, k;                                       //Now refers to gaussian node numbers

    for(i = -x_nodes; i < x_nodes + 1; i++)
    {
        for(j = -y_nodes; j < y_nodes + 1; j++)
        {
            for(k = -z_nodes; k < z_nodes + 1; k++)
            {
                double x = x_pos - (((double)i)*i_unit);
                double y = y_pos - (((double)j)*j_unit);
                double z = z_pos - (((double)k)*k_unit);

                double exp_x = exp((-1)*(pow(x, 2)/a));
                double exp_y = exp((-1)*(pow(y, 2)/b));
                double exp_z = exp((-1)*(pow(z, 2)/c));

                double expn = exp_x*exp_y*exp_z;

                double U_val = (-2)*A*(y/b)*(z/c)*exp_z*exp_y*exp_x;
                double V_val = A*(x/a)*(z/c)*exp_z*exp_x*exp_y;
                double W_val = A*(y/b)*(x/a)*exp_x*exp_y*exp_z;

                U[INDEX(i_array, j_array, k_array)] += U_val;
                V[INDEX(i_array, j_array, k_array)] += V_val;
                W[INDEX(i_array, j_array, k_array)] += W_val;

            }
        }
    }

    return 0;
}

/****************************************************************************************/
void init_pressure(double * P)
/*
  Purpose:
    Initialise the pressure fields
  Modified:
    August 2014
*/
/****************************************************************************************/
{
    int i;
    int ndof = ndof = (N_i + 2)*(N_j + 2)*(N_k + 2);

    for(i = 0; i < ndof; i++)
    {
        P[i] = 0.0;
    }
    return;
}

/******************************************************************************/
double * allocate_NULL()
/*
  Purpose:
    ALLOCATE_ARRAYS creates and zeros out the arrays U and U_NEW.
  Modified:
    August 19 2014
*/
/******************************************************************************/
{
    int i;
    int ndof;

    double * array;

    ndof = (N_i + 2)*(N_j + 2)*(N_k + 2);

    array = (double*)malloc(ndof*sizeof(double));

    for(i = 0; i < ndof; i++)
    {
        array[i] = 0.0;
    }

    return array;
}

/******************************************************************************************/
int set_F_G_and_H(double * U, double * V, double * W, double * F, double * G, double * H,
                  double dx, double dy, double dz, double Re, double dt, double g_x, double g_y, double g_z)
/*
  Purpose:
    Set the values of the F, G and H arrays
  Modified: August 19 2014
*/
/******************************************************************************************/
{
    int i, j, k;

    for(i = 1; i < N_i + 1; i++)
    {
        for(j = 1; j < N_j + 1; j++)
        {
            for(k = 1; k < N_k + 1; k++)
            {
                //---F---//

                double d2U_dx = partial_squared_wrt_x(U, dx, i, j, k);
                double d2U_dy = partial_squared_wrt_y(U, dy, i, j, k);
                double d2U_dz = partial_squared_wrt_z(U, dz, i, j, k);

                double dU2_dx = partial_U_Squared_wrt_x(U, dx, i, j, k);
                double dUV_dy = partial_UV_wrt_y(U, V, dy, i, j, k);
                double dUW_dz = partial_UW_wrt_z(U, W, dz, i, j, k);

                F[INDEX(i, j, k)] = U[INDEX(i, j, k)] + ((((d2U_dx + d2U_dy + d2U_dz)/Re) - dU2_dx - dUV_dy - dUW_dz + g_x)*dt);

                //---G---//

                double d2V_dx = partial_squared_wrt_x(V, dx, i, j, k);
                double d2V_dy = partial_squared_wrt_y(V, dy, i, j, k);
                double d2V_dz = partial_squared_wrt_z(V, dz, i, j, k);

                double dV2_dy = partial_V_Squared_wrt_y(V, dy, i, j, k);
                double dUV_dx = partial_UV_wrt_x(U, V, dx, i, j, k);
                double dVW_dz = partial_VW_wrt_z(V, W, dz, i, j, k);

                G[INDEX(i, j, k)] = V[INDEX(i, j, k)] + ((((d2V_dx + d2V_dy + d2V_dz)/Re) - dV2_dy - dUV_dx - dVW_dz + g_y)*dt);

                //---H---//

                double d2W_dx = partial_squared_wrt_x(W, dx, i, j, k);
                double d2W_dy = partial_squared_wrt_y(W, dy, i, j, k);
                double d2W_dz = partial_squared_wrt_z(W, dz, i, j, k);

                double dW2_dz = partial_W_Squared_wrt_z(W, dz, i, j, k);
                double dUW_dx = partial_UW_wrt_x(U, W, dx, i, j, k);
                double dVW_dy = partial_VW_wrt_y(V, W, dy, i, j, k);

                H[INDEX(i, j, k)] = W[INDEX(i, j, k)] + ((((d2W_dx + d2W_dy + d2W_dz)/Re) - dW2_dz - dUW_dx - dVW_dy + g_z)*dt);
            }
        }
    }

    return 0;
}

/**************************************************************************/
int set_RHS(double * R, double * F, double * G, double * H, double dx, double dy, double dz, double dt)
/*
  Purpose:
    Set the R array values
  Modified: August 19 2014
*/
/**************************************************************************/
{
    int i, j, k;

    for(i = 1; i < N_i + 1; i++)
    {
        for(j = 1; j < N_j + 1; j++)
        {
            for(k = 1; k < N_k + 1; k++)
            {
                int i_minus = i - 1;
                int j_minus = j - 1;
                int k_minus = k - 1;

                if(i == 1)
                {
                    i_minus = N_i;
                }
                if(j == 1)
                {
                    j_minus = N_j;
                }
                if(k == 1)
                {
                    k_minus = N_k;
                }

                R[INDEX(i, j, k)] = (((F[INDEX(i, j, k)] - F[INDEX(i_minus, j, k)])/dx) +
                                     ((G[INDEX(i, j, k)] - G[INDEX(i, j_minus, k)])/dy) +
                                     ((H[INDEX(i, j, k)] - H[INDEX(i, j, k_minus)])/dz))/dt;
            }
        }
    }

    return 0;
}

/***************************************************************************/
int updateField(double * U, double * V, double * W, double * F, double * G, double * H,
                double * P_new, double dx, double dy, double dz, double dt)
/*
  Purpose:
    Update the velocity field
  Modified: August 19 2014
*/
/***************************************************************************/
{
    int i, j, k;

    for(i = 1; i < N_i + 1; i++)
    {
        for(j = 1; j < N_j + 1; j++)
        {
            for(k = 1; k < N_k + 1; k++)
            {
                int i_plus = i + 1;
                int j_plus = j + 1;
                int k_plus = k + 1;

                if(i == N_i)
                {
                    i_plus = 1;
                }
                if(j == N_j)
                {
                    j_plus = 1;
                }
                if(k == N_k)
                {
                    k_plus = 1;
                }

                U[INDEX(i, j, k)] = F[INDEX(i, j, k)] - (dt/dx)*(P_new[INDEX(i_plus, j, k)] - P_new[INDEX(i, j, k)]);

                V[INDEX(i, j, k)] = G[INDEX(i, j, k)] - (dt/dy)*(P_new[INDEX(i, j_plus, k)] - P_new[INDEX(i, j, k)]);

                W[INDEX(i, j, k)] = H[INDEX(i, j, k)] - (dt/dz)*(P_new[INDEX(i, j, k_plus)] - P_new[INDEX(i, j, k)]);
            }
        }
    }

    return 0;
}

