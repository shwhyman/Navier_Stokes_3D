#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "derivatives.h"

/****************************************************************/
/*
THE FOLLOWING FUNCTIONS ARE USED TO CALCULATE THE DERIVATIVES
REQUIRED FOR SOLVING THE NAVIER STOKES EQUATIONS.
THE PURPOSE OF EACH IS INCLUDED AS THE FUNCTION NAME.
E.G., partial_V_Squared_wrt_y is used for dV^2/dy at the point (i,j).
*/
/****************************************************************/

double partial_U_Squared_wrt_x(double * U, double dx, int i, int j, int k)
{
    int i_plus = i + 1;
    int i_minus = i - 1;

    if(i == N_i)
    {
        i_plus = 1;
    }
    if(i == 1)
    {
        i_minus = N_i;
    }

    double first = pow(((U[INDEX(i, j, k)] + U[INDEX(i_plus, j, k)])*0.5), 2.0);
    double second = pow(((U[INDEX(i_minus, j, k)] + U[INDEX(i, j, k)])*0.5), 2.0);

    double derivative = (first - second)/dx;

    return derivative;
}

double partial_V_Squared_wrt_y(double * V, double dy, int i, int j, int k)
{
    int j_plus = j + 1;
    int j_minus = j - 1;

    if(j == N_j)
    {
        j_plus = 1;
    }
    if(j == 1)
    {
        j_minus = N_j;
    }

    double first = pow(((V[INDEX(i, j, k)] + V[INDEX(i, j_plus, k)])*0.5), 2.0);
    double second = pow(((V[INDEX(i, j_minus, k)] + V[INDEX(i, j, k)])*0.5), 2.0);

    double derivative = (first - second)/dy;

    return derivative;
}

double partial_W_Squared_wrt_z(double * W, double dz, int i, int j, int k)
{

    int k_plus = k + 1;
    int k_minus = k - 1;

    if(k == N_k)
    {
        k_plus = 1;
    }
    if(k == 1)
    {
        k_minus = N_k;
    }

    double first = pow(((W[INDEX(i, j, k)] + W[INDEX(i, j, k_plus)])*0.5), 2.0);
    double second = pow(((W[INDEX(i, j, k_minus)] + W[INDEX(i, j, k)])*0.5), 2.0);

    double derivative = (first - second)/dz;

    return derivative;
}

double partial_UV_wrt_x(double * U, double * V, double dx, int i, int j, int k)
{
    int i_plus = i + 1;
    int i_minus = i - 1;
    int j_plus = j + 1;

    if(i == N_i)
    {
        i_plus = 1;
    }
    if(i == 1)
    {
        i_minus = N_i;
    }
    if(j == N_j)
    {
        j_plus = 1;
    }

    double first = ((U[INDEX(i, j, k)] + U[INDEX(i, j_plus, k)])*0.5)*((V[INDEX(i, j, k)] + V[INDEX(i_plus, j, k)])*0.5);
    double second = ((U[INDEX(i_minus, j, k)] + U[INDEX(i_minus, j_plus, k)])*0.5)*((V[INDEX(i_minus, j, k)] + V[INDEX(i, j, k)])*0.5);

    double derivative = (first - second)/dx;

    return derivative;
}

double partial_UV_wrt_y(double * U, double * V, double dy, int i, int j, int k)
{
    int i_plus = i + 1;
    int j_plus = j + 1;
    int j_minus = j - 1;

    if(i == N_i)
    {
        i_plus = 1;
    }
    if(j == N_j)
    {
        j_plus = 1;
    }
    if(j == 1)
    {
        j_minus = N_j;
    }

    double first = ((U[INDEX(i, j_plus, k)] + U[INDEX(i, j, k)])*0.5)*((V[INDEX(i, j, k)] + V[INDEX(i_plus, j, k)])*0.5);
    double second = ((U[INDEX(i, j_minus, k)] + U[INDEX(i, j, k)])*0.5)*((V[INDEX(i, j_minus, k)] + V[INDEX(i_plus, j_minus, k)])*0.5);

    double derivative = (first - second)/dy;

    return derivative;
}

double partial_UW_wrt_x(double * U, double * W, double dx, int i, int j, int k)
{

    int i_plus = i + 1;
    int i_minus = i - 1;
    int k_plus = k + 1;
    int k_minus = k - 1;

    if(i == N_i)
    {
        i_plus = 1;
    }
    if(i == 1)
    {
        i_minus = N_i;
    }
    if(k == N_k)
    {
        k_plus = 1;
    }
    if(k == 1)
    {
        k_minus = N_k;
    }

    double first = ((U[INDEX(i, j, k_plus)] + U[INDEX(i, j, k)])*0.5)*((W[INDEX(i, j, k)] + W[INDEX(i_plus, j, k)])*0.5);
    double second = ((U[INDEX(i_minus, j, k)] + U[INDEX(i_minus, j, k_plus)])*0.5)*((W[INDEX(i_minus, j, k)] + W[INDEX(i, j, k)])*0.5);

    double derivative = (first - second)/dx;

    return derivative;

}

double partial_UW_wrt_z(double * U, double * W, double dz, int i, int j, int k)
{

    int i_plus = i + 1;
    int i_minus = i - 1;
    int k_plus = k + 1;
    int k_minus = k - 1;

    if(i == N_i)
    {
        i_plus = 1;
    }
    if(i == 1)
    {
        i_minus = N_i;
    }
    if(k == N_k)
    {
        k_plus = 1;
    }
    if(k == 1)
    {
        k_minus = N_k;
    }

    double first = ((U[INDEX(i, j, k_plus)] + U[INDEX(i, j, k)])*0.5)*((W[INDEX(i, j, k)] + W[INDEX(i_plus, j, k)])*0.5);
    double second = ((U[INDEX(i, j, k)] + U[INDEX(i, j, k_minus)])*0.5)*((W[INDEX(i, j, k)] + W[INDEX(i_plus, j, k_minus)])*0.5);

    double derivative;

    derivative = (first - second)/dz;

    return derivative;
}

double partial_VW_wrt_z(double * V, double * W, double dz, int i, int j, int k)
{
    int j_plus = j + 1;
    int j_minus = j - 1;
    int k_plus = k + 1;
    int k_minus = k - 1;

    if(j == N_j)
    {
        j_plus = 1;
    }
    if(j == 1)
    {
        j_minus = N_j;
    }
    if(k == N_k)
    {
        k_plus = 1;
    }
    if(k == 1)
    {
        k_minus = N_k;
    }

    double first = ((W[INDEX(i, j, k)] + W[INDEX(i, j_plus, k)])*0.5)*((V[INDEX(i, j, k_plus)] + V[INDEX(i, j, k)])*0.5);
    double second = ((W[INDEX(i, j, k_minus)] + W[INDEX(i, j_plus, k_minus)])*0.5)*((V[INDEX(i, j, k)] + V[INDEX(i, j, k_minus)])*0.5);

    double derivative = (first - second)/dz;

    return derivative;
}

double partial_VW_wrt_y(double * V, double * W, double dy, int i, int j, int k)
{
    int j_plus = j + 1;
    int j_minus = j - 1;
    int k_plus = k + 1;
    int k_minus = k - 1;

    if(j == N_j)
    {
        j_plus = 1;
    }
    if(j == 1)
    {
        j_minus = N_j;
    }
    if(k == N_k)
    {
        k_plus = 1;
    }
    if(k == 1)
    {
        k_minus = N_k;
    }

    double first = ((W[INDEX(i, j, k)] + W[INDEX(i, j_plus, k)])*0.5)*((V[INDEX(i, j, k_plus)] + V[INDEX(i, j, k)])*0.5);
    double second = ((V[INDEX(i, j_minus, k_plus)] + V[INDEX(i, j_minus, k)])*0.5)*((W[INDEX(i, j_minus, k)] + W[INDEX(i, j, k)])*0.5);

    double derivative = (first - second)/dy;

    return derivative;
}

double partial_squared_wrt_x(double * field, double dx, int i, int j, int k)
{

    int i_plus = i + 1;
    int i_minus = i - 1;

    if(i == N_i)
    {
        i_plus = 1;
    }
    if(i == 1)
    {
        i_minus = N_i;
    }

    double derivative = (field[INDEX(i_plus, j, k)] - 2*field[INDEX(i, j, k)] + field[INDEX(i_minus, j, k)])/(pow(dx, 2.0));

    return derivative;
}

double partial_squared_wrt_y(double * field, double dy, int i, int j, int k)
{

    int j_plus = j + 1;
    int j_minus = j - 1;

    if(j == N_j)
    {
        j_plus = 1;
    }
    if(j == 1)
    {
        j_minus = N_j;
    }

    double derivative = (field[INDEX(i, j_plus, k)] - 2*field[INDEX(i, j, k)] + field[INDEX(i, j_minus, k)])/(pow(dy, 2.0));

    return derivative;
}

double partial_squared_wrt_z(double * field, double dz, int i, int j, int k)
{

    int k_plus = k + 1;
    int k_minus = k - 1;

    if(k == N_k)
    {
        k_plus = 1;
    }
    if(k == 1)
    {
        k_minus = N_k;
    }

    double derivative = (field[INDEX(i, j, k_plus)] - 2*field[INDEX(i, j, k)] + field[INDEX(i, j, k_minus)])/(pow(dz, 2.0));

    return derivative;
}
