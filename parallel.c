#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "parallel.h"
#include "derivatives.h"

/************************************************************************************/
inline int INDEX(int i, int j, int k)
/*
  Purpose:
    Contiguous memory layout
  Modified:
    May 2015
*/
/************************************************************************************/
{
    int lin_index;

    lin_index = ((((N_i + 2)*(i)+(j))*(N_j + 2)) + k);

    return lin_index;
}

/*************************************************************************************/
void make_domains(int num_procs)
/*
  Purpose:
    MAKE_DOMAINS sets up the information defining the process domains.
  Modified:
    August 30 2014:  Accounts for periodic boundary conditions, and is adapted for 3D
  Parameters:
    Input, int NUM_PROCS, the number of processes.
*/
/**************************************************************************************/
{
    double d;
    double eps;
    int i;
    int p;
    double x_max;
    double x_min;

    //---ALLOCATE ARRAYS FOR PROCESS INFORMATION---//

    proc = (int*)malloc((N_i + 2)*sizeof(int));
    i_min = (int*)malloc(num_procs*sizeof(int));
    i_max = (int*)malloc(num_procs*sizeof(int));
    left_proc = (int*)malloc(num_procs*sizeof(int));
    right_proc = (int*)malloc(num_procs*sizeof(int));

    //---DIVIDE THE RANGE [(-eps+1)..(N_i+eps)] EVENLY AMONG PROCESSES---//

    eps = 0.0001;
    d = (N_i - 1.0 + 2.0*eps)/(double)num_procs;

    for (p = 0; p < num_procs; p++)
    {
        //---PROCESS DOMAIN---//

        x_min = -eps + 1.0 + (double)(p*d);
        x_max = x_min + d;

        //---IDENTIFY VERTICES---//

        for(i = 1; i <= N_i; i++)
        {
            if(x_min <= i && i < x_max)
            {
                proc[i] = p;
            }
        }
    }

    for(p = 0; p < num_procs; p++)
    {
        for(i = 1; i <= N_i; i++)       //Find the smallest vertex index in the domain.
        {
            if(proc[i] == p)
            {
                break;
            }
        }
        i_min[p] = i;

        for(i = N_i; 1 <= i; i--)        //Find the largest vertex index in the domain.
        {
            if(proc[i] == p)
            {
                break;
            }
        }
        i_max[p] = i + 1;

        //printf("  Process %d has i_min = %d, i_max = %d\n", p, i_min[p], i_max[p]);

        //---FIND PROCESSES TO THE LEFT AND RIGHT---//

        left_proc[p] = num_procs - 1;                   //Left and right boundaries are periodic.
        right_proc[p] = 0;

        if(proc[p] != -1)
        {
            if(1 < i_min[p] && i_min[p] <= N_i)
            {
                left_proc[p] = proc[i_min[p] - 1];
            }
            if( (0 < (i_max[p] - 1)) && ((i_max[p] - 1) < N_i))
            {
                right_proc[p] = proc[i_max[p]];
            }
        }

        if(my_rank == 0)
        {
            //printf("  Process %d has left = %d, right = %d\n", p, left_proc[p], right_proc[p]);
        }
    }
    return;
}

/***********************************************************************************************/
void SOR(int num_procs, double * R, double * P, double * P_new, double dx, double dy, double dz, double omega)
/*
  Purpose:
    SOR carries out the SOR iteration for the linear system.
  Modified:
    September 2014:  Accounts for periodic boundary conditions.
  Parameters:
    Input, int NUM_PROCS, the number of processes.
    Input, double R[(N_i+2)*(N_j+2)*(N_k+2)], the right hand side of the linear system.
*/
/************************************************************************************************/
{
    int i, j, k;
    MPI_Request request[4];
    int requests;
    MPI_Status status[4];

    //---UPDATE i-k GHOST LAYERS---//

    for(i = i_min[my_rank]; i < i_max[my_rank]; i++)
    {
        for(k = 0; k < N_k + 2; k++)
        {
            P[INDEX(i, 0, k)] = P[INDEX(i, N_j, k)];
            P[INDEX(i, N_j + 1, k)] = P[INDEX(i, 1, k)];
        }
    }

    //---UPDATE i-j GHOST LAYERS---//

    for(i = i_min[my_rank]; i < i_max[my_rank]; i++)
    {
        for(j = 0; j < N_j + 2; j++)
        {
            P[INDEX(i, j, 0)] = P[INDEX(i, j, N_k)];
            P[INDEX(i, j, N_k + 1)] = P[INDEX(i, j, 1)];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //---ADD SIDE (j-k) GHOST LAYERS USING NON-BLOCKING SEND AND RECEIVE---//

    requests = 0;

    if(left_proc[my_rank] >= 0 && left_proc[my_rank] < num_procs)
    {
        MPI_Irecv(P + INDEX(i_min[my_rank] - 1, 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  left_proc[my_rank], 0, MPI_COMM_WORLD,
                  request + requests++);

        MPI_Isend(P + INDEX(i_min[my_rank], 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  left_proc[my_rank], 1, MPI_COMM_WORLD,
                  request + requests++);
    }

    if(right_proc[my_rank] >= 0 && right_proc[my_rank] < num_procs)                //Check validity
    {
        MPI_Irecv(P + INDEX(i_max[my_rank], 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  right_proc[my_rank], 1, MPI_COMM_WORLD,
                  request + requests++);

        MPI_Isend(P + INDEX(i_max[my_rank] - 1, 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  right_proc[my_rank], 0, MPI_COMM_WORLD,
                  request + requests++);
    }

    MPI_Waitall(requests, request, status);       //Wait for all non-blocking communications to complete before updating boundaries.
    MPI_Barrier(MPI_COMM_WORLD);

    //---SOR UPDATE FOR ALL NODES---//

    for(i = i_min[my_rank]; i < i_max[my_rank]; i++)
    {
        for(j = 1; j < N_j + 1; j++)
        {
            for(k = 1; k < N_k + 1; k++)
            {
            P_new[INDEX(i, j, k)] = (P[INDEX(i, j, k)])*(1.0 - omega) + (omega/((2.0/(pow(dx, 2.0))) +
                                 (2.0/(pow(dy, 2.0))) + (2.0/(pow(dz, 2.0)))))*(((P[INDEX(i + 1, j, k)] +
                                  P_new[INDEX(i - 1, j, k)])/(pow(dx, 2.0))) + ((P[INDEX(i, j + 1, k)] +
                                  P_new[INDEX(i, j - 1, k)])/(pow(dy, 2.0))) +
                                ((P[INDEX(i, j, k + 1)] +
                                  P_new[INDEX(i, j, k - 1)])/(pow(dz, 2.0)))- R[INDEX(i, j, k)]);
            }
        }
    }

    return;
}

/*************************************************************************************************/
void set_F_G_H_and_R(int num_procs, double * F, double * G, double * H, double * R, double * U, double * V, double * W,
                    double dx, double dy, double dz, double Re, double dt, double g_x, double g_y, double g_z)
/*
  Purpose:
    Caculate the F, G and R fields in parallel
  Modifed:
    August 2014
*/
/*************************************************************************************************/
{
    int i, j, k;

    //---CALCULATE F, G, H BEFORE SWAPPING VALUES---//

    for(i = i_min[my_rank]; i < i_max[my_rank]; i++)
    {
        for (j = 1; j < N_j + 1; j++)
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

    //---UPDATE i-k GHOST LAYERS---//

    for(i = i_min[my_rank]; i < i_max[my_rank]; i++)
    {
        for(k = 0; k < N_k + 2; k++)
        {
            F[INDEX(i, 0, k)] = F[INDEX(i, N_j, k)];
            F[INDEX(i, N_j + 1, k)] = F[INDEX(i, 1, k)];

            G[INDEX(i, 0, k)] = G[INDEX(i, N_j, k)];
            G[INDEX(i, N_j + 1, k)] = G[INDEX(i, 1, k)];

            H[INDEX(i, 0, k)] = H[INDEX(i, N_j, k)];
            H[INDEX(i, N_j + 1, k)] = H[INDEX(i, 1, k)];
        }
    }

    //---UPDATE i-j GHOST LAYERS---//

    for(i = i_min[my_rank]; i < i_max[my_rank]; i++)
    {
        for(j = 0; j < N_j + 2; j++)
        {
            F[INDEX(i, j, 0)] = F[INDEX(i, j, N_k)];
            F[INDEX(i, j, N_k + 1)] = F[INDEX(i, j, 1)];

            G[INDEX(i, j, 0)] = G[INDEX(i, j, N_k)];
            G[INDEX(i, j, N_k + 1)] = G[INDEX(i, j, 1)];

            H[INDEX(i, j, 0)] = H[INDEX(i, j, N_k)];
            H[INDEX(i, j, N_k + 1)] = H[INDEX(i, j, 1)];
        }
    }

    //---ADD SIDE (j-k) GHOST LAYERS USING NON-BLOCKING SEND AND RECEIVE---//

    MPI_Request F_request[4];
    int F_requests = 0;
    MPI_Status F_status[4];

    MPI_Request G_request[4];
    int G_requests = 0;
    MPI_Status G_status[4];

    MPI_Request H_request[4];
    int H_requests = 0;
    MPI_Status H_status[4];

    if(left_proc[my_rank] >= 0 && left_proc[my_rank] < num_procs)
    {
        MPI_Irecv(F + INDEX(i_min[my_rank] - 1, 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  left_proc[my_rank], 0, MPI_COMM_WORLD,
                  F_request + F_requests++);

        MPI_Isend(F + INDEX(i_min[my_rank], 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  left_proc[my_rank], 1, MPI_COMM_WORLD,
                  F_request + F_requests++);

        MPI_Irecv(G + INDEX(i_min[my_rank] - 1, 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  left_proc[my_rank], 2, MPI_COMM_WORLD,
                  G_request + G_requests++);

        MPI_Isend(G + INDEX(i_min[my_rank], 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  left_proc[my_rank], 3, MPI_COMM_WORLD,
                  G_request + G_requests++);

        MPI_Irecv(H + INDEX(i_min[my_rank] - 1, 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  left_proc[my_rank], 4, MPI_COMM_WORLD,
                  H_request + H_requests++);

        MPI_Isend(H + INDEX(i_min[my_rank], 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  left_proc[my_rank], 5, MPI_COMM_WORLD,
                  H_request + H_requests++);
    }

    if(right_proc[my_rank] >= 0 && right_proc[my_rank] < num_procs)                //Check validity
    {
        MPI_Irecv(F + INDEX(i_max[my_rank], 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  right_proc[my_rank], 1, MPI_COMM_WORLD,
                  F_request + F_requests++);

        MPI_Isend(F + INDEX(i_max[my_rank] - 1, 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  right_proc[my_rank], 0, MPI_COMM_WORLD,
                  F_request + F_requests++);

        MPI_Irecv(G + INDEX(i_max[my_rank], 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  right_proc[my_rank], 3, MPI_COMM_WORLD,
                  G_request + G_requests++);

        MPI_Isend(G + INDEX(i_max[my_rank] - 1, 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  right_proc[my_rank], 2, MPI_COMM_WORLD,
                  G_request + G_requests++);

        MPI_Irecv(H + INDEX(i_max[my_rank], 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  right_proc[my_rank], 5, MPI_COMM_WORLD,
                  H_request + H_requests++);

        MPI_Isend(H + INDEX(i_max[my_rank] - 1, 0, 0), (N_j + 2)*(N_k + 2), MPI_DOUBLE,
                  right_proc[my_rank], 4, MPI_COMM_WORLD,
                  H_request + H_requests++);
    }

    MPI_Waitall(F_requests, F_request, F_status);       //Wait for all non-blocking communications to complete before updating boundaries.
    MPI_Waitall(G_requests, G_request, G_status);
    MPI_Waitall(H_requests, H_request, H_status);

    for(i = i_min[my_rank]; i < i_max[my_rank]; i++)
    {
        for (j = 1; j < N_j + 1; j++)
        {
            for(k = 1; k < N_k + 1; k++)
            {
                R[INDEX(i, j, k)] = (((F[INDEX(i, j, k)] - F[INDEX(i - 1, j, k)])/dx) +
                                     ((G[INDEX(i, j, k)] - G[INDEX(i, j - 1, k)])/dy) +
                                     ((H[INDEX(i, j, k)] - H[INDEX(i, j, k - 1)])/dz))/dt;
            }
        }
    }

    return;
}

/*************************************************************************************************/
void timestamp()
/*
  Purpose:
    TIMESTAMP prints the current YMDHMS date as a time stamp.
  Example:
    31 May 2001 09:45:54 AM
  Licensing:
    This code is distributed under the GNU LGPL license.
  Modified:
    24 September 2003
  Author:
    John Burkardt
  Parameters:
    None
*/
/*************************************************************************************************/
{
# define TIME_SIZE 40

    static char time_buffer[TIME_SIZE];
    const struct tm *tm;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

    printf("%s\n", time_buffer);

    return;
# undef TIME_SIZE
}

