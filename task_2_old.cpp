#include <math.h>
#include <mpi.h>
#include <iostream>
#include <stdlib.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

int N;
double **A;
double maxeps = 0.1e-7, eps;
int itmax = 100, i, j, k, ll, shift;

int rank, size;
int startrow, lastrow, nrow;

MPI_Request req[4];
MPI_Status status[4];

void relax();
void init();
void verify();

int main(int argc, char **argv) {
    int it;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Barrier(MPI_COMM_WORLD);
    for (N = 1000; N < 10000; N += 1000) {
        startrow = rank * N / size;
        lastrow = (rank + 1) * N / size - 1;
        nrow = lastrow - startrow + 1;
        double starttime = 0;
        double time = 0;
        for (int t = 0; t < 3; t++) {
            init();
            starttime = MPI_Wtime();
            for (it = 1; it <= itmax; it++) {
                eps = 0.;
                relax();
                if (eps < maxeps)
                    break;
            }
            time += MPI_Wtime() - starttime;
            verify();
            for (i = 0; i < nrow; i++) {
                free(A[i]);
            }
            free(A);
        }
        if (rank == 0) {
            std::cout << "N: " << N << " time = " << time / 3 << std::endl;
        }
    }
    MPI_Finalize();
    return 0;
}

void init() {
    A = (double **)malloc(nrow * sizeof(double *));
    for (i = 0; i < nrow; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        for (j = 0; j < N; j++) {
            if (i == 0 | i == N - 1 || j == 0 || j == N - 1)
                A[i][j] = 0.;
            else
                A[i][j] = 1. + startrow + i + j;
        }
    }
}

void relax(){
    if (rank != 0)
        MPI_Irecv(&A[0][0], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &req[0]);
    if (rank != size - 1)
        MPI_Isend(&A[nrow - 2][0], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &req[2]);
    if (rank != size - 1)
        MPI_Irecv(&A[nrow - 1][0], N, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &req[3]);
    if (rank != 0)
        MPI_Isend(&A[1][0], N, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &req[1]);

    ll = 4;
    shift = 0; 
    if (rank == 0) {
        ll = 2;
        shift = 2;
    }
    if (rank == size - 1)
        ll = 2;
    if (size > 1) {
        MPI_Waitall(ll, &req[shift], &status[0]);
    }

    for (i = 1; i < nrow - 1; i++) {
        if (i == 1 && rank == 0 || i == nrow - 2 && rank == size - 1)
            continue;
        for (j = 1 + i % 2; j < N; j += 2) {
            double e = A[i][j];
            A[i][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4.;
            eps = Max(eps, fabs(e - A[i][j]));
        }
    }

    for (i = 1; i < nrow - 1; i++) {
        if (i == 1 && rank == 0 || i == nrow - 2 && rank == size - 1)
            continue;
        for (j = 1 + (i + 1) % 2; j < N - 1; j += 2) {
            double e = A[i][j];
            A[i][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4.;
            eps = Max(eps, fabs(e - A[i][j]));
        }
    }
    double tmp;
    MPI_Allreduce(&eps, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    eps = tmp;
}

void verify() {
    double s = 0.;
    for (i = 0; i < nrow; i++)
        for (j = 0; j < N; j++)
            s += A[i][j] * (i + 1 + startrow) * (j + 1) / (N * N);
    double tmp;
    MPI_Allreduce(&s, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    s = tmp;
    if (rank == 0)
        std::cout << "  S = " << s << std::endl;
}
