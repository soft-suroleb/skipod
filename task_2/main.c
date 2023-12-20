#include <math.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define DOUBLE_SIZE 8
#define WORKERS 4

char file_path[] = "checkpoints";
int N = 1000;
double **A;
double maxeps = 0.1e-7, eps;
int itmax = 100, i, j, k, ll, shift;
MPI_Comm global_comm = MPI_COMM_WORLD;

int rank, size;
int startrow, lastrow, nrow;

MPI_Request req[4];
MPI_Status status[4];

void relax();
void init();
void verify();
void free_array();
void print_matrix();

static void error_handler(MPI_Comm *comm, int *err, ...) {
    int len;
    char errstr[MPI_MAX_ERROR_STRING];
    MPIX_Comm_shrink(global_comm, &global_comm);
    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_size(global_comm, &size);
    MPI_Error_string(*err, errstr, &len);
    printf("Rank %d / %d: Notified of error %s\n", rank, size, errstr);
    free_array();
    MPI_Barrier(global_comm);
    startrow = rank * N / WORKERS;
    lastrow = (rank + 1) * N / WORKERS - 1;
    nrow = lastrow - startrow + 1;
    MPI_Barrier(global_comm);
    init();
}

void print_matrix() {
    for (int r = 0; r < WORKERS; r++) {
        if (rank == r) {
            printf("rank %d:\n", rank);
            for (int t = 0; t < nrow; t++) {
                for (int k = 0; k < N; k++) {
                    printf("%5.1f ", A[t][k]);
                }
                printf("\n");
            }
        }
        MPI_Barrier(global_comm);
    }
    if (rank == WORKERS - 1) {
        printf("\n");
    }
    MPI_Barrier(global_comm);
}

void save_checkpoint() {
    MPI_File file;
    MPI_File_open(global_comm, file_path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    for (i = 0; i < nrow; i++) {
        MPI_File_write_at(file, DOUBLE_SIZE * N * (startrow + i), A[i], N, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(global_comm);
    MPI_File_close(&file);
}

void load_checkpoint() {
    MPI_File file;
    MPI_File_open(global_comm, file_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    for (int i = 0; i < nrow; i++) {
        MPI_File_read_at(file, DOUBLE_SIZE * N * (startrow + i), A[i], N, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(global_comm);
    MPI_File_close(&file);
}

int main(int argc, char **argv) {
    int it;
    MPI_Init(&argc, &argv);

    MPI_Errhandler errh;
    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_size(global_comm, &size);
    MPI_Comm_create_errhandler(error_handler, &errh);
    MPI_Comm_set_errhandler(global_comm, errh);
    MPI_Barrier(global_comm);

    startrow = rank * N / WORKERS;
    lastrow = (rank + 1) * N / WORKERS - 1;
    nrow = lastrow - startrow + 1;
    init();
    save_checkpoint();

    if (rank == 1) {
        raise(SIGKILL);
    }
    MPI_Barrier(global_comm);

    for (it = 1; it <= itmax; it++) {
        if (rank == 0){
            printf("Current iteration: %d\n", it);
        }
        eps = 0.;
        relax();
        if (eps < maxeps){
            break;
        }
    }
    verify();
    free_array();
    MPI_Finalize();
    return 0;
}

void init() {
    A = (double **)malloc(nrow * sizeof(double *));
    for (i = 0; i < nrow; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        for (j = 0; j < N; j++) {
            if ((i == 0 && rank == 0) || (i == nrow - 1 && rank == WORKERS - 1) || j == 0 || j == N - 1 || rank >= WORKERS) {
                A[i][j] = 0;
            } else {
                A[i][j] = 1 + startrow + i + j;
            }
        }
    }
}

void free_array() {
    for (i = 0; i < nrow; i++) {
        free(A[i]);
    }
    free(A);
}

void relax(){
    load_checkpoint();
    if (rank < WORKERS){
        if (rank != 0)
            MPI_Irecv(&A[0][0], N, MPI_DOUBLE, rank - 1, 1, global_comm, &req[0]);
        if (rank != WORKERS - 1)
            MPI_Isend(&A[nrow - 2][0], N, MPI_DOUBLE, rank + 1, 1, global_comm, &req[2]);
        if (rank != WORKERS - 1)
            MPI_Irecv(&A[nrow - 1][0], N, MPI_DOUBLE, rank + 1, 2, global_comm, &req[3]);
        if (rank != 0)
            MPI_Isend(&A[1][0], N, MPI_DOUBLE, rank - 1, 2, global_comm, &req[1]);
    }
    ll = 4;
    shift = 0; 
    if (rank == 0) {
        ll = 2;
        shift = 2;
    }
    if (rank == WORKERS - 1) {
        ll = 2;
    }
    if (rank >= WORKERS) {
        ll = 0;
        shift = 0;
    }
    if (WORKERS > 1) {
        MPI_Waitall(ll, &req[shift], &status[0]);
    }
    if (rank < WORKERS) {
        for (i = 1; i < nrow - 1; i++) {
            if (i == 1 && rank == 0 || i == nrow - 2 && rank == WORKERS - 1)
                continue;
            for (j = 1 + i % 2; j < N; j += 2) {
                double e = A[i][j];
                A[i][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4.;
                eps = Max(eps, fabs(e - A[i][j]));
            }
        }

        for (i = 1; i < nrow - 1; i++) {
            if (i == 1 && rank == 0 || i == nrow - 2 && rank == WORKERS - 1)
                continue;
            for (j = 1 + (i + 1) % 2; j < N - 1; j += 2) {
                double e = A[i][j];
                A[i][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4.;
                eps = Max(eps, fabs(e - A[i][j]));
            }
        }
    }
    save_checkpoint();
    double tmp;
    MPI_Allreduce(&eps, &tmp, 1, MPI_DOUBLE, MPI_MAX, global_comm);
    eps = tmp;
}

void verify() {
    double s = 0.;
    load_checkpoint();

    if (rank < WORKERS) {
        for (i = 0; i < nrow; i++)
            for (j = 0; j < N; j++)
                s += A[i][j] * (i + 1 + startrow) * (j + 1) / (N * N);
    }

    double tmp;
    MPI_Allreduce(&s, &tmp, 1, MPI_DOUBLE, MPI_SUM, global_comm);
    s = tmp;
    if (rank == 0)
        printf("  S = %f\n", s);
}
