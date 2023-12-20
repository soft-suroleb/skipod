
#include <mpi.h>
#include <iostream>
#include <unistd.h>
#include <time.h>
using namespace std;

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int parent = (rank - 1) / 2;
    int left = 2 * rank + 1;
    int right = 2 * rank + 2;

    int tmp = 0;
    if (rank != 0) {
        cout << "I'm " << rank << ", want marker from " << parent << endl;
        MPI_Recv(&tmp, 1, MPI_INT, parent, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    cout << "I'm " << rank << ", i have a marker" << endl;

    const char* file_path = "critical.txt";
    if (FILE * file = fopen(file_path, "r")) {
        throw runtime_error("bad resource sharing");
    } else {
        srand(time(NULL));
        cout << "I'm " << rank << ", i have started working" << endl;
        sleep(1 + rand() % 10);
        cout << "I'm " << rank << ", i have ended working" << endl;
        remove(file_path);
    }

    if (left < size) {
        MPI_Send(&tmp, 1, MPI_INT, left, 0, MPI_COMM_WORLD);
        MPI_Recv(&tmp, 1, MPI_INT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (right < size) {
        MPI_Send(&tmp, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
        MPI_Recv(&tmp, 1, MPI_INT, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank != 0) {
        MPI_Send(&tmp, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}