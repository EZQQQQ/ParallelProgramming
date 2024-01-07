#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0

// Function prototypes
void oddEvenIteration(int localArray[], int temporaryBufferB[], int temporaryBufferC[],
                      int localSize, int currentPhase, int evenPartner, int oddPartner,
                      int myRank, int numProcesses, MPI_Comm communicator);

void mergeSplitLow(int localArray[], int tempBufferB[], int tempBufferC[],
                   int localSize, int sizeOfPartner);

void mergeSplitHigh(int localArray[], int tempBufferB[], int tempBufferC[],
                    int localSize, int sizeOfPartner);

void parallelOddEvenSort(int localList[], int localSize, int myRank, 
        int numProcesses, MPI_Comm communicator);

void sequentialOddEvenSort(int localList[], int localSize) {
    bool sorted = false;

    while (!sorted) {
        sorted = true;

        // Perform the odd phase
        for (int i = 1; i < localSize - 1; i += 2) {
            if (localList[i] > localList[i + 1]) {
                std::swap(localList[i], localList[i + 1]);
                sorted = false;
            }
        }

        // Perform the even phase
        for (int i = 0; i < localSize - 1; i += 2) {
            if (localList[i] > localList[i + 1]) {
                std::swap(localList[i], localList[i + 1]);
                sorted = false;
            }
        }
    }
} 

// Use odd-even sort to sort global list
void parallelOddEvenSort(int localList[], int localSize, int myRank, 
        int numProcesses, MPI_Comm communicator) {
    int currentPhase;
    int *temporaryBufferEven, *temporaryBufferOdd;
    int evenPartner;  /* phase is even or left-looking */
    int oddPartner;   /* phase is odd or right-looking */

    /* Temporary storage used in merge-split */
    temporaryBufferEven = (int*)malloc((localSize + 1) * sizeof(int));
    temporaryBufferOdd = (int*)malloc(localSize * sizeof(int));

    /* Find partners:  negative rank => do nothing during phase */
    if (myRank % 2 != 0) {
        evenPartner = myRank - 1;
        oddPartner = myRank + 1;
        if (oddPartner == numProcesses)
            oddPartner = -1; // Idle during odd phase
    } else {
        evenPartner = myRank + 1;
        if (evenPartner == numProcesses)
            evenPartner = -1; // Idle during even phase
        oddPartner = myRank - 1;
    }

    // Sort local list using sequentialOddEvenSort
    sequentialOddEvenSort(localList, localSize);

    for (currentPhase = 0; currentPhase < numProcesses; currentPhase++)
        oddEvenIteration(localList, temporaryBufferEven, temporaryBufferOdd, 
            localSize, currentPhase, evenPartner, oddPartner, myRank, numProcesses, communicator);

    free(temporaryBufferEven);
    free(temporaryBufferOdd);
}

// One iteration of Odd-even transposition sort
void oddEvenIteration(int localArray[], int temporaryBufferB[], int temporaryBufferC[],
                      int localSize, int currentPhase, int evenPartner, int oddPartner,
                      int myRank, int numProcesses, MPI_Comm communicator) {
    MPI_Status status;

    // Determine the size of the even partner's vector
    int sizeOfEvenPartner;
    if (evenPartner >= 0) {
        // Use MPI_Send and MPI_Recv to exchange size information
        MPI_Send(&localSize, 1, MPI_INT, evenPartner, 0, communicator);
        MPI_Recv(&sizeOfEvenPartner, 1, MPI_INT, evenPartner, 0, communicator, &status);
    }
    // Determine the size of the odd partner's vector
    int sizeOfOddPartner;
    if (oddPartner >= 0) {
        // Use MPI_Send and MPI_Recv to exchange size information
        MPI_Send(&localSize, 1, MPI_INT, oddPartner, 0, communicator);
        MPI_Recv(&sizeOfOddPartner, 1, MPI_INT, oddPartner, 0, communicator, &status);
    }

    if (currentPhase % 2 == 0) { /* Even phase, odd process <-> rank-1 */
        if (evenPartner >= 0) {
            // Adjust send and receive counts based on the sizes
            int sendCount = localSize;
            int recvCount = sizeOfEvenPartner;
            MPI_Sendrecv(localArray, localSize, MPI_INT, evenPartner, 0,
                         temporaryBufferB, sizeOfEvenPartner, MPI_INT, evenPartner, 0, communicator,
                         &status);
            if (myRank % 2 != 0)
                mergeSplitHigh(localArray, temporaryBufferB, temporaryBufferC, localSize, sizeOfEvenPartner);
            else
                mergeSplitLow(localArray, temporaryBufferB, temporaryBufferC, localSize, sizeOfEvenPartner);
        }
    } else { /* Odd phase, odd process <-> rank+1 */
        if (oddPartner >= 0) {
            // Adjust send and receive counts based on the sizes
            int sendCount = localSize;
            int recvCount = sizeOfOddPartner;
            // Use sizeOfOddPartner when calling MPI_Sendrecv
            MPI_Sendrecv(localArray, localSize, MPI_INT, oddPartner, 0,
                         temporaryBufferB, sizeOfOddPartner, MPI_INT, oddPartner, 0, communicator,
                         &status);
            if (myRank % 2 != 0)
                mergeSplitLow(localArray, temporaryBufferB, temporaryBufferC, localSize, sizeOfOddPartner);
            else
                mergeSplitHigh(localArray, temporaryBufferB, temporaryBufferC, localSize, sizeOfOddPartner);
        }
    }
}


// Merge the smallest local_n elements in localArray and tempBufferB into tempBufferC
void mergeSplitLow(int localArray[], int tempBufferB[], int tempBufferC[],
                   int localSize, int sizeOfPartner) {
    int indexA, indexB, indexC;

    indexA = 0;
    indexB = 0;
    indexC = 0;

    // Compare the smallest elements in localArray and tempBufferB
    while (indexC < localSize) {
        if (indexA < localSize && (indexB >= sizeOfPartner || localArray[indexA] <= tempBufferB[indexB])) {
            tempBufferC[indexC] = localArray[indexA];
            indexC++;
            indexA++;
        } else {
            tempBufferC[indexC] = tempBufferB[indexB];
            indexC++;
            indexB++;
        }
    }

    // Copy tempBufferC back into localArray.
    memcpy(localArray, tempBufferC, localSize * sizeof(int));
}

// Merge the largest local_n elements in localArray and tempBufferB into tempBufferC
void mergeSplitHigh(int localArray[], int tempBufferB[], int tempBufferC[],
                    int localSize, int sizeOfPartner) {
    int indexA, indexB, indexC;

    indexA = localSize - 1;
    indexB = localSize - 1;
    indexC = localSize - 1;

    // Compare the largest elements in localArray and tempBufferB
    while (indexC >= 0) {
        if (indexA >= 0 && (indexB < 0 || localArray[indexA] >= tempBufferB[indexB])) {
            tempBufferC[indexC] = localArray[indexA];
            indexC--;
            indexA--;
        } else {
            tempBufferC[indexC] = tempBufferB[indexB];
            indexC--;
            indexB--;
        }
    }

    // Copy tempBufferC back into localArray
    memcpy(localArray, tempBufferC, localSize * sizeof(int));
}

// Wrapper function to perform odd-even sort on a vector using MPI
void oddEvenSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    int size = vec.size();
    int remaining = size % numtasks;
    int local_size = size / numtasks + (taskid < remaining ? 1 : 0);

    std::vector<int> localVec(local_size);
    int* local_A = localVec.data();  // Use localVec instead of vec for local data
    MPI_Comm comm = MPI_COMM_WORLD;

    int* sendcounts = new int[numtasks];
    int* displs = new int[numtasks];
    int sum = 0;

    // Calculate the sendcounts and displacements for MPI_Scatterv
    for (int i = 0; i < numtasks; ++i) {
        sendcounts[i] = size / numtasks + (i < remaining ? 1 : 0);
        displs[i] = sum;
        sum += sendcounts[i];
    }

    // Distribute the vector to all processes
    MPI_Scatterv(vec.data(), sendcounts, displs, MPI_INT, localVec.data(), local_size, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Perform Sorting on local vector
    parallelOddEvenSort(local_A, local_size, taskid, numtasks, comm);

    // Barrier synchronization before checking the global sorted status
    MPI_Barrier(MPI_COMM_WORLD);

    // Allocate memory for the global vector in the master process
    int* global_A = nullptr;
    if (taskid == MASTER) {
        global_A = new int[local_size * numtasks];
    }

    // Gather sorted subvectors to the master process
    MPI_Gather(local_A, local_size, MPI_INT, global_A, local_size, MPI_INT, MASTER, comm);

    // Update the original vector on the master process
    if (taskid == MASTER) {
        for (int i = 0; i < vec.size(); ++i) {
            vec[i] = global_A[i];
        }
        delete[] global_A;
    }

    // Broadcast the sorted data to all processes
    MPI_Bcast(vec.data(), vec.size(), MPI_INT, MASTER, MPI_COMM_WORLD);

    delete[] sendcounts;
    delete[] displs;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n");
    }
    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    const int size = atoi(argv[1]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    oddEvenSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        std::cout << "Odd-Even Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;

        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}
