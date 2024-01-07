//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Quick Sort with MPI
//


#include <iostream>
#include <vector>
#include <mpi.h>
#include <algorithm>
#include "../utils.hpp"

#define MASTER 0

int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }

    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

void sequentialquickSort(std::vector<int>& vec, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(vec, low, high);
        sequentialquickSort(vec, low, pivotIndex - 1);
        sequentialquickSort(vec, pivotIndex + 1, high);
    }
}

void quickSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    int size = vec.size();
    // Calculate the size of each block
    int block_size = size / numtasks;
    // Determine the starting index for the current process
    int start = taskid * block_size;
    // Determine the ending index for the current process
    int end = (taskid == numtasks - 1) ? size - 1 : start + block_size - 1;

    // Broadcast the vector to all processes
    MPI_Bcast(vec.data(), size, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Each process sorts its portion of the vector
    sequentialquickSort(vec, start, end);

    // Gather the sorted portions to the master process
    std::vector<int> sorted_vec(size);
    MPI_Gather(vec.data() + start, block_size, MPI_INT, sorted_vec.data(), block_size, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (taskid == MASTER) {
        // Merge the sorted segments received from each process
        std::vector<int> merged_vec(size);
        int* indices = new int[numtasks];

        // Initialize indices to the starting index of each process's sorted portion
        for (int i = 0; i < numtasks; i++) {
            indices[i] = i * block_size;
        }

        // Iterate over the elements of the merged vector
        for (int i = 0; i < size; i++) {
            int min_val = std::numeric_limits<int>::max();
            int min_idx = -1;

            // Iterate over each process's sorted portion
            for (int j = 0; j < numtasks; j++) {
                // Check if the current index is within bounds and the value is smaller than min_val
                if (indices[j] < (j + 1) * block_size && sorted_vec[indices[j]] < min_val) {
                    min_val = sorted_vec[indices[j]]; // Update min_val with the smaller value
                    min_idx = j; // Update min_idx with the index of the smaller value
                }
            }

            merged_vec[i] = min_val; // Assign the smallest value to the merged vector
            indices[min_idx]++; // Move to the next index in the selected process's sorted portion
        }

        // Copy the merged result back to the original vector
        vec = merged_vec;
    }
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
            );
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

    quickSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Quick Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}