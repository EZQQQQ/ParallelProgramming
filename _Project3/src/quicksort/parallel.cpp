#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include "../utils.hpp"

// Threshold for switching to sequential quicksort
const int THRESHOLD = 1000;

// Partition function for quicksort
int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;

    // Iterate through the subarray and partition elements
    for (int j = low; j < high; ++j) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }

    // Swap pivot to its correct position
    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

// Recursive parallel quicksort function
void quickSort(std::vector<int> &vec, int low, int high) {
    if (low < high) {
        // Find pivot index through partitioning
        int pivot = partition(vec, low, high);

        // Parallelize only if the size is greater than the threshold
        if (high - low > THRESHOLD) {
#pragma omp task shared(vec)
            quickSort(vec, low, pivot - 1);

#pragma omp task shared(vec)
            quickSort(vec, pivot + 1, high);

#pragma omp taskwait
        } else {
            // If the size is smaller than the threshold, use sequential quicksort
            quickSort(vec, low, pivot - 1);
            quickSort(vec, pivot + 1, high);
        }
    }
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable threads_num vector_size\n"
        );
    }

    // Extract command-line arguments
    const int thread_num = atoi(argv[1]);
    const int size = atoi(argv[2]);
    const int seed = 4005;

    // Create random vector and clone for result verification
    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    // Measure execution time using chrono
    auto start_time = std::chrono::high_resolution_clock::now();

    // OpenMP parallel region with specified number of threads
#pragma omp parallel num_threads(thread_num)
    {
        // Single nowait pragma ensures only one thread executes the following block
#pragma omp single nowait
        {
            // Call the parallel quicksort function
            quickSort(vec, 0, size - 1);
        }
    }

    // Measure and print execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    std::cout << "Quick Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    // Verify the sorting result
    checkSortResult(vec_clone, vec);

    return 0;
}
