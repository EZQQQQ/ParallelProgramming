//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0

void insertionSort(std::vector<int>& bucket) {
    for (int i = 1; i < bucket.size(); ++i) {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key) {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}

void bucketSort(std::vector<int>& vec, int num_buckets, int numtasks, int taskid, MPI_Status* status) {
    const int max_val = *std::max_element(vec.begin(), vec.end());
    const int min_val = *std::min_element(vec.begin(), vec.end());

    const int range = max_val - min_val + 1;
    const int bucket_range = range / num_buckets;

    // Calculate the range of elements each process will handle
    int local_start = min_val + taskid * (range / numtasks);
    int local_end = std::min(max_val + 1, local_start + (range / numtasks));

    // Create local buckets
    std::vector<std::vector<int>> local_buckets(num_buckets);

    // Distribute elements into local buckets based on the calculated range
    for (int i = 0; i < vec.size(); ++i) {
        if (vec[i] >= local_start && vec[i] < local_end) {
            int bucket_index = (vec[i] - min_val) / bucket_range;
            local_buckets[bucket_index].push_back(vec[i]);
        }
    }

    // Sort and flatten local buckets: Elements are sorted within each local bucket using insertion sort,
    // and the sorted contents are concatenated to the flattened_local_buckets vector.
    std::vector<int> flattened_local_buckets;
    for (int i = 0; i < num_buckets; ++i) {
        insertionSort(local_buckets[i]);  // Use insertion sort for local bucket i
        flattened_local_buckets.insert(
            flattened_local_buckets.end(),
            local_buckets[i].begin(),
            local_buckets[i].end()
        );
    }

    // Gather sizes of local buckets on the master node
    std::vector<int> all_bucket_sizes(numtasks, 0);
    int local_bucket_size = flattened_local_buckets.size();
    MPI_Gather(&local_bucket_size, 1, MPI_INT, all_bucket_sizes.data(), 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Calculate displacements for MPI_Gatherv
    std::vector<int> displs(numtasks, 0);
    if (taskid == MASTER) {
        std::partial_sum(all_bucket_sizes.begin(), all_bucket_sizes.end() - 1, displs.begin() + 1);
    }

    // Gather all local buckets to the master node
    MPI_Gatherv(
        flattened_local_buckets.data(), local_bucket_size, MPI_INT,
        vec.data(), all_bucket_sizes.data(), displs.data(), MPI_INT, MASTER, MPI_COMM_WORLD
    );
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size bucket_num\n"
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

    const int bucket_num = atoi(argv[2]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    bucketSort(vec, bucket_num, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}