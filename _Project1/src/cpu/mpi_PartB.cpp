//
// MPI implementation of image filtering
//
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include <mpi.h>

#include "utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
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

    // Read JPEG File
    const char * input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Divide the task
    int total_pixel_num = input_jpeg.width * input_jpeg.height;
    int pixel_num_per_task = total_pixel_num / numtasks;
    int left_pixel_num = total_pixel_num % numtasks;

    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_pixel_num = 0;

    for (int i = 0; i < numtasks; i++) {
        if (divided_left_pixel_num < left_pixel_num) {
            cuts[i+1] = cuts[i] + pixel_num_per_task + 1;
            divided_left_pixel_num++;
        } else cuts[i+1] = cuts[i] + pixel_num_per_task;
    }

    // The tasks for the master executor
    if (taskid == MASTER) {
        auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

        // Apply the filter to the image
        for (int i = cuts[MASTER]; i < cuts[MASTER + 1]; i++) {
            int sum_r = 0, sum_g = 0, sum_b = 0;
            // Unrolled loop for j=-1, k=-1
            int x = i % input_jpeg.width - 1;
            int y = i / input_jpeg.width - 1;
            int channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            int channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            int channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[0][0];
            sum_g += channel_value_g * filter[0][0];
            sum_b += channel_value_b * filter[0][0];

            // Unrolled loop for j=-1, k=0
            x = i % input_jpeg.width - 1;
            y = i / input_jpeg.width;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[0][1];
            sum_g += channel_value_g * filter[0][1];
            sum_b += channel_value_b * filter[0][1];

            // Unrolled loop for j=-1, k=1
            x = i % input_jpeg.width - 1;
            y = i / input_jpeg.width + 1;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[0][2];
            sum_g += channel_value_g * filter[0][2];
            sum_b += channel_value_b * filter[0][2];

            // Unrolled loop for j=0, k=-1
            x = i % input_jpeg.width;
            y = i / input_jpeg.width - 1;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[1][0];
            sum_g += channel_value_g * filter[1][0];
            sum_b += channel_value_b * filter[1][0];

            // Unrolled loop for j=0, k=0 (center pixel)
            x = i % input_jpeg.width;
            y = i / input_jpeg.width;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[1][1];
            sum_g += channel_value_g * filter[1][1];
            sum_b += channel_value_b * filter[1][1];

            // Unrolled loop for j=0, k=1
            x = i % input_jpeg.width;
            y = i / input_jpeg.width + 1;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[1][2];
            sum_g += channel_value_g * filter[1][2];
            sum_b += channel_value_b * filter[1][2];

            // Unrolled loop for j=1, k=-1
            x = i % input_jpeg.width + 1;
            y = i / input_jpeg.width - 1;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[2][0];
            sum_g += channel_value_g * filter[2][0];
            sum_b += channel_value_b * filter[2][0];

            // Unrolled loop for j=1, k=0
            x = i % input_jpeg.width + 1;
            y = i / input_jpeg.width;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[2][1];
            sum_g += channel_value_g * filter[2][1];
            sum_b += channel_value_b * filter[2][1];

            // Unrolled loop for j=1, k=1
            x = i % input_jpeg.width + 1;
            y = i / input_jpeg.width + 1;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[2][2];
            sum_g += channel_value_g * filter[2][2];
            sum_b += channel_value_b * filter[2][2];

            filteredImage[i * input_jpeg.num_channels]
                = static_cast<unsigned char>(std::round(sum_r));
            filteredImage[i * input_jpeg.num_channels + 1]
                = static_cast<unsigned char>(std::round(sum_g));
            filteredImage[i * input_jpeg.num_channels + 2]
                = static_cast<unsigned char>(std::round(sum_b));
        }

        // Receive the filtered contents from slave executors
        for (int i = MASTER + 1; i < numtasks; i++) {
            unsigned char* start_pos = filteredImage + cuts[i] * input_jpeg.num_channels;
            int length = (cuts[i+1] - cuts[i]) * input_jpeg.num_channels;
            MPI_Recv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Save the Filtered Image
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG to file\n";
            MPI_Finalize();
            return -1;
        }

        // Release the memory
        delete[] input_jpeg.buffer;
        delete[] filteredImage;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    } 
    // The tasks for the slave executor
    else {
        int length = cuts[taskid + 1] - cuts[taskid];
        auto filteredImage = new unsigned char[length * input_jpeg.num_channels];
        for (int i = cuts[taskid]; i < cuts[taskid + 1]; i++) {
            int sum_r = 0, sum_g = 0, sum_b = 0;
            // Unrolled loop for j=-1, k=-1
            int x = i % input_jpeg.width - 1;
            int y = i / input_jpeg.width - 1;
            int channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            int channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            int channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[0][0];
            sum_g += channel_value_g * filter[0][0];
            sum_b += channel_value_b * filter[0][0];

            // Unrolled loop for j=-1, k=0
            x = i % input_jpeg.width - 1;
            y = i / input_jpeg.width;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[0][1];
            sum_g += channel_value_g * filter[0][1];
            sum_b += channel_value_b * filter[0][1];

            // Unrolled loop for j=-1, k=1
            x = i % input_jpeg.width - 1;
            y = i / input_jpeg.width + 1;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[0][2];
            sum_g += channel_value_g * filter[0][2];
            sum_b += channel_value_b * filter[0][2];

            // Unrolled loop for j=0, k=-1
            x = i % input_jpeg.width;
            y = i / input_jpeg.width - 1;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[1][0];
            sum_g += channel_value_g * filter[1][0];
            sum_b += channel_value_b * filter[1][0];

            // Unrolled loop for j=0, k=0 (center pixel)
            x = i % input_jpeg.width;
            y = i / input_jpeg.width;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[1][1];
            sum_g += channel_value_g * filter[1][1];
            sum_b += channel_value_b * filter[1][1];

            // Unrolled loop for j=0, k=1
            x = i % input_jpeg.width;
            y = i / input_jpeg.width + 1;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[1][2];
            sum_g += channel_value_g * filter[1][2];
            sum_b += channel_value_b * filter[1][2];

            // Unrolled loop for j=1, k=-1
            x = i % input_jpeg.width + 1;
            y = i / input_jpeg.width - 1;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[2][0];
            sum_g += channel_value_g * filter[2][0];
            sum_b += channel_value_b * filter[2][0];

            // Unrolled loop for j=1, k=0
            x = i % input_jpeg.width + 1;
            y = i / input_jpeg.width;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[2][1];
            sum_g += channel_value_g * filter[2][1];
            sum_b += channel_value_b * filter[2][1];

            // Unrolled loop for j=1, k=1
            x = i % input_jpeg.width + 1;
            y = i / input_jpeg.width + 1;
            channel_value_r = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels];
            channel_value_g = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 1];
            channel_value_b = input_jpeg.buffer[(y * input_jpeg.width + x) * input_jpeg.num_channels + 2];
            sum_r += channel_value_r * filter[2][2];
            sum_g += channel_value_g * filter[2][2];
            sum_b += channel_value_b * filter[2][2];
            int j = i - cuts[taskid];
            filteredImage[j * input_jpeg.num_channels]
                = static_cast<unsigned char>(std::round(sum_r));
            filteredImage[j * input_jpeg.num_channels + 1]
                = static_cast<unsigned char>(std::round(sum_g));
            filteredImage[j * input_jpeg.num_channels + 2]
                = static_cast<unsigned char>(std::round(sum_b));
        }

        // Send the filtered image back to the master
        MPI_Send(filteredImage, length * input_jpeg.num_channels, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);

        // Release the memory
        delete[] filteredImage;
    }

    MPI_Finalize();
    return 0;
}
