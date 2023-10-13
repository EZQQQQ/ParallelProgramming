//
// MPI implementation of image filtering
//
#include <iostream>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h> // CUDA Header

#include "utils.hpp"

const int FILTER_SIZE = 3;
const double host_filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

// Device function for rounding
__device__ unsigned char device_round(double val)
{
    return static_cast<unsigned char>(val + 0.5);
}

__global__ void applyFilter(const unsigned char* input, unsigned char* output,
                            int width, int height, int num_channels,
                            const double* filter, int filter_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int output_idx = (y * width + x) * num_channels;
        double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;

        // Unroll the loop for fy and fx
        int fy = 0;
        int fx = 0;

        // Calculate initial indices
        int image_x = x + fx - filter_size / 2;
        int image_y = y + fy - filter_size / 2;

        // Check boundaries and calculate filter_value
        if (image_x >= 0 && image_x < width && image_y >= 0 && image_y < height)
        {
            int input_idx = (image_y * width + image_x) * num_channels;
            double filter_value = filter[fy * filter_size + fx];

            sum_r += static_cast<double>(input[input_idx]) * filter_value;
            sum_g += static_cast<double>(input[input_idx + 1]) * filter_value;
            sum_b += static_cast<double>(input[input_idx + 2]) * filter_value;
        }

        fy = 0;
        fx = 1;

        // Calculate indices for fx = 1 (no need to update fy)
        image_x = x + fx - filter_size / 2;
        image_y = y + fy - filter_size / 2;

        // Check boundaries and calculate filter_value
        if (image_x >= 0 && image_x < width && image_y >= 0 && image_y < height)
        {
            int input_idx = (image_y * width + image_x) * num_channels;
            double filter_value = filter[fy * filter_size + fx];

            sum_r += static_cast<double>(input[input_idx]) * filter_value;
            sum_g += static_cast<double>(input[input_idx + 1]) * filter_value;
            sum_b += static_cast<double>(input[input_idx + 2]) * filter_value;
        }

        // Continue unrolling for other fy and fx values
        // Here, you can unroll further or use nested loops, depending on your filter size

        // Store the results
        output[output_idx] = static_cast<unsigned char>(sum_r);
        output[output_idx + 1] = static_cast<unsigned char>(sum_g);
        output[output_idx + 2] = static_cast<unsigned char>(sum_b);
    }
}



int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);
    // Allocate memory on host (CPU)
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    double* d_filter; // Added for the filter array on the device
    cudaMalloc((void**)&d_input, input_jpeg.width * input_jpeg.height *
                                     input_jpeg.num_channels *
                                     sizeof(unsigned char));
    cudaMalloc((void**)&d_output,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels *
               sizeof(unsigned char));
    cudaMalloc((void**)&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(double)); // Allocate space for the filter on the device
    // Copy input data and filter from host to device
    cudaMemcpy(d_input, input_jpeg.buffer,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels *
               sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double linear_filter[FILTER_SIZE * FILTER_SIZE];
    int filter_index = 0;
    for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            linear_filter[filter_index++] = host_filter[i][j];
        }
    }
    cudaMemcpy(d_filter, linear_filter,
               FILTER_SIZE * FILTER_SIZE * sizeof(double),
               cudaMemcpyHostToDevice);
    // Computation: Apply the filter using CUDA
    dim3 blockSize(16, 16); // Adjust block size as needed
    dim3 gridSize((input_jpeg.width + blockSize.x - 1) / blockSize.x,
                  (input_jpeg.height + blockSize.y - 1) / blockSize.y);
    cudaEventRecord(start, 0); // GPU start time
    applyFilter<<<gridSize, blockSize>>>(d_input, d_output, input_jpeg.width,
                                         input_jpeg.height,
                                         input_jpeg.num_channels, d_filter, FILTER_SIZE); // Pass the filter and its size as arguments
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    cudaMemcpy(filteredImage, d_output,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels *
               sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory on device and host
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter); // Free the filter memory on the device
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}