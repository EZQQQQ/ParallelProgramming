#include <iostream>
#include <chrono>
#include <pthread.h>
#include <cmath>
#include "utils.hpp"
#include <omp.h>

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

// Structure to pass data to each thread
struct ThreadData {
    unsigned char* input_buffer;
    unsigned char* output_buffer;
    int start;
    int end;
    int width;
    int num_channels;
};

// Function to apply the filter to a portion of the image
void* applyFilter(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);

    for (int height = data->start; height < data->end; height++) {
        for (int width = 1; width < data->width - 1; width++) {
            int channel_value_r, channel_value_g, channel_value_b;

            // Initialize sums
            int sum_r = 0;
            int sum_g = 0;
            int sum_b = 0;

            // First iteration (-1, -1)
            channel_value_r = data->input_buffer[((height - 1) * data->width + (width - 1)) * data->num_channels];
            channel_value_g = data->input_buffer[((height - 1) * data->width + (width - 1)) * data->num_channels + 1];
            channel_value_b = data->input_buffer[((height - 1) * data->width + (width - 1)) * data->num_channels + 2];
            sum_r += channel_value_r * filter[0][0];
            sum_g += channel_value_g * filter[0][0];
            sum_b += channel_value_b * filter[0][0];

            // Second iteration (-1, 0)
            channel_value_r = data->input_buffer[((height - 1) * data->width + width) * data->num_channels];
            channel_value_g = data->input_buffer[((height - 1) * data->width + width) * data->num_channels + 1];
            channel_value_b = data->input_buffer[((height - 1) * data->width + width) * data->num_channels + 2];
            sum_r += channel_value_r * filter[0][1];
            sum_g += channel_value_g * filter[0][1];
            sum_b += channel_value_b * filter[0][1];

            // Third iteration (-1, 1)
            channel_value_r = data->input_buffer[((height - 1) * data->width + (width + 1)) * data->num_channels];
            channel_value_g = data->input_buffer[((height - 1) * data->width + (width + 1)) * data->num_channels + 1];
            channel_value_b = data->input_buffer[((height - 1) * data->width + (width + 1)) * data->num_channels + 2];
            sum_r += channel_value_r * filter[0][2];
            sum_g += channel_value_g * filter[0][2];
            sum_b += channel_value_b * filter[0][2];

            // Fourth iteration (0, -1)
            channel_value_r = data->input_buffer[(height * data->width + (width - 1)) * data->num_channels];
            channel_value_g = data->input_buffer[(height * data->width + (width - 1)) * data->num_channels + 1];
            channel_value_b = data->input_buffer[(height * data->width + (width - 1)) * data->num_channels + 2];
            sum_r += channel_value_r * filter[1][0];
            sum_g += channel_value_g * filter[1][0];
            sum_b += channel_value_b * filter[1][0];

            // Fifth iteration (0, 0)
            channel_value_r = data->input_buffer[(height * data->width + width) * data->num_channels];
            channel_value_g = data->input_buffer[(height * data->width + width) * data->num_channels + 1];
            channel_value_b = data->input_buffer[(height * data->width + width) * data->num_channels + 2];
            sum_r += channel_value_r * filter[1][1];
            sum_g += channel_value_g * filter[1][1];
            sum_b += channel_value_b * filter[1][1];

            // Sixth iteration (0, 1)
            channel_value_r = data->input_buffer[(height * data->width + (width + 1)) * data->num_channels];
            channel_value_g = data->input_buffer[(height * data->width + (width + 1)) * data->num_channels + 1];
            channel_value_b = data->input_buffer[(height * data->width + (width + 1)) * data->num_channels + 2];
            sum_r += channel_value_r * filter[1][2];
            sum_g += channel_value_g * filter[1][2];
            sum_b += channel_value_b * filter[1][2];

            // Seventh iteration (1, -1)
            channel_value_r = data->input_buffer[((height + 1) * data->width + (width - 1)) * data->num_channels];
            channel_value_g = data->input_buffer[((height + 1) * data->width + (width - 1)) * data->num_channels + 1];
            channel_value_b = data->input_buffer[((height + 1) * data->width + (width - 1)) * data->num_channels + 2];
            sum_r += channel_value_r * filter[2][0];
            sum_g += channel_value_g * filter[2][0];
            sum_b += channel_value_b * filter[2][0];

            // Eighth iteration (1, 0)
            channel_value_r = data->input_buffer[((height + 1) * data->width + width) * data->num_channels];
            channel_value_g = data->input_buffer[((height + 1) * data->width + width) * data->num_channels + 1];
            channel_value_b = data->input_buffer[((height + 1) * data->width + width) * data->num_channels + 2];
            sum_r += channel_value_r * filter[2][1];
            sum_g += channel_value_g * filter[2][1];
            sum_b += channel_value_b * filter[2][1];

            // Ninth iteration (1, 1)
            channel_value_r = data->input_buffer[((height + 1) * data->width + (width + 1)) * data->num_channels];
            channel_value_g = data->input_buffer[((height + 1) * data->width + (width + 1)) * data->num_channels + 1];
            channel_value_b = data->input_buffer[((height + 1) * data->width + (width + 1)) * data->num_channels + 2];
            sum_r += channel_value_r * filter[2][2];
            sum_g += channel_value_g * filter[2][2];
            sum_b += channel_value_b * filter[2][2];
            data->output_buffer[(height * data->width + width) * data->num_channels]
                = static_cast<unsigned char>(std::round(sum_r));
            data->output_buffer[(height * data->width + width) * data->num_channels + 1]
                = static_cast<unsigned char>(std::round(sum_g));
            data->output_buffer[(height * data->width + width) * data->num_channels + 2]
                = static_cast<unsigned char>(std::round(sum_b));
        }
    }

    return nullptr;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read input JPEG image
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);

    int num_threads = std::stoi(argv[3]); // You can change the number of threads as needed

    // Allocate memory for the filtered image
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    auto start_time = std::chrono::high_resolution_clock::now();

    int chunk_size = input_jpeg.height / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input_buffer = input_jpeg.buffer;
        thread_data[i].output_buffer = filteredImage;
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? input_jpeg.height : (i + 1) * chunk_size;
        thread_data[i].width = input_jpeg.width;
        thread_data[i].num_channels = input_jpeg.num_channels;

        pthread_create(&threads[i], nullptr, applyFilter, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};

    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}
