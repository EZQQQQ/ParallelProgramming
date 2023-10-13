//
// MPI implementation of image filtering
//
#include <iostream>
#include <cmath>
#include <chrono>
#include "utils.hpp"
#include <openacc.h> // OpenACC Header

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

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
    // Apply the filter to the image
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    unsigned char *filteredImage = new unsigned char[width * height * num_channels];
    unsigned char *buffer = new unsigned char[width * height * num_channels];
    for (int i = 0; i < width * height * num_channels; i++)
    {
        buffer[i] = input_jpeg.buffer[i];
    }
#pragma acc enter data copyin(filteredImage[0 : width * height * input_jpeg.num_channels], \
                            buffer[0 : width * height * num_channels])

#pragma acc update device(filteredImage[0 : width * height * input_jpeg.num_channels], \
                          buffer[0 : width * height * num_channels])
                          
    auto start_time = std::chrono::high_resolution_clock::now();
#pragma acc parallel present(filteredImage[0 : width * height * input_jpeg.num_channels], \
                             buffer[0 : width * height * num_channels]) \
    num_gangs(1024) 
    {
#pragma acc loop independent
        for (int height = 1; height < input_jpeg.height - 1; height++)
        {
#pragma acc loop independent
            for (int width = 1; width < input_jpeg.width - 1; width++)
            {
                int sum_r = 0, sum_g = 0, sum_b = 0;
                int channel_value_r, channel_value_g, channel_value_b;

                // Position (height - 1, width - 1)
                channel_value_r = buffer[((height - 1) * input_jpeg.width + (width - 1)) * input_jpeg.num_channels];
                channel_value_g = buffer[((height - 1) * input_jpeg.width + (width - 1)) * input_jpeg.num_channels + 1];
                channel_value_b = buffer[((height - 1) * input_jpeg.width + (width - 1)) * input_jpeg.num_channels + 2];
                sum_r += channel_value_r * filter[0][0];
                sum_g += channel_value_g * filter[0][0];
                sum_b += channel_value_b * filter[0][0];

                // Position (height - 1, width)
                channel_value_r = buffer[((height - 1) * input_jpeg.width + width) * input_jpeg.num_channels];
                channel_value_g = buffer[((height - 1) * input_jpeg.width + width) * input_jpeg.num_channels + 1];
                channel_value_b = buffer[((height - 1) * input_jpeg.width + width) * input_jpeg.num_channels + 2];
                sum_r += channel_value_r * filter[0][1];
                sum_g += channel_value_g * filter[0][1];
                sum_b += channel_value_b * filter[0][1];

                // Position (height - 1, width + 1)
                channel_value_r = buffer[((height - 1) * input_jpeg.width + (width + 1)) * input_jpeg.num_channels];
                channel_value_g = buffer[((height - 1) * input_jpeg.width + (width + 1)) * input_jpeg.num_channels + 1];
                channel_value_b = buffer[((height - 1) * input_jpeg.width + (width + 1)) * input_jpeg.num_channels + 2];
                sum_r += channel_value_r * filter[0][2];
                sum_g += channel_value_g * filter[0][2];
                sum_b += channel_value_b * filter[0][2];

                // Position (height, width - 1)
                channel_value_r = buffer[(height * input_jpeg.width + (width - 1)) * input_jpeg.num_channels];
                channel_value_g = buffer[(height * input_jpeg.width + (width - 1)) * input_jpeg.num_channels + 1];
                channel_value_b = buffer[(height * input_jpeg.width + (width - 1)) * input_jpeg.num_channels + 2];
                sum_r += channel_value_r * filter[1][0];
                sum_g += channel_value_g * filter[1][0];
                sum_b += channel_value_b * filter[1][0];

                // Position (height, width)
                channel_value_r = buffer[(height * input_jpeg.width + width) * input_jpeg.num_channels];
                channel_value_g = buffer[(height * input_jpeg.width + width) * input_jpeg.num_channels + 1];
                channel_value_b = buffer[(height * input_jpeg.width + width) * input_jpeg.num_channels + 2];
                sum_r += channel_value_r * filter[1][1];
                sum_g += channel_value_g * filter[1][1];
                sum_b += channel_value_b * filter[1][1];

                // Position (height, width + 1)
                channel_value_r = buffer[(height * input_jpeg.width + (width + 1)) * input_jpeg.num_channels];
                channel_value_g = buffer[(height * input_jpeg.width + (width + 1)) * input_jpeg.num_channels + 1];
                channel_value_b = buffer[(height * input_jpeg.width + (width + 1)) * input_jpeg.num_channels + 2];
                sum_r += channel_value_r * filter[1][2];
                sum_g += channel_value_g * filter[1][2];
                sum_b += channel_value_b * filter[1][2];

                // Position (height + 1, width - 1)
                channel_value_r = buffer[((height + 1) * input_jpeg.width + (width - 1)) * input_jpeg.num_channels];
                channel_value_g = buffer[((height + 1) * input_jpeg.width + (width - 1)) * input_jpeg.num_channels + 1];
                channel_value_b = buffer[((height + 1) * input_jpeg.width + (width - 1)) * input_jpeg.num_channels + 2];
                sum_r += channel_value_r * filter[2][0];
                sum_g += channel_value_g * filter[2][0];
                sum_b += channel_value_b * filter[2][0];

                // Position (height + 1, width)
                channel_value_r = buffer[((height + 1) * input_jpeg.width + width) * input_jpeg.num_channels];
                channel_value_g = buffer[((height + 1) * input_jpeg.width + width) * input_jpeg.num_channels + 1];
                channel_value_b = buffer[((height + 1) * input_jpeg.width + width) * input_jpeg.num_channels + 2];
                sum_r += channel_value_r * filter[2][1];
                sum_g += channel_value_g * filter[2][1];
                sum_b += channel_value_b * filter[2][1];

                // Position (height + 1, width + 1)
                channel_value_r = buffer[((height + 1) * input_jpeg.width + (width + 1)) * input_jpeg.num_channels];
                channel_value_g = buffer[((height + 1) * input_jpeg.width + (width + 1)) * input_jpeg.num_channels + 1];
                channel_value_b = buffer[((height + 1) * input_jpeg.width + (width + 1)) * input_jpeg.num_channels + 2];
                sum_r += channel_value_r * filter[2][2];
                sum_g += channel_value_g * filter[2][2];
                sum_b += channel_value_b * filter[2][2];

                // Compute the sums for each channel
                filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels]
                    = static_cast<unsigned char>(std::round(sum_r));
                filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels + 1]
                    = static_cast<unsigned char>(std::round(sum_g));
                filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels + 2]
                    = static_cast<unsigned char>(std::round(sum_b));
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
#pragma acc update self(filteredImage[0 : width * height * input_jpeg.num_channels], \
                    buffer[0 : width * height * num_channels])

#pragma acc exit data copyout(filteredImage[0 : input_jpeg.width * input_jpeg.height * input_jpeg.num_channels])
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] buffer;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
