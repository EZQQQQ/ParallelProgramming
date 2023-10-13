//
// SIMD (AVX2) implementation of applying a smoothing filter to a JPEG picture
//

#include <iostream>
#include <chrono>

#include <immintrin.h>

#include "utils.hpp"

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
    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Allocate memory for the output image
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    auto outputImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

    // Set SIMD scalars, we use AVX2 instructions
    __m256 filterValues[FILTER_SIZE][FILTER_SIZE];
    for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            filterValues[i][j] = _mm256_set1_ps(static_cast<float>(filter[i][j]));
        }
    }

    // Using SIMD to accelerate the smoothing filter
    auto start_time = std::chrono::high_resolution_clock::now();    // Start recording time
    for (int h = 1; h < height - 1; h++) {
        for (int w = 1; w < width - 1; w++) {
            // Initialize three 256-bit SIMD registers for red, green, and blue sums
            __m256 red_sum = _mm256_setzero_ps();
            __m256 green_sum = _mm256_setzero_ps();
            __m256 blue_sum = _mm256_setzero_ps();

            // Load the pixel data using SIMD
            __m256i pixel_data0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_jpeg.buffer[((h - 1) * width + w - 1) * num_channels]));
            __m256i pixel_data1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_jpeg.buffer[((h - 1) * width + w) * num_channels]));
            __m256i pixel_data2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_jpeg.buffer[((h - 1) * width + w + 1) * num_channels]));
            __m256i pixel_data3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_jpeg.buffer[(h * width + w - 1) * num_channels]));
            __m256i pixel_data4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_jpeg.buffer[(h * width + w) * num_channels]));
            __m256i pixel_data5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_jpeg.buffer[(h * width + w + 1) * num_channels]));
            __m256i pixel_data6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_jpeg.buffer[((h + 1) * width + w - 1) * num_channels]));
            __m256i pixel_data7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_jpeg.buffer[((h + 1) * width + w) * num_channels]));
            __m256i pixel_data8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_jpeg.buffer[((h + 1) * width + w + 1) * num_channels]));

            // Load the pixel data using SIMD for red channel
            __m256i red_channel0 = _mm256_and_si256(pixel_data0, _mm256_set1_epi32(0x000000FF)); // Red channel
            __m256i red_channel1 = _mm256_and_si256(pixel_data1, _mm256_set1_epi32(0x000000FF));
            __m256i red_channel2 = _mm256_and_si256(pixel_data2, _mm256_set1_epi32(0x000000FF));
            __m256i red_channel3 = _mm256_and_si256(pixel_data3, _mm256_set1_epi32(0x000000FF));
            __m256i red_channel4 = _mm256_and_si256(pixel_data4, _mm256_set1_epi32(0x000000FF));
            __m256i red_channel5 = _mm256_and_si256(pixel_data5, _mm256_set1_epi32(0x000000FF));
            __m256i red_channel6 = _mm256_and_si256(pixel_data6, _mm256_set1_epi32(0x000000FF));
            __m256i red_channel7 = _mm256_and_si256(pixel_data7, _mm256_set1_epi32(0x000000FF));
            __m256i red_channel8 = _mm256_and_si256(pixel_data8, _mm256_set1_epi32(0x000000FF));

            // Convert 256-bit packed integer values to 256-bit packed single-precision floating-point values for red channel
            __m256 red0 = _mm256_cvtepi32_ps(red_channel0);
            __m256 red1 = _mm256_cvtepi32_ps(red_channel1);
            __m256 red2 = _mm256_cvtepi32_ps(red_channel2);
            __m256 red3 = _mm256_cvtepi32_ps(red_channel3);
            __m256 red4 = _mm256_cvtepi32_ps(red_channel4);
            __m256 red5 = _mm256_cvtepi32_ps(red_channel5);
            __m256 red6 = _mm256_cvtepi32_ps(red_channel6);
            __m256 red7 = _mm256_cvtepi32_ps(red_channel7);
            __m256 red8 = _mm256_cvtepi32_ps(red_channel8);

            // Multiply the floats to the filterValues and accumulate for red channel
            red0 = _mm256_mul_ps(red0, filterValues[0][0]);
            red1 = _mm256_mul_ps(red1, filterValues[0][1]);
            red2 = _mm256_mul_ps(red2, filterValues[0][2]);
            red3 = _mm256_mul_ps(red3, filterValues[1][0]);
            red4 = _mm256_mul_ps(red4, filterValues[1][1]);
            red5 = _mm256_mul_ps(red5, filterValues[1][2]);
            red6 = _mm256_mul_ps(red6, filterValues[2][0]);
            red7 = _mm256_mul_ps(red7, filterValues[2][1]);
            red8 = _mm256_mul_ps(red8, filterValues[2][2]);

            red_sum = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(red0, red1), red2), red3), red4), red5), red6), red7), red8);

            // Load the pixel data using SIMD for green channel
            __m256i green_channel0 = _mm256_and_si256(pixel_data0, _mm256_set1_epi32(0x0000FF00)); // Green channel
            __m256i green_channel1 = _mm256_and_si256(pixel_data1, _mm256_set1_epi32(0x0000FF00));
            __m256i green_channel2 = _mm256_and_si256(pixel_data2, _mm256_set1_epi32(0x0000FF00));
            __m256i green_channel3 = _mm256_and_si256(pixel_data3, _mm256_set1_epi32(0x0000FF00));
            __m256i green_channel4 = _mm256_and_si256(pixel_data4, _mm256_set1_epi32(0x0000FF00));
            __m256i green_channel5 = _mm256_and_si256(pixel_data5, _mm256_set1_epi32(0x0000FF00));
            __m256i green_channel6 = _mm256_and_si256(pixel_data6, _mm256_set1_epi32(0x0000FF00));
            __m256i green_channel7 = _mm256_and_si256(pixel_data7, _mm256_set1_epi32(0x0000FF00));
            __m256i green_channel8 = _mm256_and_si256(pixel_data8, _mm256_set1_epi32(0x0000FF00));

            // Convert 256-bit packed integer values to 256-bit packed single-precision floating-point values for green channel
            __m256 green0 = _mm256_cvtepi32_ps(green_channel0);
            __m256 green1 = _mm256_cvtepi32_ps(green_channel1);
            __m256 green2 = _mm256_cvtepi32_ps(green_channel2);
            __m256 green3 = _mm256_cvtepi32_ps(green_channel3);
            __m256 green4 = _mm256_cvtepi32_ps(green_channel4);
            __m256 green5 = _mm256_cvtepi32_ps(green_channel5);
            __m256 green6 = _mm256_cvtepi32_ps(green_channel6);
            __m256 green7 = _mm256_cvtepi32_ps(green_channel7);
            __m256 green8 = _mm256_cvtepi32_ps(green_channel8);

            // Multiply the floats to the filterValues and accumulate for green channel
            green0 = _mm256_mul_ps(green0, filterValues[0][0]);
            green1 = _mm256_mul_ps(green1, filterValues[0][1]);
            green2 = _mm256_mul_ps(green2, filterValues[0][2]);
            green3 = _mm256_mul_ps(green3, filterValues[1][0]);
            green4 = _mm256_mul_ps(green4, filterValues[1][1]);
            green5 = _mm256_mul_ps(green5, filterValues[1][2]);
            green6 = _mm256_mul_ps(green6, filterValues[2][0]);
            green7 = _mm256_mul_ps(green7, filterValues[2][1]);
            green8 = _mm256_mul_ps(green8, filterValues[2][2]);

            green_sum = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(green0, green1), green2), green3), green4), green5), green6), green7), green8);

            // Load the pixel data using SIMD for blue channel
            __m256i blue_channel0 = _mm256_and_si256(pixel_data0, _mm256_set1_epi32(0x00FF0000)); // Blue channel
            __m256i blue_channel1 = _mm256_and_si256(pixel_data1, _mm256_set1_epi32(0x00FF0000));
            __m256i blue_channel2 = _mm256_and_si256(pixel_data2, _mm256_set1_epi32(0x00FF0000));
            __m256i blue_channel3 = _mm256_and_si256(pixel_data3, _mm256_set1_epi32(0x00FF0000));
            __m256i blue_channel4 = _mm256_and_si256(pixel_data4, _mm256_set1_epi32(0x00FF0000));
            __m256i blue_channel5 = _mm256_and_si256(pixel_data5, _mm256_set1_epi32(0x00FF0000));
            __m256i blue_channel6 = _mm256_and_si256(pixel_data6, _mm256_set1_epi32(0x00FF0000));
            __m256i blue_channel7 = _mm256_and_si256(pixel_data7, _mm256_set1_epi32(0x00FF0000));
            __m256i blue_channel8 = _mm256_and_si256(pixel_data8, _mm256_set1_epi32(0x00FF0000));

            // Convert 256-bit packed integer values to 256-bit packed single-precision floating-point values for blue channel
            __m256 blue0 = _mm256_cvtepi32_ps(blue_channel0);
            __m256 blue1 = _mm256_cvtepi32_ps(blue_channel1);
            __m256 blue2 = _mm256_cvtepi32_ps(blue_channel2);
            __m256 blue3 = _mm256_cvtepi32_ps(blue_channel3);
            __m256 blue4 = _mm256_cvtepi32_ps(blue_channel4);
            __m256 blue5 = _mm256_cvtepi32_ps(blue_channel5);
            __m256 blue6 = _mm256_cvtepi32_ps(blue_channel6);
            __m256 blue7 = _mm256_cvtepi32_ps(blue_channel7);
            __m256 blue8 = _mm256_cvtepi32_ps(blue_channel8);

            // Multiply the floats to the filterValues and accumulate for blue channel
            blue0 = _mm256_mul_ps(blue0, filterValues[0][0]);
            blue1 = _mm256_mul_ps(blue1, filterValues[0][1]);
            blue2 = _mm256_mul_ps(blue2, filterValues[0][2]);
            blue3 = _mm256_mul_ps(blue3, filterValues[1][0]);
            blue4 = _mm256_mul_ps(blue4, filterValues[1][1]);
            blue5 = _mm256_mul_ps(blue5, filterValues[1][2]);
            blue6 = _mm256_mul_ps(blue6, filterValues[2][0]);
            blue7 = _mm256_mul_ps(blue7, filterValues[2][1]);
            blue8 = _mm256_mul_ps(blue8, filterValues[2][2]);

            blue_sum = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(blue0, blue1), blue2), blue3), blue4), blue5), blue6), blue7), blue8);

            // Convert the red, green, and blue sums to 32-bit integers
            __m256i red_int = _mm256_cvtps_epi32(red_sum);
            __m256i green_int = _mm256_cvtps_epi32(green_sum);
            __m256i blue_int = _mm256_cvtps_epi32(blue_sum);

            // Pack the 32-bit integers into 16-bit integers
            __m256i red_green_packed = _mm256_packus_epi32(red_int, green_int);

            // Pack the packed red-green with the blue into 8-bit integers
            __m256i result = _mm256_packus_epi16(red_green_packed, blue_int);

            // Store the results in the output image
            __m128i pixel_result = _mm256_castsi256_si128(result);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&outputImage[(h * width + w) * num_channels]), pixel_result);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();  // Stop recording time
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output filtered JPEG Image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{outputImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] outputImage;
    std::cout << "Smoothing Filter Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}