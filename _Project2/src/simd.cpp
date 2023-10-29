//
// SIMD + Reordering Matrix Multiplication
//

#include <iostream>
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

Matrix matrix_multiply_simd(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    Matrix result(M, N);

    size_t BLOCK_SIZE = 64; 

    if (M < 64 && K < 64 && N < 64) {
        size_t BLOCK_SIZE = 8;
        for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
            for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
                #pragma omp parallel for collapse(2)
                for (size_t i = 0; i < M; i++) {
                    for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); j+=8) {
                        __m256i sum = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result[i][j]));
                        for (size_t k = kk; k < std::min(kk + BLOCK_SIZE, K); ++k) {
                            __m256i multiplier = _mm256_set1_epi32(matrix1[i][k]);
                            __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix2[k][j]));
                            __m256i product1 = _mm256_mullo_epi32(multiplier, b1);

                            sum = _mm256_add_epi32(sum, product1);
                        }
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i][j]), sum);
                    }
                }
            }
        }

        return result;
    }
    else{
        for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
            for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
                for (size_t i = 0; i < M; i++) {
                    for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, N); j+=64) {
                        __m256i sum1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result[i][j]));
                        __m256i sum2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result[i][j + 8]));
                        __m256i sum3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result[i][j + 16]));
                        __m256i sum4 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result[i][j + 24]));
                        __m256i sum5 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result[i][j + 32]));
                        __m256i sum6 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result[i][j + 40]));
                        __m256i sum7 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result[i][j + 48]));
                        __m256i sum8 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result[i][j + 56]));

                        for (size_t k = kk; k < std::min(kk + BLOCK_SIZE, K); ++k) {
                            __m256i multiplier = _mm256_set1_epi32(matrix1[i][k]);
                            __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix2[k][j]));
                            __m256i b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix2[k][j + 8]));
                            __m256i b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix2[k][j + 16]));
                            __m256i b4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix2[k][j + 24]));
                            __m256i b5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix2[k][j + 32]));
                            __m256i b6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix2[k][j + 40]));
                            __m256i b7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix2[k][j + 48]));
                            __m256i b8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matrix2[k][j + 56]));

                            __m256i product1 = _mm256_mullo_epi32(multiplier, b1);
                            __m256i product2 = _mm256_mullo_epi32(multiplier, b2);
                            __m256i product3 = _mm256_mullo_epi32(multiplier, b3);
                            __m256i product4 = _mm256_mullo_epi32(multiplier, b4);
                            __m256i product5 = _mm256_mullo_epi32(multiplier, b5);
                            __m256i product6 = _mm256_mullo_epi32(multiplier, b6);
                            __m256i product7 = _mm256_mullo_epi32(multiplier, b7);
                            __m256i product8 = _mm256_mullo_epi32(multiplier, b8);

                            sum1 = _mm256_add_epi32(sum1, product1);
                            sum2 = _mm256_add_epi32(sum2, product2);
                            sum3 = _mm256_add_epi32(sum3, product3);
                            sum4 = _mm256_add_epi32(sum4, product4);
                            sum5 = _mm256_add_epi32(sum5, product5);
                            sum6 = _mm256_add_epi32(sum6, product6);
                            sum7 = _mm256_add_epi32(sum7, product7);
                            sum8 = _mm256_add_epi32(sum8, product8);

                        }

                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i][j]), sum1);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i][j + 8]), sum2);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i][j + 16]), sum3);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i][j + 24]), sum4);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i][j + 32]), sum5);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i][j + 40]), sum6);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i][j + 48]), sum7);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i][j + 56]), sum8);

                    }
                }
            }
        }

        return result;
    }
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_simd(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}