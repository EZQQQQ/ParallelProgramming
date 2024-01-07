#include "simple_ml_openacc.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>

void matrix_copy_openacc(float *A, const float *B, size_t m, size_t n)
{
    #pragma acc data copyin(B[0:m*n]) copyout(A[0:m*n])
    {
        #pragma acc parallel loop present(A[0:m*n], B[0:m*n])
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A[i * n + j] = B[i * n + j];
            }
        }
    } // End of data region
}


void matrix_dot_openacc(const float *A, const float *B,
                        float *C, size_t m, size_t n, size_t k)
{
    #pragma acc data copyin(A[0:m*n], B[0:n*k]) copyout(C[0:m*k])
    #pragma acc parallel loop collapse(2)
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            float temp_sum = 0.0;

            // Unroll the innermost loop
            #pragma acc loop vector reduction(+:temp_sum)
            for (size_t l = 0; l < n; ++l)
            {
                temp_sum += A[i * n + l] * B[l * k + j];
            }

            // Accumulate the result
            C[i * k + j] = temp_sum;
        }
    }
}

void matrix_dot_trans_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    #pragma acc data copyin(A[0:n*m], B[0:n*k]) copyout(C[0:m*k])
    #pragma acc parallel loop collapse(2)
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            float temp_sum = 0.0;

            // Unroll the innermost loop
            #pragma acc loop vector reduction(+:temp_sum)
            for (size_t l = 0; l < n; ++l)
            {
                temp_sum += A[l * m + i] * B[l * k + j];
            }

            // Accumulate the result
            C[i * k + j] = temp_sum;
        }
    }
}


void matrix_trans_dot_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    #pragma acc data copyin(A[0:m*n], B[0:k*n]) copyout(C[0:m*k])
    #pragma acc parallel loop collapse(2)
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            float temp_sum = 0.0;

            // Unroll the innermost loop
            #pragma acc loop vector reduction(+:temp_sum)
            for (size_t l = 0; l < n; ++l)
            {
                temp_sum += A[i * n + l] * B[j * n + l];
            }

            // Accumulate the result
            C[i * k + j] = temp_sum;
        }
    }
}


void matrix_minus_openacc(float *A, const float *B, size_t m, size_t n)
{
    #pragma acc data copyin(A[0:m*n], B[0:m*n]) copyout(A[0:m*n])
    {
        #pragma acc parallel loop independent present(A[:m*n], B[:m*n])
        for (size_t i = 0; i < m; ++i) {
            #pragma acc loop independent
            for (size_t j = 0; j < n; ++j) {
                A[i * n + j] -= B[i * n + j];
            }
        }
    } // Data will be copied back to the host after this block
}


void matrix_mul_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    #pragma acc data copyin(C[0:m*n]) copyout(C[0:m*n])
    {
        #pragma acc parallel loop independent present(C[0:m*n])
        for (size_t i = 0; i < m; ++i) {
            #pragma acc loop
            for (size_t j = 0; j < n; ++j) {
                C[i * n + j] *= scalar;
            }
        }
    } // End of data region
}


void matrix_div_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    #pragma acc data copyin(C[0:m*n]) copyout(C[0:m*n])
    {
        #pragma acc parallel loop independent present(C[0:m*n])
        for (size_t i = 0; i < m; ++i) {
            #pragma acc loop
            for (size_t j = 0; j < n; ++j) {
                C[i * n + j] /= scalar;
            }
        }
    } // End of data region
}


void matrix_softmax_normalize_openacc(float *C, size_t m, size_t n)
{
    #pragma acc data copyin(C[:m*n]) 
    {
        #pragma acc parallel loop present(C[:m*n])
        for (size_t i = 0; i < m; ++i) {
            // Find the maximum value in the row
            float max_val = C[i * n];
            #pragma acc loop reduction(max:max_val)
            for (size_t j = 1; j < n; ++j) {
                max_val = std::max(max_val, C[i * n + j]);
            }

            // Calculate the exponential and sum_exp
            float sum_exp = 0.0;
            #pragma acc loop reduction(+:sum_exp)
            for (size_t j = 0; j < n; ++j) {
                C[i * n + j] = std::exp(C[i * n + j] - max_val);
                sum_exp += C[i * n + j];
            }

            // Normalize by dividing by sum_exp
            #pragma acc loop
            for (size_t j = 0; j < n; ++j) {
                C[i * n + j] /= sum_exp;
            }
        }
        #pragma acc update self(C[:m*n])
    }
}

void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y, size_t m, size_t n)
{
    #pragma acc data copyin(y[0:m]) copyout(Y[0:(m*n)])
    #pragma acc parallel loop gang
    for (size_t i = 0; i < m; ++i)
    {
        #pragma acc loop vector
        for (size_t j = 0; j < n; ++j)
        {
            // If the column index equals the value in y[i], set Y[i * n + j] to 1.0, otherwise set it to 0.0.
            Y[i * n + j] = (j == y[i]) ? 1.0f : 0.0f;
        }
    }
}

void softmax_regression_epoch_openacc(const float *X, const unsigned char *y, float *theta, size_t m, size_t n, size_t k, float lr, size_t batch)
{
    float *Y = new float[m * k];
    float *gradients = new float[n * k];
    
    for (size_t i = 0; i < m; i += batch) {
        const float *X_b = X + (i * n);
        size_t current_batch = std::min(batch, m - i);

        // Create one-hot matrix Y using vector_to_one_hot_matrix function
        std::fill(Y, Y + current_batch * k, 0.0); // Initialize Y with zeros
        vector_to_one_hot_matrix(&y[i], Y, current_batch, k);

        // Compute logits using matrix_dot function
        float logits[current_batch * k];
        matrix_dot_openacc(X_b, theta, logits, current_batch, n, k);

        // Apply softmax using matrix_softmax_normalize function
        matrix_softmax_normalize(logits, current_batch, k);

        // Compute gradients
        #pragma acc enter data copyin(X_b[0:current_batch*n], logits[0:current_batch * k], Y[0:current_batch * k], gradients[0:n*k])
        #pragma acc parallel loop present(X_b[0:current_batch*n], logits[0:current_batch*k], Y[0:current_batch * k], gradients[0:n*k])
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < current_batch; ++l) {
                #pragma acc loop
                for (size_t p = 0; p < k; ++p) {
                    gradients[j * k + p] += X_b[l * n + j] * (logits[l * k + p] - Y[l * k + p]);
                }
            }
        }
        #pragma acc exit data copyout(X_b[0:n*current_batch], logits[0:current_batch * k], Y[0:current_batch * k], gradients[0:n*k])

        // Normalize and update theta
        #pragma acc parallel loop independent
        for (size_t idx = 0; idx < n * k; ++idx) {
            gradients[idx] /= batch;
            gradients[idx] *= lr;
        }

        // Update theta
        matrix_minus(theta, gradients, n, k);
    }


}

void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;

    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    {
        for (size_t epoch = 0; epoch < epochs; epoch++)
        {
            // Softmax regression epoch
            softmax_regression_epoch_openacc(train_data->images_matrix, train_data->labels_array, theta, train_data->images_num, train_data->input_dim, num_classes, lr, batch);

            // Evaluate on training and test data
            matrix_dot_openacc(train_data->images_matrix, theta, train_result, train_data->images_num, train_data->input_dim, num_classes);
            matrix_dot_openacc(test_data->images_matrix, theta, test_result, test_data->images_num, test_data->input_dim, num_classes);

            train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
            test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
            train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
            test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);

            std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                        << std::fixed << std::setprecision(5) << train_loss << " |   "
                        << std::fixed << std::setprecision(5) << train_err << " |   "
                        << std::fixed << std::setprecision(5) << test_loss << " |  "
                        << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}


float mean_softmax_loss_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // Initialize the total loss
    float total_loss = 0.0f;

    // Enter data region to copy in labels_array and result
    #pragma acc enter data copyin(labels_array[0:images_num], result[0:images_num*num_classes])
    // Iterate over each example in the batch
    #pragma acc parallel loop reduction(+:total_loss) present(result[0:images_num*num_classes], labels_array[0:images_num])
    for (size_t i = 0; i < images_num; ++i)
    {
        // Extract the true label for the current example
        unsigned char true_label = labels_array[i];

        // Extract the logits for the current example
        const float *logits = result + i * num_classes;

        // Compute the log-softmax for numerical stability
        float max_logit = *std::max_element(logits, logits + num_classes);
        float exp_sum = 0.0f;
        // Temporary array to store modified logits
        float temp_logits[num_classes];
        std::fill(temp_logits, temp_logits + num_classes, 0.0); // Initialize temp_logits with zeros

        #pragma acc loop reduction(+:exp_sum)
        for (size_t j = 0; j < num_classes; ++j)
        {
            temp_logits[j] = std::exp(logits[j] - max_logit);
            exp_sum += temp_logits[j];
        }

        // Compute the log-softmax for the true class
        float log_softmax_true_class = std::log(temp_logits[true_label] / exp_sum);

        // Accumulate the loss
        total_loss += -log_softmax_true_class;
    }

    // Exit data region to copy out labels_array and result
    #pragma acc exit data copyout(labels_array[0:images_num], result[0:images_num*num_classes])

    // Compute the mean loss
    return total_loss / static_cast<float>(images_num);
}


float mean_err_openacc(float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // Count the total number of errors
    size_t total_errors = 0;

    // Explicitly copy labels_array to the device
    #pragma acc enter data copyin(labels_array[0:images_num], result[0:images_num*num_classes]) 
    // Iterate over each example in the batch
    #pragma acc parallel loop reduction(+:total_errors) present(result[0:images_num*num_classes], labels_array[0:images_num])
    for (size_t i = 0; i < images_num; ++i)
    {
        // Extract the logit predictions and true label for the current example
        const float *logits = result + i * num_classes;
        unsigned char true_label = labels_array[i];

        // Find the predicted class with the maximum logit
        size_t predicted_class = std::distance(logits, std::max_element(logits, logits + num_classes));

        // Update the total error count if the predicted class is incorrect
        total_errors += (predicted_class != static_cast<size_t>(true_label)) ? 1 : 0;
    }

    // Explicitly copy labels_array back to the host
    #pragma acc exit data copyout(labels_array[0:images_num], result[0:images_num*num_classes])

    // Calculate the average error rate over all examples
    return static_cast<float>(total_errors) / static_cast<float>(images_num);
}

void matrix_mul_openacc(float *A, const float *B, size_t size)
{
    #pragma acc data copyin(A[0:size], B[0:size]) copyout(A[0:size])
    #pragma acc parallel loop
    for (size_t i = 0; i < size; ++i){
        A[i] *= B[i];
    }
}


void nn_epoch_cpp_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch) {
    // Loop through the data in batches
    for (size_t i = 0; i < m; i += batch) {
        // Extract a batch of input data
        const float *batch_X = &X[i * n];  // X_b = X[i : i + batch]

        // Forward pass
        float *Z1 = new float[batch * l];
        std::fill(Z1, Z1 + batch * l, 0.0); // Initialize Z1 with zeros

        // Calculate the dot products for the entire batch and hidden units
        matrix_dot_openacc(batch_X, W1, Z1, batch, n, l);  // Z1 = np.maximum(0, np.dot(X_b, W1))

        // Apply ReLU activation to the entire Z1 matrix
        #pragma acc data copyin(Z1[0:batch*l]) copyout(Z1[0:batch*l])
        {
            #pragma acc parallel loop collapse(2) present(Z1[0:batch*l])
            for (size_t j = 0; j < batch; ++j) {
                for (size_t h = 0; h < l; ++h) {
                    Z1[j * l + h] = fmax(0.0f, Z1[j * l + h]);
                }
            }
        } // End of data region

        // Compute exponential for softmax
        float *h_Z1_exp = new float[batch * k];
        std::fill(h_Z1_exp, h_Z1_exp + batch * k, 0.0); // Initialize h_Z1_exp with zero

        // Calculate the dot products for the entire batch and output units (h_Z1_exp = Z1 * W2)
        matrix_dot_openacc(Z1, W2, h_Z1_exp, batch, l, k);  // h_Z1_exp = np.exp(np.dot(Z1, W2))

        // Softmax activation
        matrix_softmax_normalize_openacc(h_Z1_exp, batch, k);  // Z2 = h_Z1_exp / np.sum(h_Z1_exp, axis=1)[:, None]

        // Convert labels to one-hot encoding using the provided function
        float *Y = new float[batch * k];
        std::fill(Y, Y + batch * k, 0.0); // Initialize Y with zeros
        vector_to_one_hot_matrix_openacc(y + i, Y, batch, k);  // Y = np.zeros(Z2.shape, np.float32), Y[np.arange(y[i : i + batch].size), y[i : i + batch]] = 1

        // Backward pass (calculate gradients)
        // Create a matrix for (Z2 - Y)
        float *Z2_minus_Y = new float[batch * k];
        std::fill(Z2_minus_Y, Z2_minus_Y + batch * k, 0.0); // Initialize Z2_minus_Y with zeros
        // Now use matrix_copy to copy values from Z2 to Z2_minus_Y
        matrix_copy_openacc(Z2_minus_Y, h_Z1_exp, batch, k);
        // Perform matrix_minus on Z2_minus_Y and Y
        matrix_minus(Z2_minus_Y, Y, batch, k);

        // Calculate G1 using matrix_trans_dot
        float *G1 = new float[batch * l];
        std::fill(G1, G1 + batch * l, 0.0); // Initialize G1 with zeros
        // Perform matrix_trans_dot dot(Z2 - Y, W2.T)
        matrix_trans_dot_openacc(Z2_minus_Y, W2, G1, batch, k, l);  // G1 = np.dot(Z2 - Y, W2.T)

        #pragma acc parallel loop independent
        // Apply element-wise multiplication with ReLU derivative (Z1 > 0)
        for (size_t idx = 0; idx < batch * l; ++idx) {
            G1[idx] *= (Z1[idx] > 0.0f) ? 1.0f : 0.0f;  // G1 = G1 * (Z1 > 0)
        }

        // Update weights using gradients and learning rate
        // Calculate W1_l using matrix_dot_trans and matrix_mul_scalar
        float *W1_l = new float[n * l];
        std::fill(W1_l, W1_l + n * l, 0.0); // Initialize W1_l with zeros

        matrix_dot_trans_openacc(batch_X, G1, W1_l, n, batch, l);  // W1_l = np.dot(X_b.T, G1) / batch * lr
        matrix_mul_scalar_openacc(W1_l, lr / batch, n, l);

        // Calculate W2_l using matrix_dot_trans and matrix_mul_scalar
        float *W2_l = new float[l * k];
        std::fill(W2_l, W2_l + l * k, 0.0); // Initialize W2_l with zeros
        matrix_dot_trans_openacc(Z1, Z2_minus_Y, W2_l, l, batch, k);  // W2_l = np.dot(Z1.T, Z2 - Y) / batch * lr
        matrix_mul_scalar_openacc(W2_l, lr / batch, l, k);

        // Update W1 and W2
        matrix_minus(W1, W1_l, n, l);
        matrix_minus(W2, W2_l, l, k);
    }
}

void evaluate_nn_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float *result) {
    // Forward pass
    float *Z1 = new float[m * l];
    std::fill(Z1, Z1 + m * l, 0.0); // Initialize Z1 with zeros

    matrix_dot_openacc(X, W1, Z1, m, n, l);

    // Apply ReLU activation
    #pragma acc data copyin(Z1[0:m*l]) copyout(Z1[0:m*l])
    {
        #pragma acc parallel loop collapse(2) present(Z1[0:m*l])
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < l; ++j) {
                Z1[i * l + j] = fmax(0.0f, Z1[i * l + j]);
            }
        }
    } // End of data region

    // Compute exponential for softmax
    float *h_Z1_exp = new float[m * k];
    std::fill(h_Z1_exp, h_Z1_exp + m * k, 0.0); // Initialize h_Z1_exp with zeros

    matrix_dot_openacc(Z1, W2, h_Z1_exp, m, l, k);

    // Copy result to the output
    std::copy(h_Z1_exp, h_Z1_exp + m * k, result);

    // Cleanup
    delete[] Z1;
    delete[] h_Z1_exp;
}

void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_w1 = train_data->input_dim * hidden_dim;
    size_t size_w2 = hidden_dim * num_classes;
    float *W1 = new float[size_w1];
    float *W2 = new float[size_w2];
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(0.0, 1.0);
    for (size_t i = 0; i < size_w1; i++)
    {
        W1[i] = dist(rng);
    }
    for (size_t i = 0; i < size_w2; i++)
    {
        W2[i] = dist(rng);
    }
    matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
    matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
  
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        nn_epoch_cpp_openacc(train_data->images_matrix, train_data->labels_array, W1, W2, train_data->images_num, train_data->input_dim, hidden_dim, num_classes, lr, batch);


        // Evaluate on training data
        evaluate_nn_openacc(train_data->images_matrix, train_data->labels_array, W1, W2, train_data->images_num, train_data->input_dim, hidden_dim, num_classes, train_result);

        // Evaluate on test data
        evaluate_nn_openacc(test_data->images_matrix, test_data->labels_array, W1, W2, test_data->images_num, test_data->input_dim, hidden_dim, num_classes, test_result);

        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
    // END YOUR CODE
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
