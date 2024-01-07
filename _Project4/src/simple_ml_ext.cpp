#include "simple_ml_ext.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>

DataSet::DataSet(size_t images_num, size_t input_dim)
    : images_num(images_num), input_dim(input_dim)
{
    images_matrix = new float[images_num * input_dim];
    labels_array = new unsigned char[images_num];
}

DataSet::~DataSet()
{
    delete[] images_matrix;
    delete[] labels_array;
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

/**
 *Read an images and labels file in MNIST format.  See this page:
 *http://yann.lecun.com/exdb/mnist/ for a description of the file format.
 *Args:
 *    image_filename (str): name of images file in MNIST format (idx3-ubyte)
 *    label_filename (str): name of labels file in MNIST format (idx1-ubyte)
 **/
DataSet *parse_mnist(const std::string &image_filename, const std::string &label_filename)
{
    std::ifstream images_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream labels_file(label_filename, std::ios::in | std::ios::binary);
    uint32_t magic_num, images_num, rows_num, cols_num;

    images_file.read(reinterpret_cast<char *>(&magic_num), 4);
    labels_file.read(reinterpret_cast<char *>(&magic_num), 4);

    images_file.read(reinterpret_cast<char *>(&images_num), 4);
    labels_file.read(reinterpret_cast<char *>(&images_num), 4);
    images_num = swap_endian(images_num);

    images_file.read(reinterpret_cast<char *>(&rows_num), 4);
    rows_num = swap_endian(rows_num);
    images_file.read(reinterpret_cast<char *>(&cols_num), 4);
    cols_num = swap_endian(cols_num);

    DataSet *dataset = new DataSet(images_num, rows_num * cols_num);

    labels_file.read(reinterpret_cast<char *>(dataset->labels_array), images_num);
    unsigned char *pixels = new unsigned char[images_num * rows_num * cols_num];
    images_file.read(reinterpret_cast<char *>(pixels), images_num * rows_num * cols_num);
    for (size_t i = 0; i < images_num * rows_num * cols_num; i++)
    {
        dataset->images_matrix[i] = static_cast<float>(pixels[i]) / 255;
    }

    delete[] pixels;

    return dataset;
}

/**
 *Print Matrix
 *Print the elements of a matrix A with size m * n.
 *Args:
 *      A (float*): Matrix of size m * n
 **/
void print_matrix(float *A, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/**
 * Matrix Copy
 * Copy values from matrix B to matrix A
 * Args:
 *     A (float*): Destination matrix of size m * n
 *     B (const float*): Source matrix of size m * n
 *     m (size_t): Number of rows
 *     n (size_t): Number of columns
 **/
void matrix_copy(float *A, const float *B, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i * n + j] = B[i * n + j];
        }
    }
}

/**
 * Matrix Dot Multiplication
 * Efficiently compute C = A.dot(B)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            C[i * k + j] = 0;

            // Unroll the innermost loop
            size_t l;
            for (l = 0; l < n - 4; l += 4)
            {
                C[i * k + j] += A[i * n + l] * B[l * k + j] +
                                A[i * n + l + 1] * B[(l + 1) * k + j] +
                                A[i * n + l + 2] * B[(l + 2) * k + j] +
                                A[i * n + l + 3] * B[(l + 3) * k + j];
            }

            // Handle the remaining elements (if n is not divisible by 4)
            for (; l < n; ++l)
            {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size n * m
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot_trans(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            C[i * k + j] = 0;

            // Unroll the innermost loop
            size_t l;
            for (l = 0; l < n - 4; l += 4)
            {
                C[i * k + j] += A[l * m + i] * B[l * k + j] +
                                A[(l + 1) * m + i] * B[(l + 1) * k + j] +
                                A[(l + 2) * m + i] * B[(l + 2) * k + j] +
                                A[(l + 3) * m + i] * B[(l + 3) * k + j];
            }

            // Handle the remaining elements (if n is not divisible by 4)
            for (; l < n; ++l)
            {
                C[i * k + j] += A[l * m + i] * B[l * k + j];
            }
        }
    }
}


/**
 * Matrix Dot Multiplication Trans Version 2
 * Efficiently compute C = A.dot(B.T)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size k * n
 *     C (float*): Matrix of size m * k
 **/
void matrix_trans_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            C[i * k + j] = 0;

            // Unroll the innermost loop
            size_t l;
            for (l = 0; l < n - 4; l += 4)
            {
                C[i * k + j] += A[i * n + l] * B[j * n + l] +
                                A[i * n + l + 1] * B[j * n + l + 1] +
                                A[i * n + l + 2] * B[j * n + l + 2] +
                                A[i * n + l + 3] * B[j * n + l + 3];
            }

            // Handle the remaining elements (if n is not divisible by 4)
            for (; l < n; ++l)
            {
                C[i * k + j] += A[i * n + l] * B[j * n + l];
            }
        }
    }
}


/**
 * Matrix Minus
 * Efficiently compute A = A - B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_minus(float *A, const float *B, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i * n + j] -= B[i * n + j];
        }
    }
}

/**
 * Matrix Multiplication Scalar
 * For each element C[i] of C, C[i] *= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_mul_scalar(float *C, float scalar, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i){
        for (size_t j = 0; j < n; ++j){
            C[i * n + j] *= scalar;
        }
    }
}

/**
 * Matrix Division Scalar
 * For each element C[i] of C, C[i] /= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_div_scalar(float *C, float scalar, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i){
        for (size_t j = 0; j < n; ++j){
            C[i * n + j] /= scalar;
        }
    }
}

/**
 * Matrix Softmax Normalize
 * For each row of the matrix, we do softmax normalization
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void matrix_softmax_normalize(float *C, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        // Find the maximum value in the row
        float max_val = C[i * n];
        for (size_t j = 1; j < n; ++j) {
            max_val = std::max(max_val, C[i * n + j]);
        }

        // Calculate the exponential and sum_exp
        float sum_exp = 0.0;
        for (size_t j = 0; j < n; ++j) {
            C[i * n + j] = std::exp(C[i * n + j] - max_val);
            sum_exp += C[i * n + j];
        }

        // Normalize by dividing by sum_exp
        for (size_t j = 0; j < n; ++j) {
            C[i * n + j] /= sum_exp;
        }
    }
}

/**
 * Vector to One-Hot Matrix
 * Transform a label vector y to the one-hot encoding matrix Y
 * Args:
 *     y (unsigned char *): vector of size m * 1
 *     Y (float*): Matrix of size m * n
 **/
void vector_to_one_hot_matrix(const unsigned char *y, float *Y, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            Y[i * n + j] = (j == static_cast<size_t>(y[i])) ? 1.0 : 0.0;
        }
    }
}

/**
 * A C++ version of the softmax regression epoch code.  This should run a
 * single epoch over the data defined by X and y (and sizes m,n,k), and
 * modify theta in place.  Your function will probably want to allocate
 * (and then delete) some helper arrays to store the logits and gradients.
 *
 * Args:
 *     X (const float *): pointer to X data, of size m*n, stored in row
 *          major (C) format
 *     y (const unsigned char *): pointer to y data, of size m
 *     theta (float *): pointer to theta data, of size n*k, stored in row
 *          major (C) format
 *     m (size_t): number of examples
 *     n (size_t): input dimension
 *     k (size_t): number of classes
 *     lr (float): learning rate / SGD step size
 *     batch (int): size of SGD batch
 *
 * Returns:
 *     (None)
 */
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, float *theta, size_t m, size_t n, size_t k, float lr, size_t batch) {
    for (size_t i = 0; i < m; i += batch) {
        const float *X_b = X + (i * n);
        size_t current_batch = std::min(batch, m - i);

        // Create one-hot matrix Y using vector_to_one_hot_matrix function
        float *Y = new float[current_batch * k];
        vector_to_one_hot_matrix(&y[i], Y, current_batch, k);

        // Compute logits using matrix_dot function
        float *logits = new float[current_batch * k];
        matrix_dot(X_b, theta, logits, current_batch, n, k);

        // Apply softmax using matrix_softmax_normalize function
        matrix_softmax_normalize(logits, current_batch, k);

        // Compute gradients
        float *gradients = new float[n * k];
        std::fill(gradients, gradients + n * k, 0.0); // Initialize gradients with zeros
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < current_batch; ++l) {
                for (size_t p = 0; p < k; ++p) {
                    gradients[j * k + p] += X_b[l * n + j] * (logits[l * k + p] - Y[l * k + p]);
                }
            }
        }
        for (size_t j = 0; j < n * k; ++j) {
            gradients[j] /= batch;
            gradients[j] *= lr;
        }

        // Update theta
        matrix_minus(theta, gradients, n, k);

        // Clean up dynamically allocated memory
        delete[] Y;
        delete[] logits;
        delete[] gradients;
    }
}

/**
 *Example function to fully train a softmax classifier
 **/
void train_softmax(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // Softmax regression epoch
        softmax_regression_epoch_cpp(train_data->images_matrix, train_data->labels_array, theta, train_data->images_num, train_data->input_dim, num_classes, lr, batch);

        // Evaluate on training and test data
        matrix_dot(train_data->images_matrix, theta, train_result, train_data->images_num, train_data->input_dim, num_classes);
        matrix_dot(test_data->images_matrix, theta, test_result, test_data->images_num, test_data->input_dim, num_classes);

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
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

/*
 *Return softmax loss.  Note that for the purposes of this assignment,
 *you don't need to worry about "nicely" scaling the numerical properties
 *of the log-sum-exp computation, but can just compute this directly.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average softmax loss over the sample.
 */
float mean_softmax_loss(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // Initialize the total loss
    float total_loss = 0.0f;

    // Iterate over each example in the batch
    for (size_t i = 0; i < images_num; ++i)
    {
        // Extract the true label for the current example
        unsigned char true_label = labels_array[i];

        // Extract the logits for the current example
        const float *logits = result + i * num_classes;

        // Compute the log-softmax for numerical stability
        float max_logit = *std::max_element(logits, logits + num_classes);
        float exp_sum = 0.0f;
        float temp_logits[num_classes];  // Temporary array to store modified logits

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

    // Compute the mean loss
    return total_loss / static_cast<float>(images_num);
}

/*
 *Return error.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average error over the sample.
 */
float mean_err(float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // Count the total number of errors
    size_t total_errors = 0;

    // Iterate over each example in the batch
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

    // Calculate the average error rate over all examples
    return static_cast<float>(total_errors) / static_cast<float>(images_num);
}

/**
 * Matrix Multiplication
 * Efficiently compute A = A * B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_mul(float *A, const float *B, size_t size)
{
    for (size_t i = 0; i < size; ++i){
        A[i] *= B[i];
    }
}

/*
Run a single epoch of SGD for a two-layer neural network defined by the
weights W1 and W2 (with no bias terms):
    logits = ReLU(X * W1) * W2
The function should use the step size lr, and the specified batch size (and
again, without randomizing the order of X).  It should modify the
W1 and W2 matrices in place.
Args:
    X: 1D input array of size
        (num_examples x input_dim).
    y: 1D class label array of size (num_examples,)
    W1: 1D array of first layer weights, of shape
        (input_dim x hidden_dim)
    W2: 1D array of second layer weights, of shape
        (hidden_dim x num_classes)
    m: num_examples
    n: input_dim
    l: hidden_dim
    k: num_classes
    lr (float): step size (learning rate) for SGD
    batch (int): size of SGD batch
*/
void nn_epoch_cpp(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch) {
    // Loop through the data in batches
    for (size_t i = 0; i < m; i += batch) {
        // Extract a batch of input data
        const float *batch_X = &X[i * n];  // X_b = X[i : i + batch]

        // Forward pass
        float *Z1 = new float[batch * l];

        // Calculate the dot products for the entire batch and hidden units
        matrix_dot(batch_X, W1, Z1, batch, n, l);  // Z1 = np.maximum(0, np.dot(X_b, W1))

        // Apply ReLU activation to the entire Z1 matrix
        for (size_t j = 0; j < batch; ++j) {
            for (size_t h = 0; h < l; ++h) {
                Z1[j * l + h] = std::max(0.0f, Z1[j * l + h]);
            }
        }

        // Compute exponential for softmax
        float *h_Z1_exp = new float[batch * k];

        // Calculate the dot products for the entire batch and output units (h_Z1_exp = Z1 * W2)
        matrix_dot(Z1, W2, h_Z1_exp, batch, l, k);  // h_Z1_exp = np.exp(np.dot(Z1, W2))

        // Softmax activation
        matrix_softmax_normalize(h_Z1_exp, batch, k);  // Z2 = h_Z1_exp / np.sum(h_Z1_exp, axis=1)[:, None]

        // Convert labels to one-hot encoding using the provided function
        float *Y = new float[batch * k];
        vector_to_one_hot_matrix(y + i, Y, batch, k);  // Y = np.zeros(Z2.shape, np.float32), Y[np.arange(y[i : i + batch].size), y[i : i + batch]] = 1

        // Backward pass (calculate gradients)
        // Create a matrix for (Z2 - Y)
        float *Z2_minus_Y = new float[batch * k];
        // Now use matrix_copy to copy values from Z2 to Z2_minus_Y
        matrix_copy(Z2_minus_Y, h_Z1_exp, batch, k);
        // Perform matrix_minus on Z2_minus_Y and Y
        matrix_minus(Z2_minus_Y, Y, batch, k);

        // Calculate G1 using matrix_trans_dot
        float *G1 = new float[batch * l];
        // Perform matrix_trans_dot dot(Z2 - Y, W2.T)
        matrix_trans_dot(Z2_minus_Y, W2, G1, batch, k, l);  // G1 = np.dot(Z2 - Y, W2.T)

        // Apply element-wise multiplication with ReLU derivative (Z1 > 0)
        for (size_t idx = 0; idx < batch * l; ++idx) {
            G1[idx] *= (Z1[idx] > 0.0f) ? 1.0f : 0.0f;  // G1 = G1 * (Z1 > 0)
        }

        // Update weights using gradients and learning rate
        // Calculate W1_l using matrix_dot_trans and matrix_mul_scalar
        float *W1_l = new float[n * l];
        matrix_dot_trans(batch_X, G1, W1_l, n, batch, l);  // W1_l = np.dot(X_b.T, G1) / batch * lr
        matrix_mul_scalar(W1_l, lr / batch, n, l);

        // Calculate W2_l using matrix_dot_trans and matrix_mul_scalar
        float *W2_l = new float[l * k];
        matrix_dot_trans(Z1, Z2_minus_Y, W2_l, l, batch, k);  // W2_l = np.dot(Z1.T, Z2 - Y) / batch * lr
        matrix_mul_scalar(W2_l, lr / batch, l, k);

        // Update W1 and W2
        matrix_minus(W1, W1_l, n, l);
        matrix_minus(W2, W2_l, l, k);
        
        // Cleanup: Release memory for temporary variables
        delete[] Z1;
        delete[] h_Z1_exp;
        delete[] Y;
        delete[] Z2_minus_Y;
        delete[] G1;
        delete[] W1_l;
        delete[] W2_l;
    }
}

void evaluate_nn(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float *result) {
    // Forward pass
    float *Z1 = new float[m * l];
    matrix_dot(X, W1, Z1, m, n, l);

    // Apply ReLU activation
    for (size_t i = 0; i < m * l; ++i) {
        Z1[i] = std::max(0.0f, Z1[i]);
    }

    // Compute exponential for softmax
    float *h_Z1_exp = new float[m * k];
    matrix_dot(Z1, W2, h_Z1_exp, m, l, k);

    // Copy result to the output
    std::copy(h_Z1_exp, h_Z1_exp + m * k, result);

    // Cleanup
    delete[] Z1;
    delete[] h_Z1_exp;
}


/**
 *Example function to fully train a nn classifier
 **/
void train_nn(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_w1 = train_data->input_dim * hidden_dim;
    size_t size_w2 = hidden_dim * num_classes;
    float *W1 = new float[size_w1];
    float *W2 = new float[size_w2];

    // Initialization of W1 and W2
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

    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;

    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        nn_epoch_cpp(train_data->images_matrix, train_data->labels_array, W1, W2, train_data->images_num, train_data->input_dim, hidden_dim, num_classes, lr, batch);


        // Evaluate on training data
        evaluate_nn(train_data->images_matrix, train_data->labels_array, W1, W2, train_data->images_num, train_data->input_dim, hidden_dim, num_classes, train_result);

        // Evaluate on test data
        evaluate_nn(test_data->images_matrix, test_data->labels_array, W1, W2, test_data->images_num, test_data->input_dim, hidden_dim, num_classes, test_result);

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

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";

    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
