#include "network.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "utils.h"

double nn_train_pattern(NeuralNetwork* nn, TrainData data);
void nn_adjust_weights(NeuralNetwork* nn);
void nn_calculate_deltas(NeuralNetwork* nn, double* target);
void nn_run_internal(NeuralNetwork* nn, double* inputs);

double sigmoid(double x) {
    if (x > 3.4517) return 0.96;
    if (x < -3.4517) return 0.001;
    // For |x| > 1.81
    // centered at x=2.357:
    // f(x) = 0.913489018914 + 0.0790268312376 * (x - 2.357)
    // f(x) = 0.727223 + 0.0790268312376 * x
    // centered at x=-2.357:
    // f(x) = 0.0865109810861 + 0.0790268312376 * (x + 2.357)
    // f(x) = 0.27277722231 + 0.0790268312376 * x
    // x > 1.81 -> σ(x) = min(f(x), 1)
    // x < -1.81 -> σ(x) = 1 - min(1 - f(x), 1)
    if (x > 1.81) return 0.727223 + 0.0790268312376 * x;
    if (x < -1.81) return 0.27277722231 + 0.0790268312376 * x;

    // Taylor approximation of the sigmoid centered at x=0, accurate in the interval [-1.274, +1.274] for x
    return 0.5 + 0.25 * x - 0.0208333333 * x * x * x + 0.0020833333 * x * x * x * x * x;
}

double relu(double x) {
    return x < 0 ? 0 : x;
}

double rand_float() {
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }

    return (double)rand() / (double)RAND_MAX;
}

double* random_weights(uint size) {
    double* weights = (double*)malloc(size * sizeof(double));
    malloc_check(weights);
    for (int i = 0; i < size; i++) {
        weights[i] = rand_float() * 0.4 - 0.2;
    }
    return weights;
}

static inline double mse(double* errors, uint errors_size) {
    double sum = 0.0;
    uint i = 0;

    __m256d vec_sum = _mm256_setzero_pd();

    for (; i + 4 <= errors_size; i += 4) {
        __m256d vec_errors = _mm256_loadu_pd(&errors[i]);
        vec_sum = _mm256_add_pd(vec_sum, _mm256_mul_pd(vec_errors, vec_errors));
    }

    double temp[4];
    _mm256_storeu_pd(temp, vec_sum);
    sum = temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < errors_size; i++) {
        sum += errors[i] * errors[i];
    }

    return sum / errors_size;
}

NeuralNetwork* nn_make(uint input_size, uint* hidden_layers, uint hidden_amount, uint output_size) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    malloc_check(nn);
    nn->binary_thresh = 0.5;
    nn->error_thresh = 0.005;
    nn->max_iterations = 20000;
    nn->learning_rate = 0.3;
    nn->momentum = 0.1;
    nn->timeout = 0;
    nn->log_period = 1000;

    int layer_amount = 2 + hidden_amount;

    nn->layer_amount = layer_amount;
    nn->sizes = (uint*)malloc(layer_amount * sizeof(uint));
    malloc_check(nn->sizes);
    nn->sizes[0] = input_size;
    nn->sizes[layer_amount - 1] = output_size;

    for (int i = 1; i < layer_amount - 1; i++) {
        nn->sizes[i] = hidden_layers[i - 1];
    }

    nn->biases = (double**)malloc(layer_amount * sizeof(double*));
    malloc_check(nn->biases);

    nn->weights = (double***)malloc(layer_amount * sizeof(double**));
    malloc_check(nn->weights);

    nn_init_runtime(nn);
    nn_randomize_wb(nn);
    return nn;
}

// Randomizes weights and biases
void nn_randomize_wb(NeuralNetwork* nn) {
    uint layer_amount = nn->layer_amount;
    nn->weights[0] = NULL;

    for (int layer = 0; layer < layer_amount; layer++) {
        uint size = nn->sizes[layer];
        if (layer > 0) {
            nn->biases[layer] = random_weights(size);
            nn->weights[layer] = (double**)malloc(size * sizeof(double*));
            malloc_check(nn->weights[layer]);
            uint prev_size = nn->sizes[layer - 1];
            for (int neuron = 0; neuron < size; neuron++) {
                nn->weights[layer][neuron] = random_weights(prev_size);
            }
        }
    }
}

// Inits runtime properties like error, activations, deltas etc.
void nn_init_runtime(NeuralNetwork* nn) {
    uint layer_amount = nn->layer_amount;

    nn->changes = (double***)malloc(layer_amount * sizeof(double**));
    malloc_check(nn->changes);

    nn->activations = (double**)malloc(layer_amount * sizeof(double*));
    malloc_check(nn->activations);
    nn->deltas = (double**)malloc(layer_amount * sizeof(double*));
    malloc_check(nn->deltas);
    nn->errors = (double**)malloc(layer_amount * sizeof(double*));
    malloc_check(nn->errors);

    for (int layer = 0; layer < layer_amount; layer++) {
        uint size = nn->sizes[layer];
        nn->deltas[layer] = (double*)malloc(size * sizeof(double));
        malloc_check(nn->deltas[layer]);
        nn->errors[layer] = (double*)malloc(size * sizeof(double));
        malloc_check(nn->errors[layer]);
        if (layer > 0) {
            nn->activations[layer] = (double*)malloc(size * sizeof(double));
            malloc_check(nn->activations[layer]);
            nn->changes[layer] = (double**)malloc(size * sizeof(double*));
            malloc_check(nn->changes[layer]);
            uint prev_size = nn->sizes[layer - 1];
            for (int neuron = 0; neuron < size; neuron++) {
                nn->changes[layer][neuron] = (double*)malloc(prev_size * sizeof(double));
                malloc_check(nn->changes[layer][neuron]);
            }
        }
    }
    nn->activations[0] = NULL;
}

double* nn_run(NeuralNetwork* nn, double* inputs) {
    nn_run_internal(nn, inputs);
    nn->activations[0] = NULL;

    uint size = nn->sizes[nn->layer_amount - 1] * sizeof(double);
    double* output = nn->activations[nn->layer_amount - 1];
    double* result = (double*)malloc(size);
    memcpy(result, output, size);

    return result;
}

void nn_run_internal(NeuralNetwork* nn, double* inputs) {
    nn->activations[0] = inputs;
    double* output;

    for (int layer = 1; layer < nn->layer_amount; layer++) {
        uint size = nn->sizes[layer];
        uint prev_size = nn->sizes[layer - 1];
        double** layer_weights = nn->weights[layer];
        double* layer_biases = nn->biases[layer];
        double* layer_activations = nn->activations[layer];

#pragma omp parallel for
        for (int neuron_to = 0; neuron_to < size; neuron_to++) {
            double* weights = layer_weights[neuron_to];
            double sum = layer_biases[neuron_to];

            for (int neuron_from = 0; neuron_from < prev_size; neuron_from++) {
                sum += weights[neuron_from] * inputs[neuron_from];
            }

            layer_activations[neuron_to] = sigmoid(sum);
        }
        inputs = layer_activations;
    }
}

TrainingReport nn_train(NeuralNetwork* nn, TrainData* data_list, uint data_amount) {
    double error = 1;
    uint iterations = 0;
    int start_time = (int)time(NULL);
    int end_time = nn->timeout && start_time + nn->timeout;
    uint log_period = nn->log_period;

    while (iterations < nn->max_iterations && error > nn->error_thresh && (end_time == 0 || time(NULL) < end_time)) {
        iterations++;
        error = 0;

        // #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < data_amount; i++) {
            error += nn_train_pattern(nn, data_list[i]);
        }

        error /= data_amount;

        if (iterations % log_period == 0) {
            printf("%d iterations in %ld seconds with error: %f\n", iterations, time(NULL) - start_time, error);
        }
    }
    return (TrainingReport){error, iterations, time(NULL) - start_time};
}

double nn_train_pattern(NeuralNetwork* nn, TrainData data) {
    nn_run_internal(nn, data.inputs);
    nn_calculate_deltas(nn, data.outputs);
    nn_adjust_weights(nn);
    nn->activations[0] = NULL;
    uint output_layer = nn->layer_amount - 1;
    return mse(nn->errors[output_layer], nn->sizes[output_layer]);
}

void nn_calculate_deltas(NeuralNetwork* nn, double* target) {
    uint layer_amount = nn->layer_amount;
    double learning_rate = nn->learning_rate;
    double momentum = nn->momentum;
    uint* sizes = nn->sizes;
    double** errors = nn->errors;
    double** activations = nn->activations;
    double** deltas = nn->deltas;
    double*** changes = nn->changes;
    double*** weights = nn->weights;
    double** biases = nn->biases;

    for (int layer = layer_amount - 1; layer >= 0; layer--) {
        uint size = sizes[layer];
        uint next_size = sizes[layer + 1];
        double* layer_activations = activations[layer];
        double* layer_errors = errors[layer];
        double* layer_deltas = deltas[layer];
        double** layer_weights = weights[layer];

        double* next_deltas = NULL;
        double** next_layer = NULL;
        if (layer < layer_amount - 1) {
            next_deltas = deltas[layer + 1];
            next_layer = weights[layer + 1];
        }

        for (int neuron_to = 0; neuron_to < size; neuron_to++) {
            double output = layer_activations[neuron_to];
            double error = 0;

            if (layer == layer_amount - 1) {
                error = target[neuron_to] - output;
            } else {
                for (int neuron_from = 0; neuron_from < next_size; neuron_from++) {
                    error += next_deltas[neuron_from] * next_layer[neuron_from][neuron_to];
                }
            }

            layer_errors[neuron_to] = error;
            layer_deltas[neuron_to] = error * output * (1 - output);
        }
    }
}

void nn_adjust_weights(NeuralNetwork* nn) {
    uint layer_amount = nn->layer_amount;
    double learning_rate = nn->learning_rate;
    double momentum = nn->momentum;
    uint* sizes = nn->sizes;
    double** activations = nn->activations;
    double** deltas = nn->deltas;
    double*** changes = nn->changes;
    double*** weights = nn->weights;
    double** biases = nn->biases;

    for (int layer = 1; layer < layer_amount; layer++) {
        uint size = sizes[layer];
        uint prev_size = sizes[layer - 1];
        double* incoming = activations[layer - 1];
        double* layer_deltas = deltas[layer];
        double** layer_changes = changes[layer];
        double** layer_weights = weights[layer];
        double* layer_biases = biases[layer];

        double learning_momentum = learning_rate * momentum;

        for (int neuron_to = 0; neuron_to < size; neuron_to++) {
            double delta = layer_deltas[neuron_to];
            double bias_change = learning_rate * delta;
            double* layer_changes_to = layer_changes[neuron_to];

            for (int neuron_from = 0; neuron_from < prev_size; neuron_from++) {
                double input = incoming[neuron_from];
                double change = bias_change * input + learning_momentum * layer_changes_to[neuron_from];
                layer_changes[neuron_to][neuron_from] = change;
                layer_weights[neuron_to][neuron_from] += change;
            }

            layer_biases[neuron_to] += bias_change;
        }
    }
}

void nn_free(NeuralNetwork* nn) {
    for (int layer = 0; layer < nn->layer_amount; layer++) {
        uint size = nn->sizes[layer];
        rm_free(nn->biases[layer]);
        rm_free(nn->deltas[layer]);
        rm_free(nn->errors[layer]);
        if (layer > 0) {
            rm_free(nn->activations[layer]);
            uint prev_size = nn->sizes[layer - 1];
            for (int neuron_index = 0; neuron_index < size; neuron_index++) {
                rm_free(nn->weights[layer][neuron_index]);
                rm_free(nn->changes[layer][neuron_index]);
            }
            rm_free(nn->weights[layer]);
            rm_free(nn->changes[layer]);
        }
    }
    rm_free(nn->sizes);
    rm_free(nn->biases);
    rm_free(nn->activations);
    rm_free(nn->deltas);
    rm_free(nn->errors);
    rm_free(nn->weights);
    rm_free(nn->changes);
    rm_free(nn);
}

void data_list_free(TrainData* list, uint data_amount) {
    for (int i = 0; i < data_amount; i++) {
        train_data_free(list[i]);
    }
}

void train_data_free(TrainData data) {
    rm_free(data.inputs);
    rm_free(data.outputs);
}