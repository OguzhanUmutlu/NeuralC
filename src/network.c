#include "network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "utils.h"

double nn_calculate_training_error(NeuralNetwork* nn, TrainData* data_list, uint data_amount);
double nn_train_pattern(NeuralNetwork* nn, TrainData data);
void nn_adjust_weights(NeuralNetwork* nn);
void nn_calculate_deltas(NeuralNetwork* nn, double* target);
void nn_run_internal(NeuralNetwork* nn, double* inputs);

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
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

double mse(double* errors, uint errors_size) {
    double sum = 0;
    for (int i = 0; i < errors_size; i++) {
        sum += errors[i] * errors[i];
    }
    return sum / errors_size;
}

NeuralNetwork* nn_make(uint input_size, uint* hidden_layers, uint hidden_amount, uint output_size) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    malloc_check(nn);
    nn->binary_thresh = 0.5;
    nn->error_thresh = 0.00005;
    nn->max_iterations = 20000;
    nn->learning_rate = 0.3;
    nn->momentum = 0.1;
    nn->timeout = 0;

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
        for (int neuron = 0; neuron < size; neuron++) {
            double* weights = layer_weights[neuron];
            double sum = layer_biases[neuron];
            for (int k = 0; k < prev_size; k++) {
                sum += weights[k] * inputs[k];
            }
            layer_activations[neuron] = sigmoid(sum);
        }
        inputs = layer_activations;
    }
}

TrainingReport nn_train(NeuralNetwork* nn, TrainData* data_list, uint data_amount) {
    double error = 1;
    uint iterations = 0;
    int start_time = (int)time(NULL);
    int end_time = nn->timeout && start_time + nn->timeout;

    while (1) {
        time_t t = time(NULL);
        if (iterations >= nn->max_iterations ||
            error <= nn->error_thresh ||
            (end_time != 0 && t >= end_time)) {
            return (TrainingReport){error, iterations, t - start_time};
        }
        iterations++;
        error = nn_calculate_training_error(nn, data_list, data_amount);
    }
}

double nn_calculate_training_error(NeuralNetwork* nn, TrainData* data_list, uint data_amount) {
    double sum = 0;
    for (int i = 0; i < data_amount; ++i) {
        sum += nn_train_pattern(nn, data_list[i]);
    }
    return sum / data_amount;
}

double nn_train_pattern(NeuralNetwork* nn, TrainData data) {
    nn_run_internal(nn, data.inputs);
    nn_calculate_deltas(nn, data.outputs);
    nn_adjust_weights(nn);
    nn->activations[0] = NULL;
    return mse(nn->errors[nn->layer_amount - 1], nn->sizes[nn->layer_amount - 1]);
}

void nn_calculate_deltas(NeuralNetwork* nn, double* target) {
    for (int layer = nn->layer_amount - 1; layer >= 0; layer--) {
        uint active_size = nn->sizes[layer];
        uint next_size = nn->sizes[layer + 1];
        double* active_values = nn->activations[layer];
        double* active_error = nn->errors[layer];
        double* active_deltas = nn->deltas[layer];
        double** next_layer = nn->weights[layer + 1];
        for (int neuron = 0; neuron < active_size; neuron++) {
            double output = active_values[neuron];
            double error = 0;
            if (layer == nn->layer_amount - 1) {
                error = target[neuron] - output;
            } else {
                double* deltas = nn->deltas[layer + 1];
                for (int k = 0; k < next_size; k++) {
                    error += deltas[k] * next_layer[k][neuron];
                }
            }
            active_error[neuron] = error;
            active_deltas[neuron] = error * output * (1 - output);
        }
    }
}

void nn_adjust_weights(NeuralNetwork* nn) {
    for (int layer = 1; layer < nn->layer_amount; layer++) {
        uint prev_size = nn->sizes[layer - 1];
        double* incoming = nn->activations[layer - 1];
        uint active_size = nn->sizes[layer];
        double* active_delta = nn->deltas[layer];
        double** active_changes = nn->changes[layer];
        double** active_weights = nn->weights[layer];
        double* active_biases = nn->biases[layer];
        for (int neuron = 0; neuron < active_size; neuron++) {
            double delta = active_delta[neuron];
            for (int k = 0; k < prev_size; k++) {
                double change = nn->learning_rate * delta * incoming[k] + nn->momentum * active_changes[neuron][k];
                active_changes[neuron][k] = change;
                active_weights[neuron][k] += change;
            }
            active_biases[neuron] += nn->learning_rate * delta;
        }
    }
}

void nn_free(NeuralNetwork* nn) {
    for (int layer = 0; layer < nn->layer_amount; layer++) {
        uint size = nn->sizes[layer];
        free(nn->biases[layer]);
        free(nn->deltas[layer]);
        free(nn->errors[layer]);
        if (layer > 0) {
            free(nn->activations[layer]);
            uint prev_size = nn->sizes[layer - 1];
            for (int neuron_index = 0; neuron_index < size; neuron_index++) {
                free(nn->weights[layer][neuron_index]);
                free(nn->changes[layer][neuron_index]);
            }
            free(nn->weights[layer]);
            free(nn->changes[layer]);
        }
    }
    free(nn->sizes);
    free(nn->biases);
    free(nn->activations);
    free(nn->deltas);
    free(nn->errors);
    free(nn->weights);
    free(nn->changes);
    free(nn);
}

void data_list_free(TrainData* list, uint data_amount) {
    for (int i = 0; i < data_amount; i++) {
        train_data_free(list[i]);
    }
}

void train_data_free(TrainData data) {
    free(data.inputs);
    free(data.outputs);
}