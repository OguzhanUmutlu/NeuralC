#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <math.h>
#include <stdbool.h>

typedef unsigned int uint;

struct NeuralNetwork;
typedef struct NeuralNetwork NeuralNetwork;

struct NeuralNetwork {
    double binary_thresh;
    double error_thresh;
    uint max_iterations;
    double learning_rate;
    double momentum;
    int timeout;  // in seconds
    uint log_period;

    uint layer_amount;
    uint* sizes;
    double** biases;
    double*** weights;
    double*** changes;
    double** activations;
    double** deltas;
    double** errors;
};

typedef struct {
    double* inputs;
    double* outputs;
} TrainData;

typedef struct {
    double error;
    uint iterations;
    uint time_elapsed;
} TrainingReport;

NeuralNetwork* nn_make(uint input_size, uint* hidden_layers, uint hidden_amount, uint output_size);
double* nn_run(NeuralNetwork* nn, double* inputs);
TrainingReport nn_train(NeuralNetwork* nn, TrainData* data_list, uint data_amount);
void nn_free(NeuralNetwork* nn);
void data_list_free(TrainData* list, uint data_amount);
void train_data_free(TrainData data);
void nn_randomize_wb(NeuralNetwork* nn);
void nn_init_runtime(NeuralNetwork* nn);

double sigmoid(double x);
double relu(double x);

#endif