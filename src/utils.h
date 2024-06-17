#ifndef NN_UTILS_H
#define NN_UTILS_H

#include "network.h"

long long current_time_millis();

#define throw_err(...)            \
    fprintf(stderr, __VA_ARGS__); \
    exit(1)

#define malloc_check(var)                                                                  \
    if (var == NULL) {                                                                     \
        throw_err("Memory allocation failed in line %d in file " __FILE__ "\n", __LINE__); \
    }

#define rm_free(t) \
    free(t);       \
    t = NULL

TrainData* make_train_data_str(NeuralNetwork* nn, char* str, uint* data_size);
NeuralNetwork* nn_make_str(uint input_size, char* hidden, uint output_size);
void print_train_data(NeuralNetwork* nn, TrainData data);
void print_train_data_list(NeuralNetwork* nn, TrainData* data_list, uint data_amount);
double* nn_run_str(NeuralNetwork* nn, char* str);
char* nn_to_string(NeuralNetwork* nn);
NeuralNetwork* nn_from_string(char* str);
void nn_save_to_file(NeuralNetwork* nn, char* filepath);
NeuralNetwork* nn_load_from_file(char* filepath);
TrainData* load_train_data_from_file(NeuralNetwork* nn, char* filepath, uint* data_size);

#endif