#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "network.h"
#include "utils.h"

void example_save();
void example_load();

int main(void) {
    example_save();
    return 0;
}

void example_save() {
    NeuralNetwork* nn = nn_make_str(2, "3", 1);

    uint data_size;
    TrainData* data_list = make_train_data_str(nn, strdup("0,0=0\n0,1=1\n1,0=1\n1,1=0"), &data_size);

    nn_train(nn, data_list, data_size);

    double* result = nn_run_str(nn, strdup("1,0"));

    printf("Result: %f\n", result[0]);

    nn_save_to_file(nn, "./my_neural_network.txt");

    data_list_free(data_list, data_size);
    free(result);
    nn_free(nn);
    printf("\n\n");
}

void example_load() {
    NeuralNetwork* nn = nn_load_from_file("./my_neural_network.txt");

    double* result = nn_run_str(nn, strdup("1,0"));

    printf("Result: %f\n", result[0]);

    free(result);
    nn_free(nn);
    printf("\n\n");
}