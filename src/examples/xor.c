#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "../network.h"
#include "../utils.h"

void example_xor() {
    NeuralNetwork* nn = nn_make_str(2, strdup("3"), 1);

    uint data_size;
    TrainData* data_list = make_train_data_str(nn, strdup("0,0=0\n0,1=1\n1,0=1\n1,1=0"), &data_size);

    nn_train(nn, data_list, data_size);

    double* result = nn_run_str(nn, strdup("1,0"));

    printf("Result: %f\n", result[0]);

    data_list_free(data_list, data_size);
    rm_free(result);
    nn_free(nn);
    printf("\n\n");
}