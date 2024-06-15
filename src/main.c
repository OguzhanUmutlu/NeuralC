#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "network.h"
#include "utils.h"

int main(void) {
    NeuralNetwork* nn = nn_load_from_file("./model.txt");

    double* result = nn_run_str(nn, strdup("1,0"));

    printf("Result: %f\n", result[0]);

    free(result);
    nn_free(nn);
    printf("\n\n");
    return 0;
}