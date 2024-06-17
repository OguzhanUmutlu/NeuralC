#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../network.h"
#include "../utils.h"

uint find_max_ind(double* list, uint size) {
    double max = 0;
    int ind = 0;
    for (int i = 0; i < size; i++) {
        if (list[i] > max) {
            ind = i;
            max = list[i];
        }
    }
    return ind;
}

#define def_time() long long _tim
#define st_time() _tim = current_time_millis()
#define get_time() (double)(current_time_millis() - _tim) / 1000
#define log_time(str) printf(str " in %.3f seconds.\n", get_time())

void example_image_classification() {
    def_time();

    st_time();
    NeuralNetwork* nn = nn_load_from_file("./model.txt");
    bool valid_nn = nn != NULL;

    if (valid_nn) {
        log_time("Model has been loaded");
    }

    if (!valid_nn) {
        nn = nn_make_str(784, strdup("128,64"), 10);
        nn->max_iterations = 20000;
        nn->error_thresh = 0.005;
    }

    st_time();
    uint data_size;
    TrainData* data_list = load_train_data_from_file(nn, "./data.txt", &data_size);
    log_time("Dataset has been loaded");
    data_size = 100;

    if (!valid_nn) {
        printf("Couldn't find the model! Training a new one...\n");
        st_time();
        TrainingReport report = nn_train(nn, data_list, data_size);
        log_time("Training has been completed");
        printf("Training error: %f, iterations: %d\n", report.error, report.iterations);
    }

    st_time();
    double* result = nn_run(nn, data_list[0].inputs);
    log_time("Test case has been inputted");

    uint ind = find_max_ind(result, 10);
    printf("Test case result: %d, sureness: %f%%\n", ind, result[ind] * 100);

    st_time();
    nn_save_to_file(nn, "./model.txt");
    log_time("Model has been saved to 'model.txt'");

    data_list_free(data_list, data_size);
    rm_free(result);
    // nn_free(nn);
    printf("\n\n");
}