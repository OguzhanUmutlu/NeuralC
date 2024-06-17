#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "network.h"

// hidden = 1,2,3,4
NeuralNetwork *nn_make_str(uint input_size, char *hidden, uint output_size) {
    uint len = strlen(hidden);
    uint hidden_amount = 1;

    while (hidden[len - 1] == ',') {
        len--;
    }

    for (char *p = hidden; *p; p++) {
        if (*p == ',') {
            hidden_amount++;
        }
    }

    uint *hidden_list = (uint *)malloc(hidden_amount * sizeof(uint));
    malloc_check(hidden_list);
    char *token = strtok(hidden, ",");
    int i = 0;
    while (token != NULL) {
        hidden_list[i++] = (uint)atoi(token);
        token = strtok(NULL, ",");
    }

    NeuralNetwork *nn = nn_make(input_size, hidden_list, hidden_amount, output_size);

    rm_free(hidden_list);

    return nn;
}

// str = 0.5129,0.1239891,0.28923
double *nn_run_str(NeuralNetwork *nn, char *str) {
    uint len = strlen(str);
    uint input_size = nn->sizes[0];

    double *inputs = (double *)malloc(input_size * sizeof(double));
    malloc_check(inputs);
    char *token = strtok(str, ",");
    int i = 0;
    while (token != NULL) {
        inputs[i++] = strtod(token, NULL);
        token = strtok(NULL, ",");
    }

    return nn_run(nn, inputs);
}

// 1.235,2.235,3.25,4.24=1.623,2.123,3.425\n...
TrainData *make_train_data_str(NeuralNetwork *nn, char *str, uint *data_size) {
    uint input_size = nn->sizes[0];
    uint output_size = nn->sizes[nn->layer_amount - 1];

    uint len = strlen(str);
    uint list_size = 1;
    while (str[len - 1] == '\n') {
        len--;
    }

    for (char *p = str; *p; p++) {
        if (*p == '\n') list_size++;
    }

    *data_size = list_size;

    TrainData *data_list = (TrainData *)malloc(list_size * sizeof(TrainData));
    malloc_check(data_list);

    char *str_copy = strdup(str);
    char *line = strtok(str_copy, "\n");
    int index = 0;

    while (line != NULL) {
        char *equal_sign = strchr(line, '=');
        if (equal_sign == NULL) {
            fprintf(stderr, "Invalid format in line(%d): %s\n", index, line);
            exit(1);
        }

        *equal_sign = '\0';
        char *input_str = line;
        char *output_str = equal_sign + 1;

        char *token;
        int inp_size = 1, out_size = 1;

        for (char *p = input_str; *p; p++) {
            if (*p == ',') inp_size++;
        }
        for (char *p = output_str; *p; p++) {
            if (*p == ',') out_size++;
        }
        if (inp_size != input_size) {
            throw_err("training_data[%d].inputs size(%d) doesn't match the neural network's input size(%d).\n", index, inp_size, input_size);
        }
        if (out_size != output_size) {
            throw_err("training_data[%d].outputs size(%d) doesn't match the neural network's output size(%d).\n", index, out_size, output_size);
        }

        data_list[index].inputs = (double *)malloc(input_size * sizeof(double));
        malloc_check(data_list[index].inputs);
        data_list[index].outputs = (double *)malloc(output_size * sizeof(double));
        malloc_check(data_list[index].outputs);

        input_str = line;
        for (int i = 0; i < input_size; i++) {
            data_list[index].inputs[i] = strtod(input_str, &input_str);
            if (*input_str == ',') {
                input_str++;
            }
        }

        output_str = equal_sign + 1;
        for (int i = 0; i < output_size; i++) {
            data_list[index].outputs[i] = strtod(output_str, &output_str);
            if (*output_str == ',') {
                output_str++;
            }
        }

        line = strtok(NULL, "\n");
        index++;
    }
    return data_list;
}

void print_train_data_list(NeuralNetwork *nn, TrainData *data_list, uint data_amount) {
    printf("[\n");
    for (int i = 0; i < data_amount; i++) {
        printf("  ");
        print_train_data(nn, data_list[i]);
        if (i != data_amount - 1) printf(",\n");
    }
    printf("\n]\n");
}

void print_train_data(NeuralNetwork *nn, TrainData data) {
    uint input_size = nn->sizes[0];
    uint output_size = nn->sizes[nn->layer_amount - 1];

    printf("{ 'inputs': (");
    for (int i = 0; i < input_size; i++) {
        printf("%f", data.inputs[i]);
        if (i != input_size - 1) printf(", ");
    }
    printf("), 'outputs': (");
    for (int i = 0; i < output_size; i++) {
        printf("%f", data.outputs[i]);
        if (i != output_size - 1) printf(", ");
    }
    printf(") }");
}

char *nn_to_string(NeuralNetwork *nn) {
    uint input_size = nn->sizes[0];

    uint neuron_amount = 0;
    for (int i = 0; i < nn->layer_amount; i++) {
        neuron_amount += nn->sizes[i];
    }

    uint weight_amount = 0;
    for (int i = 0; i < nn->layer_amount - 1; i++) {
        weight_amount += nn->sizes[i] * nn->sizes[i + 1];
    }

    size_t str_size = 1024 + neuron_amount * 32 + weight_amount * 64;
    char *str = (char *)malloc(str_size * sizeof(char));
    malloc_check(str);

    char *ptr = str;

    ptr += sprintf(ptr, "%lf %lf %u %lf %lf %d %u %u ",
                   nn->binary_thresh, nn->error_thresh, nn->max_iterations,
                   nn->learning_rate, nn->momentum, nn->timeout, nn->log_period,
                   nn->layer_amount);

    for (int layer = 0; layer < nn->layer_amount; layer++) {
        ptr += sprintf(ptr, "%u ", nn->sizes[layer]);
    }

    for (int layer = 1; layer < nn->layer_amount; layer++) {
        for (int neuron = 0; neuron < nn->sizes[layer]; neuron++) {
            ptr += sprintf(ptr, "%lf ", nn->biases[layer][neuron]);
        }
    }

    for (int layer = 1; layer < nn->layer_amount; layer++) {
        for (int neuron_to = 0; neuron_to < nn->sizes[layer]; neuron_to++) {
            for (int neuron_from = 0; neuron_from < nn->sizes[layer - 1]; neuron_from++) {
                ptr += sprintf(ptr, "%lf ", nn->weights[layer][neuron_to][neuron_from]);
            }
        }
    }

    return str;
}

NeuralNetwork *nn_from_string(char *str) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    malloc_check(nn);

    char *ptr = str;

    int param_len;
    sscanf(ptr, "%lf %lf %u %lf %lf %d %u %u %n",
           &nn->binary_thresh, &nn->error_thresh, &nn->max_iterations,
           &nn->learning_rate, &nn->momentum, &nn->timeout, &nn->log_period,
           &nn->layer_amount, &param_len);
    ptr += param_len;

    nn->sizes = (uint *)malloc(nn->layer_amount * sizeof(uint));
    malloc_check(nn->sizes);

    for (int layer = 0; layer < nn->layer_amount; layer++) {
        sscanf(ptr, "%u", &nn->sizes[layer]);
        while (*ptr != ' ') ptr++;
        ptr++;
    }

    nn->biases = (double **)malloc(nn->layer_amount * sizeof(double *));
    malloc_check(nn->biases);

    for (int layer = 0; layer < nn->layer_amount; layer++) {
        nn->biases[layer] = (double *)malloc(nn->sizes[layer] * sizeof(double));
        malloc_check(nn->biases[layer]);
    }

    for (int layer = 1; layer < nn->layer_amount; layer++) {
        for (int neuron = 0; neuron < nn->sizes[layer]; neuron++) {
            sscanf(ptr, "%lf", &nn->biases[layer][neuron]);
            while (*ptr != ' ') ptr++;
            ptr++;
        }
    }

    nn->weights = (double ***)malloc(nn->layer_amount * sizeof(double **));
    malloc_check(nn->weights);
    nn->weights[0] = NULL;
    for (int layer = 1; layer < nn->layer_amount; layer++) {
        nn->weights[layer] = (double **)malloc(nn->sizes[layer] * sizeof(double *));
        malloc_check(nn->weights[layer]);
        for (int neuron = 0; neuron < nn->sizes[layer]; neuron++) {
            nn->weights[layer][neuron] = (double *)malloc(nn->sizes[layer - 1] * sizeof(double));
            malloc_check(nn->weights[layer][neuron]);
        }
    }
    for (int layer = 1; layer < nn->layer_amount; layer++) {
        for (int neuron_to = 0; neuron_to < nn->sizes[layer]; neuron_to++) {
            for (int neuron_from = 0; neuron_from < nn->sizes[layer - 1]; neuron_from++) {
                sscanf(ptr, "%lf", &nn->weights[layer][neuron_to][neuron_from]);
                while (*ptr != ' ') ptr++;
                ptr++;
            }
        }
    }

    nn_init_runtime(nn);

    return nn;
}

void nn_save_to_file(NeuralNetwork *nn, char *filepath) {
    FILE *file = fopen(filepath, "w");

    if (file == NULL) {
        throw_err("Error opening file.\n");
        return;
    }

    fprintf(file, "%s", nn_to_string(nn));

    fclose(file);
}

char *read_file(char *filepath) {
    FILE *file = fopen(filepath, "r");

    if (file == NULL) return NULL;

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    char *buffer = (char *)malloc((fileSize + 1) * sizeof(char));
    if (buffer == NULL) {
        fclose(file);
        return NULL;
    }

    size_t bytesRead = fread(buffer, sizeof(char), fileSize, file);
    buffer[bytesRead] = '\0';

    fclose(file);
    return buffer;
}

NeuralNetwork *nn_load_from_file(char *filepath) {
    char *buffer = read_file(filepath);
    if (buffer == NULL) return NULL;
    NeuralNetwork *nn = nn_from_string(buffer);
    rm_free(buffer);
    return nn;
}

TrainData *load_train_data_from_file(NeuralNetwork *nn, char *filepath, uint *data_size) {
    char *buffer = read_file(filepath);
    if (buffer == NULL) return NULL;
    TrainData *data_list = make_train_data_str(nn, buffer, data_size);
    rm_free(buffer);
    return data_list;
}

#ifdef _WIN32
#include <windows.h>

long long current_time_millis() {
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    ULARGE_INTEGER ull;
    ull.LowPart = ft.dwLowDateTime;
    ull.HighPart = ft.dwHighDateTime;
    return ull.QuadPart / 10000;  // Convert from 100-nanosecond intervals to milliseconds
}

#else
#include <sys/time.h>

long long current_time_millis() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000LL + tv.tv_usec / 1000;
}

#endif