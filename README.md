# NeuralC
A simple neural network made in C.

## Example usage

```c
int main(void) {
    // Create the neural network
    // First argument is a positive integer indicating how many inputs there will be.
    // Second argument is a string that has positive integers in it separated by commas. This indicates the hidden layers' sizes.
    // For example if there are 3 hidden layers with (3, 4 and 5 neurons) in them, you would do "3,4,5" for the hidden layers.
    // Third argument is a positive integer indicating how many outputs there will be.
    NeuralNetwork* nn = nn_make_str(2, "3", 1);

    // Train data can be created like this. Inputs=Outputs and each training data separated by a \n(new line.)
    // If you have this in a file it will look like this:
    // 0,0=0
    // 0,1=1
    // 1,0=1
    // 1,1=0
    uint data_size;
    TrainData* data_list = make_train_data_str(nn, strdup("0,0=0\n0,1=1\n1,0=1\n1,1=0"), &data_size);

    // This trains the neural network with the given training data
    nn_train(nn, data_list, data_size);

    // This returns a double array which indicate the output neurons' values.
    double* result = nn_run_str(nn, strdup("1,0"));

    // Prints the result out. (In this case there is one input so we do result[0])
    printf("Result: %f\n", result[0]);

    // Free everything so the program doesn't leak memory
    data_list_free(data_list, data_size);
    free(result);
    nn_free(nn);
    return 0;
}
```

## Saving and loading neural networks

### Saving neural networks

```c
NeuralNetwork* nn = ...;

// Easily save it via nn_save_to_file() function like so:
nn_save_to_file(nn, "./my_neural_network.txt");
```

### Loading neural networks

```c
// Once you've saved your neural network, you can load it like so:
NeuralNetwork* nn = nn_load_from_file("./my_neural_network.txt");
```
