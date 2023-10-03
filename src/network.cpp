#include <node_api.h>
#include <thread>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <ctime>
#include "main.h"

#define _ARG(__argc)      \
    size_t argc = __argc; \
    napi_value args[1];   \
    prepare_args(_, argv, argc, args);

#define ACTIVATION_SIGMOID 0
#define ACTIVATION_RELU 1
#define ACTIVATION_LEAKY_RELU 2
#define ACTIVATION_TANH 3

#define ACTIVATION_MIN 0
#define ACTIVATION_MAX 3

#define PRAXIS_NORMAL 0
#define PRAXIS_ADAM 1

#define PRAXIS_MIN 0
#define PRAXIS_MAX 1

#define RUN_INPUT(fn)                                                    \
    if (outputLayer < 1)                                                 \
    {                                                                    \
        NAPI_ERROR(_, "output was empty");                               \
        return std::vector<double>();                                    \
    }                                                                    \
    outputs[0] = input;                                                  \
    std::vector<double> output;                                          \
    for (int layer = 1; layer <= outputLayer; layer++)                   \
    {                                                                    \
        int activeLayer = sizes[layer];                                  \
        std::vector<std::vector<double>> activeWeights = weights[layer]; \
        std::vector<double> activeBiases = biases[layer];                \
        for (int node = 0; node < activeLayer; node++)                   \
        {                                                                \
            std::vector<double> weights = activeWeights[node];           \
            int weightsSize = weights.size();                            \
            int sum = activeBiases[node];                                \
            for (int k = 0; k < weightsSize; k++)                        \
            {                                                            \
                sum += weights[k] * input[k];                            \
            }                                                            \
            outputs[layer][node] = fn;                                   \
        }                                                                \
        input = outputs[layer];                                          \
        output = input;                                                  \
    }                                                                    \
    return output;

#define CALCULATE_DELTAS(fn)                                                     \
    int deltasSize = deltas.size();                                              \
    for (int layer = outputLayer; layer >= 0; layer--)                           \
    {                                                                            \
        int activeSize = sizes[layer];                                           \
        std::vector<double> activeOutput = outputs[layer];                       \
        for (int node = 0; node < activeSize; node++)                            \
        {                                                                        \
            double output = activeOutput[node];                                  \
            double error = 0;                                                    \
            if (layer == outputLayer)                                            \
            {                                                                    \
                error = target[node] - output;                                   \
            }                                                                    \
            else                                                                 \
            {                                                                    \
                std::vector<std::vector<double>> nextLayer = weights[layer + 1]; \
                std::vector<double> delta = deltas[layer + 1];                   \
                int deltaSize = delta.size();                                    \
                for (int k = 0; k < deltaSize; k++)                              \
                {                                                                \
                    error += delta[k] * nextLayer[k][node];                      \
                }                                                                \
            }                                                                    \
            errors[layer][node] = error;                                         \
            deltas[layer][node] = fn;                                            \
        }                                                                        \
    }

double randomDouble()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0f, 1.0f);
    return distribution(gen);
}

double randomWeight()
{
    return randomDouble() * 0.4 - 0.2;
}

double getDateNow()
{
    time_t currentTime = time(nullptr);
    return static_cast<double>(currentTime);
}

std::vector<double> randos(int size)
{
    std::vector<double> rands;
    for (int i = 0; i < size; i++)
    {
        rands.push_back(randomWeight());
    }
    return rands;
}

struct NeuralNetworkOptions
{
    NeuralNetworkOptions()
    {
        inputSize = 0;
        hiddenLayers = std::vector<int>();
        outputSize = 0;
        binaryThresh = 0.5;
    };
    int inputSize;
    std::vector<int> hiddenLayers;
    int outputSize;
    double binaryThresh;
};

struct NeuralNetworkTrainOptions
{
    short activation = ACTIVATION_SIGMOID;
    int iterations = 20000;
    double errorThresh = 0.005;
    bool log = false;
    int logPeriod = 10;
    double leakyReluAlpha = 0.01;
    double learningRate = 0.3;
    double momentum = 0.1;
    napi_value callback = nullptr;
    int callbackPeriod = 10;
    int timeout = -1;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    short praxis = PRAXIS_NORMAL;
};

struct Status
{
    double error;
    int iterations;
};

struct NeuralNetwork
{
    NeuralNetwork(napi_env _, napi_value _this) : _this(), _()
    {
        options = NeuralNetworkOptions();
        trainOptions = NeuralNetworkTrainOptions();
        outputLayer = -1;
        iterations = 0;
        errorCheckInterval = 1;
    };
    napi_value _this;
    napi_env _;
    NeuralNetworkOptions options;
    NeuralNetworkTrainOptions trainOptions;
    std::vector<int> sizes;
    int outputLayer;
    std::vector<std::vector<double>> biases;                   // double[outputLayer] [0] is empty?
    std::vector<std::vector<double>> biasChangesLow;           // double[outputLayer] [0] is empty?
    std::vector<std::vector<double>> biasChangesHigh;          // double[outputLayer] [0] is empty?
    std::vector<std::vector<std::vector<double>>> changesLow;  // double[outputLayer][sizes[firstIndex]] [0] is empty?
    std::vector<std::vector<std::vector<double>>> changesHigh; // double[outputLayer][sizes[firstIndex]] [0] is empty?
    std::vector<std::vector<std::vector<double>>> weights;     // double[outputLayer][sizes[firstIndex]] [0] is empty?
    std::vector<std::vector<double>> outputs;                  // double[outputLayer][sizes[firstIndex]]
    std::vector<std::vector<double>> deltas;                   // double[outputLayer][sizes[firstIndex]]
    std::vector<std::vector<std::vector<double>>> changes;     // double[outputLayer]
    std::vector<std::vector<double>> errors;                   // double[outputLayer][sizes[firstIndex]]
    unsigned int iterations;
    unsigned int errorCheckInterval;
    void initialize()
    {
        if (!isRunnable())
        {
            NAPI_ERROR(_, "Sizes must be set before initializing");
            return;
        }
        outputLayer = sizes.size() - 1;
        biases.clear();
        weights.clear();
        outputs.clear();
        deltas.clear();
        changes.clear();
        errors.clear();
        for (int layerIndex = 0; layerIndex <= outputLayer; layerIndex++)
        {
            int size = sizes[layerIndex];
            deltas.push_back(std::vector<double>(size));
            errors.push_back(std::vector<double>(size));
            outputs.push_back(std::vector<double>(sizes[layerIndex]));
            if (layerIndex > 0)
            {
                biases.push_back(randos(size));
                std::vector<std::vector<double>> weightL;
                std::vector<std::vector<double>> changeL;
                for (int nodeIndex = 0; nodeIndex < size; nodeIndex++)
                {
                    int prevSize = sizes[layerIndex - 1];
                    weightL.push_back(randos(prevSize));
                    changeL.push_back(std::vector<double>(prevSize));
                }
                weights.push_back(weightL);
                changes.push_back(changeL);
            }
            else
            {
                biases.push_back(std::vector<double>());
                weights.push_back(std::vector<std::vector<double>>());
                changes.push_back(std::vector<std::vector<double>>());
            }
        }
        if (trainOptions.praxis == PRAXIS_ADAM)
        {
            setupAdam();
        }
    };
    std::vector<double> run(std::vector<double> input)
    {
        if (!isRunnable())
        {
            NAPI_ERROR(_, "network not runnable");
            return std::vector<double>(0);
        }
        int inputSize = sizes[0];
        if (input.size() != inputSize)
        {
            NAPI_ERROR(_, "input length must match options.inputSize");
            return std::vector<double>(0);
        }
        return runInput(input);
    };
    std::vector<double> runInput(std::vector<double> input)
    {
        if (trainOptions.activation == ACTIVATION_SIGMOID)
            return _runInputSigmoid(input);
        if (trainOptions.activation == ACTIVATION_RELU)
            return _runInputRelu(input);
        if (trainOptions.activation == ACTIVATION_LEAKY_RELU)
            return _runInputLeakyRelu(input);
        if (trainOptions.activation == ACTIVATION_TANH)
            return _runInputTanh(input);
        NAPI_ERROR(_, "Invalid activation code.");
        return std::vector<double>(0);
    };
    std::vector<double> _runInputSigmoid(std::vector<double> input)
    {
        RUN_INPUT(1 / (1 + std::exp(-sum)));
    };
    std::vector<double> _runInputRelu(std::vector<double> input)
    {
        RUN_INPUT(sum < 0 ? 0 : sum);
    };
    std::vector<double> _runInputLeakyRelu(std::vector<double> input)
    {
        RUN_INPUT(std::max(static_cast<double>(sum), trainOptions.leakyReluAlpha * sum));
    };
    std::vector<double> _runInputTanh(std::vector<double> input)
    {
        RUN_INPUT(std::tanh(sum));
    };
    void calculateDeltas(std::vector<double> target)
    {
        if (trainOptions.activation == ACTIVATION_SIGMOID)
            return _calculateDeltasSigmoid(target);
        if (trainOptions.activation == ACTIVATION_RELU)
            return _calculateDeltasRelu(target);
        if (trainOptions.activation == ACTIVATION_LEAKY_RELU)
            return _calculateDeltasLeakyRelu(target);
        if (trainOptions.activation == ACTIVATION_TANH)
            return _calculateDeltasTanh(target);
        NAPI_ERROR(_, "Invalid activation code.");
    };
    void _calculateDeltasSigmoid(std::vector<double> target)
    {
        CALCULATE_DELTAS(error * output * (1 - output));
    };
    void _calculateDeltasRelu(std::vector<double> target)
    {
        CALCULATE_DELTAS(output > 0 ? error : 0);
    };
    void _calculateDeltasLeakyRelu(std::vector<double> target)
    {
        double alpha = trainOptions.leakyReluAlpha;
        CALCULATE_DELTAS(output > 0 ? error : alpha * error);
    };
    void _calculateDeltasTanh(std::vector<double> target)
    {
        CALCULATE_DELTAS((1 - output * output) * error);
    };
    bool isRunnable()
    {
        return sizes.size() > 0;
    };
    void verifyIsInitialized(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs)
    {
        if (sizes.size() > 0 && outputLayer > 0)
            return;
        sizes.clear();
        sizes.push_back(inputs[0].size());
        if (options.hiddenLayers.size() == 0)
        {
            int v = std::floor(inputs[0].size() / 2);
            sizes.push_back(v > 3 ? v : 3);
        }
        else
        {
            int hiddenLayersSize = options.hiddenLayers.size();
            for (int i = 0; i < hiddenLayersSize; i++)
            {
                sizes.push_back(options.hiddenLayers[i]);
            }
        }
        sizes.push_back(outputs[0].size());
        initialize();
    };

    double calculateTrainingError(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs)
    {
        double sum = 0;
        int size = inputs.size();
        for (int i = 0; i < size; ++i)
        {
            sum += trainPattern(inputs[i], outputs[i], true);
        }
        return sum / size;
    };

    void trainPatterns(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs)
    {
        int size = inputs.size();
        for (int i = 0; i < size; ++i)
        {
            trainPattern(inputs[i], outputs[i], false);
        }
    };

    bool trainingTick(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs, Status *status, double endTime)
    {
        if (status->iterations >= trainOptions.iterations ||
            status->error <= trainOptions.errorThresh ||
            (endTime != -1 && getDateNow() >= endTime))
        {
            return false;
        }
        status->iterations++;
        if (trainOptions.log && (status->iterations % trainOptions.logPeriod) == 0)
        {
            status->error = calculateTrainingError(inputs, outputs);
            std::cout << "Train Log | Iterations: " << status->iterations << ", error: " << status->error << "\n";
        }
        else if (status->iterations % errorCheckInterval == 0)
        {
            status->error = calculateTrainingError(inputs, outputs);
        }
        else
        {
            trainPatterns(inputs, outputs);
        }
        if (trainOptions.callback != nullptr && status->iterations % trainOptions.callbackPeriod == 0)
        {
            napi_value args[1];
            args[0] = create_object(_);
            set_object_key(_, args[0], "iterations", create_int32(_, status->iterations));
            set_object_key(_, args[0], "error", create_double(_, status->error));
            call_function(_, trainOptions.callback, 1, args);
        }
        return true;
    };

    Status *train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs)
    {
        Status *status = new Status();
        status->error = 1;
        status->iterations = 0;
        int endTime = trainOptions.timeout == -1 ? -1 : getDateNow() + trainOptions.timeout;
        if (inputs.size() != outputs.size())
        {
            NAPI_ERROR(_, "Expected the same amount of inputs and outputs.");
            return nullptr;
        }
        verifyIsInitialized(inputs, outputs);
        if (inputs[0].size() != sizes[0])
        {
            NAPI_ERROR(_, "Unexpected input size.");
            return nullptr;
        }
        if (outputs[0].size() != sizes.back())
        {
            NAPI_ERROR(_, "Unexpected output size.");
            return nullptr;
        }
        while (trainingTick(inputs, outputs, status, endTime))
        {
        }
        return status;
    };

    double trainPattern(std::vector<double> input, std::vector<double> output, bool logErrorRate)
    {
        runInput(input);
        calculateDeltas(output);
        adjustWeights();
        if (logErrorRate)
        {
            std::vector<double> errList = errors[outputLayer];
            double sum = 0;
            int errorsSize = errors.size();
            for (int i = 0; i < errorsSize; i++)
            {
                sum += std::pow(errList[i], 2);
            }
            return sum / errorsSize;
        }
        return NULL;
    };

    void adjustWeights()
    {
        for (int layer = 1; layer <= outputLayer; layer++)
        {
            std::vector<double> incoming = outputs[layer - 1];
            int incomingSize = incoming.size();
            int activeSize = sizes[layer];
            std::vector<double> activeDelta = deltas[layer];
            std::vector<double> *activeBiases = &biases[layer];
            std::vector<std::vector<double>> *activeChanges = &changes[layer];
            std::vector<std::vector<double>> *activeWeights = &weights[layer];
            for (int node = 0; node < activeSize; node++)
            {
                double delta = activeDelta[node];
                std::vector<double> *activeChangeSub = &(*activeChanges)[node];
                std::vector<double> *activeWeightsSub = &(*activeWeights)[node];
                for (int k = 0; k < incomingSize; k++)
                {
                    double change = (*activeChangeSub)[k];
                    change = trainOptions.learningRate * delta * incoming[k] + trainOptions.momentum * change;
                    (*activeChangeSub)[k] = change;
                    (*activeWeightsSub)[k] += change;
                }
                (*activeBiases)[node] += trainOptions.learningRate * delta;
            }
        }
    };

    void setupAdam()
    {
        biasChangesLow.clear();
        biasChangesHigh.clear();
        changesLow.clear();
        changesHigh.clear();
        iterations = 0;
        for (int layer = 0; layer <= outputLayer; layer++)
        {
            int size = sizes[layer];
            if (layer > 0)
            {
                biasChangesLow[layer] = std::vector<double>();
                biasChangesHigh[layer] = std::vector<double>();
                changesLow[layer] = std::vector<std::vector<double>>();
                changesHigh[layer] = std::vector<std::vector<double>>();
                for (int node = 0; node < size; node++)
                {
                    int prevSize = sizes[layer - 1];
                    changesLow[layer][node] = std::vector<double>();
                    changesHigh[layer][node] = std::vector<double>();
                }
            }
        }
    };

    void _adjustWeightsAdam()
    { // todo: adam.
        iterations++;
        for (int layer = 1; layer <= outputLayer; layer++)
        {
            std::vector<double> incoming = outputs[layer - 1];
            int incomingSize = incoming.size();
            int currentSize = sizes[layer];
            std::vector<double> currentDeltas = deltas[layer];
            std::vector<std::vector<double>> currentChangesLow = changesLow[layer];
            std::vector<std::vector<double>> currentChangesHigh = changesHigh[layer];
            std::vector<std::vector<double>> currentWeights = weights[layer];
            std::vector<double> currentBiases = biases[layer];
            std::vector<double> currentBiasChangesLow = biasChangesLow[layer];
            std::vector<double> currentBiasChangesHigh = biasChangesHigh[layer];
            // todo: fix for pointers
            for (int node = 0; node < currentSize; node++)
            {
                double delta = currentDeltas[node];
                for (int k = 0; k < incomingSize; k++)
                {
                    double gradient = delta * incoming[k];
                    double changeLow = currentChangesLow[node][k] * trainOptions.beta1 + (1 - trainOptions.beta1) * gradient;
                    double changeHigh = currentChangesHigh[node][k] * trainOptions.beta2 +
                                        (1 - trainOptions.beta2) * gradient * gradient;
                    double momentumCorrection = changeLow / (1 - std::pow(trainOptions.beta1, iterations));
                    double gradientCorrection = changeHigh / (1 - std::pow(trainOptions.beta2, iterations));
                    currentChangesLow[node][k] = changeLow;
                    currentChangesHigh[node][k] = changeHigh;
                    currentWeights[node][k] +=
                        (trainOptions.learningRate * momentumCorrection) /
                        (std::sqrt(gradientCorrection) + trainOptions.epsilon);
                }
                double biasGradient = currentDeltas[node];
                double biasChangeLow = currentBiasChangesLow[node] * trainOptions.beta1 + (1 - trainOptions.beta1) * biasGradient;
                double biasChangeHigh = currentBiasChangesHigh[node] * trainOptions.beta2 +
                                        (1 - trainOptions.beta2) * biasGradient * biasGradient;
                double biasMomentumCorrection = currentBiasChangesLow[node] / (1 - std::pow(trainOptions.beta1, iterations));
                double biasGradientCorrection = currentBiasChangesHigh[node] / (1 - std::pow(trainOptions.beta2, iterations));
                currentBiasChangesLow[node] = biasChangeLow;
                currentBiasChangesHigh[node] = biasChangeHigh;
                currentBiases[node] +=
                    (trainOptions.learningRate * biasMomentumCorrection) /
                    (std::sqrt(biasGradientCorrection) + trainOptions.epsilon);
            }
        }
    };
};

#define GET_NORMAL_OPTION(t, create) set_object_key(_, opts, #t, create(_, nn->options.t))

napi_value getOptions(napi_env _, napi_callback_info argv)
{
    napi_value _this;
    NeuralNetwork *nn;
    prepare_this(_, argv, _this);
    DO_UNWRAP(_this, nn, nullptr);
    napi_value opts = create_object(_);
    GET_NORMAL_OPTION(inputSize, create_int32);
    napi_value hiddenLayers = create_array(_);
    int hiddenLayersSize = nn->options.hiddenLayers.size();
    for (int i = 0; i < hiddenLayersSize; i++)
    {
        napi_set_named_property(_, hiddenLayers, std::to_string(i).c_str(), create_int32(_, nn->options.hiddenLayers[i]));
    }
    set_object_key(_, opts, "hiddenLayers", hiddenLayers);
    GET_NORMAL_OPTION(outputSize, create_int32);
    GET_NORMAL_OPTION(binaryThresh, create_double);
    return opts;
};

#define SET_NORMAL_OPTION(t, read, type, cond)                  \
    if (has_object_key(_, opts, #t))                            \
    {                                                           \
        napi_value k = get_object_key(_, opts, #t);             \
        EXPECT_TYPE(k, type, false);                            \
        int r = read(_, k);                                     \
        if (cond)                                               \
        {                                                       \
            NAPI_ERROR(_, "Failed to set the option: " #t "."); \
            return false;                                       \
        }                                                       \
        nn->options.t = r;                                      \
    }

bool updateOptionsAlias(napi_env _, NeuralNetwork *nn, napi_value opts)
{
    SET_NORMAL_OPTION(inputSize, read_int32, napi_number, r <= 0);
    SET_NORMAL_OPTION(outputSize, read_int32, napi_number, r <= 0);
    SET_NORMAL_OPTION(binaryThresh, read_double, napi_number, r <= 0 || r >= 1);
    if (has_object_key(_, opts, "hiddenLayers"))
    {
        nn->options.hiddenLayers.clear();
        napi_value arr = get_object_key(_, opts, "hiddenLayers");
        if (!is_array(_, arr))
        {
            NAPI_ERROR(_, "Expected options.hiddenLayers to be an array.");
            return nullptr;
        }
        int arrSize = get_array_length(_, arr);
        for (int i = 0; i < arrSize; i++)
        {
            nn->options.hiddenLayers.push_back(read_int32(_, get_object_key(_, arr, std::to_string(i).c_str())));
        }
    }
    return true;
}

napi_value updateOptions(napi_env _, napi_callback_info argv)
{
    napi_value _this;
    NeuralNetwork *nn;
    size_t argc = 1;
    napi_value args[1];
    prepare_args_this(_, argv, argc, args, _this);
    DO_UNWRAP(_this, nn, nullptr);
    napi_value opts = args[0];
    updateOptionsAlias(_, nn, opts);
    return nullptr;
};

#define GET_TRAINING_OPTION(t, read) set_object_key(_, opts, #t, read(_, nn->trainOptions.t))

napi_value getTrainingOptions(napi_env _, napi_callback_info argv)
{
    napi_value _this;
    NeuralNetwork *nn;
    prepare_this(_, argv, _this);
    DO_UNWRAP(_this, nn, nullptr);
    napi_value opts = create_object(_);
    GET_TRAINING_OPTION(activation, create_int32);
    GET_TRAINING_OPTION(iterations, create_int32);
    GET_TRAINING_OPTION(errorThresh, create_double);
    GET_TRAINING_OPTION(log, create_bool);
    GET_TRAINING_OPTION(logPeriod, create_int32);
    GET_TRAINING_OPTION(leakyReluAlpha, create_double);
    GET_TRAINING_OPTION(learningRate, create_double);
    GET_TRAINING_OPTION(momentum, create_double);
    set_object_key(_, opts, "callback", nn->trainOptions.callback == nullptr ? create_undefined(_) : nn->trainOptions.callback);
    GET_TRAINING_OPTION(callbackPeriod, create_int32);
    GET_TRAINING_OPTION(timeout, create_int32);
    GET_TRAINING_OPTION(beta1, create_double);
    GET_TRAINING_OPTION(beta2, create_double);
    GET_TRAINING_OPTION(epsilon, create_double);
    GET_TRAINING_OPTION(praxis, create_int32);
    return opts;
};

#define SET_TRAINING_OPTION(t, read, type, cond)                         \
    if (has_object_key(_, opts, #t))                                     \
    {                                                                    \
        napi_value k = get_object_key(_, opts, #t);                      \
        EXPECT_TYPE(k, type, false);                                     \
        int r = read(_, k);                                              \
        if (cond)                                                        \
        {                                                                \
            NAPI_ERROR(_, "Failed to set the training option: " #t "."); \
            return false;                                                \
        }                                                                \
        nn->trainOptions.t = r;                                          \
    }

bool updateTrainingOptionsAlias(napi_env _, NeuralNetwork *nn, napi_value opts)
{
    SET_TRAINING_OPTION(activation, read_int32, napi_number, r < ACTIVATION_MIN || r > ACTIVATION_MAX);
    SET_TRAINING_OPTION(iterations, read_int32, napi_number, r <= 0);
    SET_TRAINING_OPTION(errorThresh, read_double, napi_number, r <= 0 || r >= 1);
    SET_TRAINING_OPTION(log, read_bool, napi_boolean, false);
    SET_TRAINING_OPTION(logPeriod, read_int32, napi_number, r <= 0);
    SET_TRAINING_OPTION(leakyReluAlpha, read_double, napi_number, r <= 0 || r >= 1);
    SET_TRAINING_OPTION(learningRate, read_double, napi_number, r <= 0 || r >= 1);
    SET_TRAINING_OPTION(momentum, read_double, napi_number, r <= 0 || r >= 1);

    if (has_object_key(_, opts, "callback"))
    {
        napi_value r = get_object_key(_, opts, "callback");
        napi_valuetype type;
        napi_typeof(_, r, &type);
        if (type == napi_undefined)
        {
            nn->trainOptions.callback = nullptr;
        }
        else
        {
            if (type != napi_function)
            {
                NAPI_ERROR(_, "Failed to set the training option: callback. Expected a function.");
                return nullptr;
            }
            nn->trainOptions.callback = r;
            call_function(_, r, 0, nullptr);
        }
    }

    SET_TRAINING_OPTION(callbackPeriod, read_int32, napi_number, r <= 0);
    SET_TRAINING_OPTION(timeout, read_int32, napi_number, r <= 0);
    SET_TRAINING_OPTION(beta1, read_double, napi_number, r <= 0 || r >= 1);
    SET_TRAINING_OPTION(beta2, read_double, napi_number, r <= 0 || r >= 1);
    SET_TRAINING_OPTION(epsilon, read_double, napi_number, r <= 0 || r >= 1);
    SET_TRAINING_OPTION(praxis, read_int32, napi_number, r < PRAXIS_MIN || r > PRAXIS_MAX);
    return true;
}

napi_value updateTrainingOptions(napi_env _, napi_callback_info argv)
{
    napi_value _this;
    NeuralNetwork *nn;
    size_t argc = 1;
    napi_value args[1];
    prepare_args_this(_, argv, argc, args, _this);
    DO_UNWRAP(_this, nn, nullptr);
    napi_value opts = args[0];
    updateTrainingOptionsAlias(_, nn, opts);
    return nullptr;
};

napi_value train(napi_env _, napi_callback_info argv)
{
    napi_value _this;
    NeuralNetwork *nn;
    size_t argc = 1;
    napi_value args[1];
    prepare_args_this(_, argv, argc, args, _this);
    DO_UNWRAP(_this, nn, nullptr);
    if (argc != 1 && argc != 2)
    {
        NAPI_ERROR(_, "Only one or two argument expected for .train()");
        return nullptr;
    }
    if(argc == 2) {
        napi_value gotTrainingOptions = args[1];
        updateTrainingOptionsAlias(_, nn, gotTrainingOptions);
    }
    napi_value elements = args[0];
    if (!is_array(_, elements))
    {
        NAPI_ERROR(_, "Expected the first argument to be an array.");
        return nullptr;
    }
    int elementsSize = get_array_length(_, elements);
    if (elementsSize == 0)
    {
        NAPI_ERROR(_, "Expected an input for the .train() call.");
        return nullptr;
    }
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;
    int inputSize = 0;
    int outputSize = 0;
    for (int i = 0; i < elementsSize; i++)
    {
        napi_value el = get_object_key(_, elements, std::to_string(i).c_str());
        napi_value in = get_object_key(_, el, "input");
        napi_value out = get_object_key(_, el, "output");
        if (!is_array(_, in))
        {
            NAPI_ERROR(_, "Expected options.hiddenLayers to be an array.");
            return nullptr;
        }
        if (!is_array(_, out))
        {
            NAPI_ERROR(_, "Expected options.hiddenLayers to be an array.");
            return nullptr;
        }
        int inSize = get_array_length(_, in);
        int outSize = get_array_length(_, out);
        if (i == 0)
        {
            if (inSize == 0)
            {
                NAPI_ERROR(_, "Expected the input for the .train() call to be non-empty.");
                return nullptr;
            }
            if (outSize == 0)
            {
                NAPI_ERROR(_, "Expected the output for the .train() call to be non-empty.");
                return nullptr;
            }
            inputSize = inSize;
            outputSize = outSize;
        }
        else
        {
            if (inputSize != inSize)
            {
                NAPI_ERROR(_, "Expected all inputs to have the same size.");
                return nullptr;
            }
            if (outputSize != outSize)
            {
                NAPI_ERROR(_, "Expected all inputs to have the same size.");
                return nullptr;
            }
        }
        std::vector<double> newInputs;
        std::vector<double> newOutputs;
        for (int j = 0; j < inSize; j++)
        {
            double currentIn = read_double(_, get_object_key(_, in, std::to_string(j).c_str()));
            newInputs.push_back(currentIn);
        }
        for (int j = 0; j < outSize; j++)
        {
            double currentOut = read_double(_, get_object_key(_, out, std::to_string(j).c_str()));
            newOutputs.push_back(currentOut);
        }
        inputs.push_back(newInputs);
        outputs.push_back(newOutputs);
    }
    Status *status = nn->train(inputs, outputs);
    if (status == nullptr)
    {
        return nullptr;
    }
    napi_value obj = create_object(_);
    set_object_key(_, obj, "error", create_double(_, status->error));
    set_object_key(_, obj, "iterations", create_int32(_, status->iterations));
    return obj;
}

napi_value run(napi_env _, napi_callback_info argv)
{
    napi_value _this;
    NeuralNetwork *nn;
    size_t argc = 1;
    napi_value args[1];
    prepare_args_this(_, argv, argc, args, _this);
    DO_UNWRAP(_this, nn, nullptr);
    if (nn->sizes.size() == 0 || nn->outputLayer <= 0)
    {
        NAPI_ERROR(_, "Neural network hasn't been initialised/trained yet.");
        return nullptr;
    }
    if (argc != 1)
    {
        NAPI_ERROR(_, "Only 1 argument expected for .run()");
        return nullptr;
    }
    napi_value inputs = args[0];
    if (!is_array(_, inputs))
    {
        NAPI_ERROR(_, "Expected options.hiddenLayers to be an array.");
        return nullptr;
    }
    int inputsSize = get_array_length(_, inputs);
    if (inputsSize != nn->sizes[0])
    {
        NAPI_ERROR(_, "Expected the size of the input to be the same as the trained input's size.");
        return nullptr;
    }
    std::vector<double> list;
    for (int i = 0; i < inputsSize; i++)
    {
        napi_value el = get_object_key(_, inputs, std::to_string(i).c_str());
        list.push_back(read_int32(_, el));
    }
    std::vector<double> runResult = nn->run(list);
    int runResultSize = runResult.size();
    napi_value resultArray = create_array_with_length(_, runResultSize);
    for (int i = 0; i < runResultSize; i++)
    {
        double out = runResult[i];
        set_object_key(_, resultArray, std::to_string(i).c_str(), create_double(_, out));
    }
    return resultArray;
}

napi_value neuralNetworkConstructor(napi_env _, napi_callback_info argv)
{
    napi_value _this = create_object(_);
    NeuralNetwork *nn = new NeuralNetwork(_, _this);
    size_t argc = 1;
    napi_value args[1];
    prepare_args(_, argv, argc, args);
    if (argc > 0)
    {
        if (!updateOptionsAlias(_, nn, args[0]))
            return nullptr;
        if (argc > 1 && !updateTrainingOptionsAlias(_, nn, args[1]))
            return nullptr;
    }
    DO_WRAP(_this, nn, NeuralNetwork *, nullptr);
    add_function_to_object(_, _this, "getOptions", getOptions);
    add_function_to_object(_, _this, "updateOptions", updateOptions);
    add_function_to_object(_, _this, "getTrainingOptions", getTrainingOptions);
    add_function_to_object(_, _this, "updateTrainingOptions", updateTrainingOptions);
    add_function_to_object(_, _this, "train", train);
    add_function_to_object(_, _this, "run", run);
    return _this;
}