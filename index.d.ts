type int32 = number;
// type int64 = number; // not used atm
type double = number;
type bool = boolean;

type NeuralNetworkOptions = {
    inputSize: int32,
    hiddenLayers: int32[],
    outputSize: int32,
    binaryThresh: double
};

type NeuralNetworkTrainingOptions = {
    activation: int32,
    iterations: int32,
    errorThresh: double,
    log: bool,
    logPeriod: int32,
    leakyReluAlpha: double,
    learningRate: double,
    momentum: double,
    callback: Function,
    callbackPeriod: int32,
    timeout: int32,
    beta1: double,
    beta2: double,
    epsilon: double,
    praxis: int32
};

declare class NeuralNetwork {
    constructor(options?: Partial<NeuralNetworkOptions>, trainOptions?: Partial<NeuralNetworkTrainingOptions>);

    getOptions(): NeuralNetworkOptions;

    updateOptions(options: Partial<NeuralNetworkOptions>): void;

    getTrainingOptions(): NeuralNetworkTrainingOptions;

    updateTrainingOptions(options: Partial<NeuralNetworkTrainingOptions>): void;

    train(data: { input: int32[], output: int32[] }[], trainOptions?: Partial<NeuralNetworkTrainingOptions>): {
        error: double,
        iterations: int32
    };

    run(input: int32[]): int32[];
}

declare const __: { NeuralNetwork: typeof NeuralNetwork };
export = __;