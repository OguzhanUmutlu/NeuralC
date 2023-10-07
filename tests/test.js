const { NeuralNetwork } = require("../index");

Error.stackTraceLimit = 1;

function testData(data) {
    const net = new NeuralNetwork();
    console.time("train");
    net.train(data);
    console.timeEnd("train");

    for (const dat of data) {
        const res = [...net.run(dat.input)];
        console.log(dat.input, dat.output, res, res.map(i => i.toFixed(3) * 1));
        for (let i = 0; i < res.length; i++) {
            const v = res[i] > 0.5 ? 1 : 0;
            if (v !== dat.output[i]) throw new Error("Wrong.");
        }
    }
}


testData([
    { input: [0, 0], output: [0, 1] },
    { input: [0, 1], output: [1, 1] },
    { input: [1, 0], output: [1, 1] },
    { input: [1, 1], output: [0, 0] },
]);