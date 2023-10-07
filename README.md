# NeuralC
A neural network library written in C++ which supports JavaScript. Inspired by brain.js

# Performance TODOs

- Currently since I first wanted to port the JavaScript version to C++ I'm using C-vectors which are slower than C-style arrays that point directly to memory. Vectors should be changed into C-style arrays.
- Currently I'm using generalised and easy to use functions that I've made to fetch/create data from/to JavaScript. I should be using raw node_api functions.