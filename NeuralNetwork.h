#ifndef MNIST_NEURALNETWORK_H
#define MNIST_NEURALNETWORK_H

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

#include "Dataset.h"


class NeuralNetwork {
private:
    struct Neuron {
        double u, y, e;
    };

    std::vector<int> _levels;
    int _pattern_size, _neurons, _weights;
    int *_weight_start;
    double *_weight;
    Neuron *_neuron;

    void _init();
    static double _f(const Neuron &n);
    static double _fp(const Neuron &n);
    static double _get_d(short d, int bit) {
        return (d >> bit) & 1;
    }

    void _calc_outputs(const unsigned char *pattern);
    bool _calc_errors(double max_error, const unsigned char *pattern,
                      short d, double rate);

public:
    NeuralNetwork(const std::vector<int> &levels, int pattern_size);
    ~NeuralNetwork();
    void learn(const Dataset *dataset, double rate, double max_error,
               int max_repetitions=100000);
    double test(const Dataset *dataset);

    void show_weights() const;
};


#endif //MNIST_NEURALNETWORK_H
