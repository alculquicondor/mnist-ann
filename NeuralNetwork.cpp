#include <chrono>
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(const std::vector<int> &levels, int pattern_size) :
        _levels(levels), _pattern_size(pattern_size) {
    _weight_start = new int[levels.size()];
    _weights = (pattern_size + 1) * levels.front();
    for (int i = 0; i < levels.size() - 1; ++i) {
        _weight_start[i] = _weights;
        _weights += (levels[i] + 1) * levels[i + 1];
    }
    for (int i = 1; i < _levels.size(); ++i)
        _levels[i] += _levels[i - 1];
    _neurons = _levels.back();
    _weight = new double[_weights];
    _neuron = new Neuron[_neurons];
}


NeuralNetwork::~NeuralNetwork() {
    delete[] _weight_start;
    delete[] _weight;
    delete[] _neuron;
}


void NeuralNetwork::_init() {
    std::default_random_engine generator((unsigned long)std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> uniform(-0.1, 0.1);
    for (int i = 0; i < _weights; ++i)
        _weight[i] = uniform(generator);
}


double NeuralNetwork::_f(const Neuron &n) {
    return 1. / (1 + exp(-n.u));
}


double NeuralNetwork::_fp(const Neuron &n) {
    return n.y * (1 - n.y);
}


void NeuralNetwork::_calc_outputs(const unsigned char *pattern) {
    for (int i = 0; i < _neurons; ++i)
        _neuron[i].u = 0;
    int w_id = 0;

    // first level
    for (int i = 0; i < _pattern_size; ++i)
        for (int n = 0; n < _levels.front(); ++n)
            _neuron[n].u += _weight[w_id++] * pattern[i];
    for (int n = 0; n < _levels.front(); ++n) {
        _neuron[n].u += _weight[w_id++];  // bias
        _neuron[n].y = _f(_neuron[n]);
    }

    // each level
    for (int l = 0; l < _levels.size() - 1; ++l) {
        for (int n = l > 0 ? _levels[l - 1] : 0; n < _levels[l]; ++n)
            for (int m = _levels[l]; m < _levels[l + 1]; ++m)
                _neuron[m].u += _weight[w_id++] * _neuron[n].y;

        for (int m = _levels[l]; m < _levels[l + 1]; ++m) {
            _neuron[m].u += _weight[w_id++];  // bias
            _neuron[m].y = _f(_neuron[m]);
        }
    }
}


bool NeuralNetwork::_calc_errors(double max_error, const unsigned char *pattern,
                                 short d, double rate) {
    bool desired = true;
    for (int i = 0; i < _neurons; ++i)
        _neuron[i].e = 0;
    int w_id;

    // last level
    for (int n = _levels.size() > 1 ? _levels[_levels.size() - 2] : 0; n < _levels.back(); ++n) {
        int bit = n - _levels[_levels.size() - 2];
        _neuron[n].e = (_get_d(d, bit) - _neuron[n].y) * _fp(_neuron[n]);
        desired &= fabs(_neuron[n].e) < max_error;
    }

    // internal levels
    for (int l = (int)_levels.size() - 2; l >= 0; --l) {
        w_id = _weight_start[l];

        for (int n = l > 0 ? _levels[l - 1] : 0; n < _levels[l]; ++n) {
            for (int m = _levels[l]; m < _levels[l + 1]; ++m) {
                _neuron[n].e += _neuron[m].e * _weight[w_id];
                _weight[w_id++] += rate * _neuron[n].y * _neuron[m].e;
            }
            _neuron[n].e *= _fp(_neuron[n]);
            desired &= fabs(_neuron[n].e) < max_error;
        }

        // weights for bias
        for (int m = _levels[l]; m < _levels[l + 1]; ++m)
            _weight[w_id++] += rate * _neuron[m].e;
    }

    // weights for input
    w_id = 0;
    for (int i = 0; i < _pattern_size; ++i)
        for (int m = 0; m < _levels.front(); ++m)
            _weight[w_id++] += rate * pattern[i] * _neuron[m].e;
    for (int m = 0; m < _levels.front(); ++m)
        _weight[w_id++] += rate * _neuron[m].e;

    return desired;
}


void NeuralNetwork::learn(const Dataset *dataset, double rate,
                          double max_error, int max_repetitions) {
    _init();
    int repetitions = 0;
    bool desired;
    do {
        desired = true;
        for (int i = 0; i < dataset->size(); ++i) {
            _calc_outputs(dataset->get_data(i));
            desired &= _calc_errors(max_error,
                                        dataset->get_data(i),
                                        dataset->get_label(i),
                                        rate);
        }
        ++repetitions;
        if (repetitions % 1000 == 0)
            std::cerr << repetitions << std::endl;
    } while (not desired and repetitions < max_repetitions);
    std::cerr << repetitions << " repetitions" << std::endl;
}


double NeuralNetwork::test(const Dataset *dataset) {
    int correct_cnt = 0;
    for (int i = 0; i < dataset->size(); ++i) {
        _calc_outputs(dataset->get_data(i));
        int label = 0;
        for (int n = _levels[_levels.size() - 2]; n < _levels.back(); ++n) {
            int bit = n - _levels[_levels.size() - 2];
            label |= (_neuron[n].y > 0.5 ? 1 : 0) << bit;
        }
        //std::cerr << "label " << label << ' ' << (int)dataset->get_label(i) << std::endl;
        correct_cnt += label == dataset->get_label(i);
    }
    return (double)correct_cnt / dataset->size();
}


void NeuralNetwork::show_weights() const {
    int w_id = 0;

    // first level
    for (int i = 0; i < _pattern_size; ++i)
        for (int n = 0; n < _levels.front(); ++n)
            std::cout << 'i' << i << ' ' << n << ' ' << _weight[w_id++] << std::endl;
    for (int n = 0; n < _levels.front(); ++n)
        std::cout << "b0 " << n << ' ' << _weight[w_id++] << std::endl;

    // each level
    for (int l = 0; l < _levels.size() - 1; ++l) {
        for (int n = l > 0 ? _levels[l - 1] : 0; n < _levels[l]; ++n)
            for (int m = _levels[l]; m < _levels[l + 1]; ++m)
                std::cout << n << ' ' << m << ' ' << _weight[w_id++] << std::endl;

        for (int m = _levels[l]; m < _levels[l + 1]; ++m)
            std::cout << 'b' << l << ' ' << m << ' ' << _weight[w_id++] << std::endl;
    }
}
