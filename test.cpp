#include "load.h"
#include "NeuralNetwork.h"

#include <cassert>


void test_or() {
    Dataset *dataset = new Dataset(4, 2);
    dataset->get_data(0)[0] = 0;
    dataset->get_data(0)[1] = 0;
    dataset->get_data(1)[0] = 0;
    dataset->get_data(1)[1] = 1;
    dataset->get_data(2)[0] = 1;
    dataset->get_data(2)[1] = 0;
    dataset->get_data(3)[0] = 1;
    dataset->get_data(3)[1] = 1;
    dataset->get_labels()[0] = 0;
    dataset->get_labels()[1] = 1;
    dataset->get_labels()[2] = 1;
    dataset->get_labels()[3] = 1;

    NeuralNetwork *neural_network = new NeuralNetwork({1}, 2);
    neural_network->learn(dataset, .2, .1);

    assert(neural_network->test(dataset) == 1);

    delete dataset;
    delete neural_network;
}


void test_and() {
    Dataset *dataset = new Dataset(4, 2);
    dataset->get_data(0)[0] = 0;
    dataset->get_data(0)[1] = 0;
    dataset->get_data(1)[0] = 0;
    dataset->get_data(1)[1] = 1;
    dataset->get_data(2)[0] = 1;
    dataset->get_data(2)[1] = 0;
    dataset->get_data(3)[0] = 1;
    dataset->get_data(3)[1] = 1;
    dataset->get_labels()[0] = 0;
    dataset->get_labels()[1] = 0;
    dataset->get_labels()[2] = 0;
    dataset->get_labels()[3] = 1;

    NeuralNetwork *neural_network = new NeuralNetwork({1}, 2);
    neural_network->learn(dataset, .2, .1);

    assert(neural_network->test(dataset) == 1);

    delete dataset;
    delete neural_network;
}


void test_xor() {
    Dataset *dataset = new Dataset(4, 2);
    dataset->get_data(0)[0] = 0;
    dataset->get_data(0)[1] = 0;
    dataset->get_data(1)[0] = 0;
    dataset->get_data(1)[1] = 1;
    dataset->get_data(2)[0] = 1;
    dataset->get_data(2)[1] = 0;
    dataset->get_data(3)[0] = 1;
    dataset->get_data(3)[1] = 1;
    dataset->get_labels()[0] = 0;
    dataset->get_labels()[1] = 1;
    dataset->get_labels()[2] = 1;
    dataset->get_labels()[3] = 0;

    NeuralNetwork *neural_network = new NeuralNetwork({2, 1}, 2);
    neural_network->learn(dataset, .2, .1);

    assert(neural_network->test(dataset) == 1);

    delete dataset;
    delete neural_network;
}


int main(int argc, char *argv[]) {
    test_or();
    test_and();
    test_xor();
    return 0;
}
