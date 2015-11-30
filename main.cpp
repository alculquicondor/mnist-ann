#include <iostream>

#include "load.h"
#include "NeuralNetwork.h"


using namespace std;


int main(int argc, char *argv[]) {
    string dir(argv[1]);
    Dataset *train_dataset = read_images(dir + "/train-images-idx3-ubyte", 1000);
    read_labels(dir + "/train-labels-idx1-ubyte", train_dataset);
    for (int i = 0; i < train_dataset->size(); ++i)
        train_dataset->set_label(i, (short)(1 << train_dataset->get_label(i)));

    // 20 25 10  .005 -> 39%
    NeuralNetwork *neural_network = new NeuralNetwork({15, 20, 10}, train_dataset->features());
    neural_network->learn(train_dataset, 0.001, .1, 100000);

    cout << neural_network->test(train_dataset) * 100 << endl;

    delete train_dataset;

    Dataset *test_dataset = read_images(dir + "/t10k-images-idx3-ubyte");
    read_labels(dir + "/t10k-labels-idx1-ubyte", test_dataset);
    for (int i = 0; i < test_dataset->size(); ++i)
        test_dataset->set_label(i, (short)(1 << test_dataset->get_label(i)));

    cout << neural_network->test(test_dataset) * 100 << endl;

    delete test_dataset;
    delete neural_network;
    return 0;
}