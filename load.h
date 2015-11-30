#ifndef MNIST_LOAD_H
#define MNIST_LOAD_H
#include <fstream>
#include <limits>

#include "Dataset.h"


int get_int(char buffer[], int offset=0);

Dataset *read_header(std::istream &is, int max_elements=std::numeric_limits<int>::max());

Dataset *read_images(std::string filename, int max_elements=std::numeric_limits<int>::max());

Dataset *read_images(std::string filename, Dataset *dataset);

Dataset *read_labels(std::string filename, Dataset *dataset);

#endif //MNIST_LOAD_H
