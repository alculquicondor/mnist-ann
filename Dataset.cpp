//
// Created by alculquicondor on 11/10/15.
//

#include "Dataset.h"

Dataset::Dataset(int n, int features) :
        _n(n), _features(features) {
    _data = new unsigned char *[n];
    _data[0] = new unsigned char[n * features];
    for (int i = 1; i < n; ++i)
        _data[i] = _data[i - 1] + features;
    _label = new short[n];
}

Dataset::~Dataset() {
    delete[] _data[0];
    delete[] _data;
    delete[] _label;
}
