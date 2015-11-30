#ifndef MNIST_DATASET_H
#define MNIST_DATASET_H

#include <vector>


class Dataset {
private:
    int _n, _features;
    unsigned char **_data;
    short *_label;
public:
    Dataset(int n, int features);
    ~Dataset();

    short *get_labels() {
        return _label;
    }

    unsigned char *get_data(int id) {
        return _data[id];
    }

    const unsigned char *get_data(int id) const {
        return _data[id];
    }

    const short get_label(int id) const {
        return _label[id];
    }

    void set_label(int id, short v) {
        _label[id] = v;
    }

    const int &size() const {
        return _n;
    }

    const int &features() const {
        return _features;
    }
};


#endif //MNIST_DATASET_H
