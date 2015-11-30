#include "load.h"


int get_int(char buffer[], int offset) {
    int result = (unsigned char) buffer[offset + 3] |
            ((unsigned char) buffer[offset + 2] << 8) |
            ((unsigned char) buffer[offset + 1] << 16) |
            ((unsigned char) buffer[offset] << 24);
    return result;
}


Dataset *read_header(std::istream &is, int max_elements) {
    char buffer[12];
    int n, height, width;

    is.seekg(4);
    is.read(buffer, 12);
    n = std::min(get_int(buffer, 0), max_elements);
    height = get_int(buffer, 4);
    width = get_int(buffer, 8);

    return new Dataset(n, width * height);
}


Dataset *read_images(std::string filename, int max_elements) {
    std::ifstream data_stream(filename, std::ios::binary);
    Dataset *dataset = read_header(data_stream, max_elements);

    data_stream.read((char *)dataset->get_data(0),
                     dataset->size() * dataset->features());

    data_stream.close();
    return dataset;
}


Dataset *read_images(std::string filename, Dataset *dataset) {
    std::ifstream data_stream(filename, std::ios::binary);
    data_stream.seekg(16);
    data_stream.read((char *)dataset->get_data(0),
                     dataset->size() * dataset->features());
    data_stream.close();
    return dataset;
}


Dataset *read_labels(std::string filename, Dataset *dataset) {
    std::ifstream data_stream(filename, std::ios::binary);
    data_stream.seekg(8);

    char *buffer = new char[dataset->size()];
    data_stream.read(buffer, dataset->size());
    data_stream.close();

    for (int i = 0; i < dataset->size(); ++i)
        dataset->set_label(i, buffer[i]);

    delete[] buffer;
    return dataset;
}
