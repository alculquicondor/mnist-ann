#ifndef MNIST_UTIL_H
#define MNIST_UTIL_H

#include "CImg.h"

using namespace cimg_library;


typedef CImg<unsigned char> Image;

Image get_image(unsigned int height, unsigned int width, unsigned char *data);

#endif //MNIST_UTIL_H
