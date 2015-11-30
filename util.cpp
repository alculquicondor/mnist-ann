#include "util.h"

Image get_image(unsigned int height, unsigned int width, unsigned char *data) {
    return Image(data, width, height);
}
