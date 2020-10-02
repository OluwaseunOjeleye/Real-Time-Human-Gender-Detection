#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "box.h"
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif

void embed_image(image source, image dest, int dx, int dy);

image make_empty_image(int w, int h, int c);

#ifdef __cplusplus
}
#endif

#endif

