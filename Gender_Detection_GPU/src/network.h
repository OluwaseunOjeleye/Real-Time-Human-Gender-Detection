#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"
#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GPU
void pull_network_output(network *net);
#endif

network *make_network(int n);
void calc_network_cost(network *net);

#ifdef __cplusplus
}
#endif

#endif

