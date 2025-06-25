#ifndef __SV_DENSE_NETWORK_H__
#define __SV_DENSE_NETWORK_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dense_network.h"
#include "sv_conv.h"

int sv_dense_neural_network(Model_Dense* dense_model, const float* input, float* prob);

#endif
