#ifndef __KWS_H__
#define __KWS_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dense_network.h"

int8_t processing_kws_model(const float* input);
void init_kws_model();
void allocate_layers_kws(Layer_Dense** layers, int num_layers);

#endif
