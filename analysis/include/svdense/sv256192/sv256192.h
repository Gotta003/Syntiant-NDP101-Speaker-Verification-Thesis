#ifndef __SV256192_H__
#define __SV256192_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dense_network.h"

void processing_sv256192_model(const float* input);
void init_sv256192_model();
void allocate_layers_sv256192(Layer_Dense** layers, int num_layers);

#endif
