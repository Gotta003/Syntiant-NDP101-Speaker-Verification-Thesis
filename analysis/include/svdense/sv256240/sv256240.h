#ifndef __SV256240_H__
#define __SV256240_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dense_network.h"

void processing_sv256240_model(const float* input);
void init_sv256240_model();
void allocate_layers_sv256240(Layer_Dense** layers, int num_layers);

#endif
