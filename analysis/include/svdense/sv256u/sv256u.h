#ifndef __SV256U_H__
#define __SV256U_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dense_network.h"

void processing_sv256u_model(const float* input);
void init_sv256u_model();
void allocate_layers_sv256u(Layer_Dense** layers, int num_layers);

#endif
