#ifndef __SV128_H__
#define __SV128_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "model.h"

void allocate_layers_sv_128(Layer_Conv** layers, int num_layers);
void init_sv_128_model();
void processing_sv_128_model(const float* input);

#endif
