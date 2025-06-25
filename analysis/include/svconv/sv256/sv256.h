#ifndef __SV256_H__
#define __SV256_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "model.h"

void allocate_layers_sv_256(Layer_Conv** layers, int num_layers);
void init_sv_256_model();
void processing_sv_256_model(const float* input);

#endif
