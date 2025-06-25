#ifndef __SVQ8_QAT_SV_DENSE_INT8_ALT_H__
#define __SVQ8_QAT_SV_DENSE_INT8_ALT_H__

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_LAYERS_SV_INT8_ALT 4
#define INPUT_SIZE_SV_INT8_ALT 1600 //Input Features from MFE
#define HIDDEN_LAYER_1_SIZE_SV_INT8_ALT 192 //Hidden Layer 1
#define HIDDEN_LAYER_2_SIZE_SV_INT8_ALT 192 //Hidden Layer 2
#define HIDDEN_LAYER_3_SIZE_SV_INT8_ALT 192 //Hidden Layer 3
#define OUTPUT_SIZE_SV_INT8_ALT 256 //Output Layer
#define SIMILARITY_THRESHOLD_INT8_ALT 0.7

int sv_dense_int8_alt_neural_network(const float input[INPUT_SIZE_SV_INT8_ALT], int sv_elaborate);

#endif
