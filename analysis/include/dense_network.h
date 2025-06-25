#ifndef __DENSE_NETWORK_H__
#define __DENSE_NETWORK_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "class_names.h"
#include "model.h"

float relu(float x);
void fully_connected_layer(Layer_Dense* layer);
void softmax(float input[], int size);
int dense_neural_network(Model_Dense* dense, const float* input, float* prob);
void elaborateResult(float output[], float* prob);

#endif
