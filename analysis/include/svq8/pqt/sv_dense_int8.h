#ifndef __SVQ8_PQT_SV_DENSE_INT8_H__
#define __SVQ8_PQT_SV_DENSE_INT8_H__

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_LAYERS_SV_INT8 4
#define INPUT_SIZE_SV_INT8 1600 //Input Features from MFE
#define HIDDEN_LAYER_1_SIZE_SV_INT8 240 //Hidden Layer 1
#define HIDDEN_LAYER_2_SIZE_SV_INT8 240 //Hidden Layer 2
#define HIDDEN_LAYER_3_SIZE_SV_INT8 240 //Hidden Layer 3
#define OUTPUT_SIZE_SV_INT8 256 //Output Layer
#define SIMILARITY_THRESHOLD_INT8 0.7

typedef enum Layer {INPUT, L1, L2, L3, OUTPUT} Layer;

void QuantizeMultiplier(double real_multiplier, int32_t* quantized_multiplier, int* shift);
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift);
float switching_scales(Layer layer);
void fully_connected_layer_int8(const int8_t input[], int8_t output[], const int8_t weights[], const float weight_scales[], const int32_t biases[], int input_size, int output_size, Layer layer_input, Layer layer_output);
int sv_dense_int8_neural_network(const float input[INPUT_SIZE_SV_INT8], int sv_elaborate);
void input_quantization_int8(const float* input, int8_t* q_input, int size, float scale, int32_t zero_point);
void input_dequantization_int32(int32_t* q_output, float* output, int size, float scale, int32_t zero_point);
void input_dequantization_int8(int8_t* q_output, float* output, int size, float scale, int32_t zero_point);

#endif
