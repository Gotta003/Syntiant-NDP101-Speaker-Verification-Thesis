#ifndef __SVQ4_SV_DENSE_INT4_H__
#define __SVQ4_SV_DENSE_INT4_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_LAYERS_SV_INT4 4
#define INPUT_SIZE_SV_INT4 1600 //Input Features from MFE
#define HIDDEN_LAYER_1_SIZE_SV_INT4 192 //Hidden Layer 1
#define HIDDEN_LAYER_2_SIZE_SV_INT4 192 //Hidden Layer 2
#define HIDDEN_LAYER_3_SIZE_SV_INT4 192 //Hidden Layer 3
#define OUTPUT_SIZE_SV_INT4 256 //Output Layer
#define DVECTORS 256

int sv_dense_int4_neural_network(const float input[INPUT_SIZE_SV_INT4], int sv_elaborate);
void features_extraction(const float input[INPUT_SIZE_SV_INT4], int8_t output[OUTPUT_SIZE_SV_INT4]);
void features_extraction(const float input[INPUT_SIZE_SV_INT4], int8_t output[OUTPUT_SIZE_SV_INT4]);
void quantization(const float input[INPUT_SIZE_SV_INT4], int8_t quantized_input[INPUT_SIZE_SV_INT4]);
float compute_similarity_int(const int8_t input_vector[DVECTORS], const int8_t d_vectors[][DVECTORS], int num_vectors);
void bestmatching_int(const int8_t input_vectors[][DVECTORS], const int8_t d_vectors[][DVECTORS], float y_prediction_prob[], int num_inputs, int vector_size);
void mean_d_vector_int(const int8_t input_vectors[][DVECTORS], int8_t mean_output[DVECTORS], int num_vectors);
void mean_cos_int(const int8_t input_vectors[][DVECTORS], const int8_t d_vectors[][DVECTORS], float y_prediction_prob[], int num_inputs, int vector_size);
float cosine_similarity_int(const int8_t vec1[DVECTORS], const int8_t vec2[DVECTORS]);
void fully_connected_layer_int(const int8_t input[], int8_t output[],const int8_t weights[], const int32_t biases[], int input_size, int output_size);
#endif
