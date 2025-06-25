#ifndef __SVCONV_H__
#define __SVCONV_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "model.h"
#define MAX_ITER 1000
#define EPSILON 1e-5
#define HUBER_DELTA 1.0f

#define max(a, b) ((a) > (b) ? (a) : (b))

int8_t sv_conv_neural_network(Model_Conv* conv_model, const float mfe_input[], float* prob);
void batch_normalization(Model_Conv* conv, const float* input);
void conv2d(Layer_Conv* layer);
void max_pool2d(Layer_Conv* layer);
float bestmatching(const float* input_vector, const float d_vectors[][MAX_DVECTOR_SIZE], int num_vectors, int d_vector_size);
float cosine_similarity(const float* vec1, const float* vec2, int size);
float relu_sv(float x);
void mean_d_vector(const float d_vectors[][MAX_DVECTOR_SIZE], float* mean_output, int num_vectors, int d_vector_size);
float mean_cos(const float* input_vector, const float d_vectors[][MAX_DVECTOR_SIZE], int num_vectors, int dvector_size);
int floor_div(int a, int b);
/*NEW TRY AGGREGATE FUNCTIONS*/
void huber_computation(const float d_vectors[][MAX_DVECTOR_SIZE], float* median_output, int num_vectors, int d_vector_size);
float huber_vector(const float* input_vector, const float d_vectors[][MAX_DVECTOR_SIZE], int num_vectors, int dvector_size);
void geometrical_median_computation(const float d_vectors[][MAX_DVECTOR_SIZE], float* max_output, int num_vectors, int d_vector_size);
float geometrical_median_vector(const float* input_vector, const float d_vectors[][MAX_DVECTOR_SIZE], int num_vectors, int dvector_size);
float distance(const float* a, const float* b, int dim);
#endif
