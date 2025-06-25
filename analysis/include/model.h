#ifndef __MODEL_H__
#define __MODEL_H__

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdbool.h"
#include "stdint.h"
#include "time.h"
#define MAX_CONV_LEVELS 8
#define MAX_DENSE_LEVELS 6
#define MAX_DVECTOR_SIZE 256
#define NUM_THRESHOLDS 10 
#define NUM_REFERENCES 4

typedef enum Model_Types{
    KWS,
    SV_CONV,
    SV_DENSE,
    SVQ8,
    SVQ4
} Model_Types;

typedef enum Activation{
    NONE,
    RELU,
    SOFTMAX
} Activation;

typedef enum Type_Layer_Conv {
    CONV2D,
    MAXPOOL2D
} Type_Layer_Conv;

typedef enum Padding_Conv {
    SAME,
    VALID
} Padding_Conv;

typedef struct Shape_Conv {
    int16_t batch;
    int16_t height;
    int16_t width;
    int16_t channels;
} Shape_Conv;

typedef struct Shape_Dense {
    int16_t batch;
    int16_t size;
} Shape_Dense;

typedef struct Layer_Conv {
    float* in_data;
    float* out_data;
    Type_Layer_Conv type_layer_conv;
    Shape_Conv input_shape;
    Shape_Conv output_shape;
    const float* weights;
    const float* biases;
    Activation activation;
    Padding_Conv padding_conv;
    int kernel_size;
    int stride;
} Layer_Conv;

typedef struct Layer_Dense {
    float* in_data;
    float* out_data;
    Shape_Dense input_shape;
    Shape_Dense output_shape;
    const float* weights;
    const float* biases;
    Activation activation;
} Layer_Dense;

typedef struct Model_Conv {
    Model_Types model_type;
    int num_conv_layers;
    Layer_Conv* layers;
    bool batch_norm;
    float beta;
    float gamma;
    float threshold;
    const float (*dvectors)[MAX_DVECTOR_SIZE];
    int num_references;
    int d_vector_size;
} Model_Conv;

typedef struct Model_Dense {
    Model_Types model_type;
    int num_dense_layers;
    Layer_Dense* layers;
    float threshold;
    const float (*dvectors)[MAX_DVECTOR_SIZE];
    int num_references;
    int d_vector_size;
} Model_Dense;

extern Model_Dense* kws;
extern Model_Conv* sv_128;
extern Model_Conv* sv_256;
extern Model_Dense* sv_dense_128_256;
extern Model_Dense* sv_dense_256_unbalanced;
extern Model_Dense* sv_dense_256_192;
extern Model_Dense* sv_dense_256_240;
extern Model_Dense* sv_dense_256_256;

extern Shape_Conv* shapes_conv;
extern Shape_Dense* shapes_dense;
extern const int kernel_size;
static const float thresholds[NUM_THRESHOLDS]={0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95};
static const int references[NUM_REFERENCES]={1, 8, 16, 64};

int compute_flatten_size_conv(Shape_Conv conv);
int compute_flatten_size_dense(Shape_Dense dense);
void emergency();
void allocate_float_array(float** array, int16_t size);
void deallocate_float_array(float* array);
void deallocate_conv_model(Model_Conv* conv);
void deallocate_dense_model (Model_Dense* dense);
void allocate_shapes_conv();
void allocate_shapes_dense();
void allocate_intermediate_shapes_conv(Model_Conv* conv);
void allocate_intermediate_shapes_dense(Model_Dense* dense);
void allocate_conv_model(Model_Conv** conv);
void allocate_dense_model(Model_Dense** dense);
void start_time_computing (struct timespec* start);
void end_time_computing (struct timespec* start, struct timespec* end, long* nsecs);
char* decompose_neurons();
void save_results(long nsecs, float prob, int8_t result, float threshold, int model_size, Model_Types model);
void adapt_shape_output_to_model(int num_neurons);
int model_size_computation_conv(Model_Conv* model);
int model_size_computation_dense(Model_Dense* model);
char* mode_output();

extern int8_t mode;
extern int8_t sv_elaborate;
extern int16_t dvector_model;
extern int8_t dense_neurons_mode;
extern int8_t num_refs;
extern int8_t bypass_kws;
extern int8_t bypass_sv;
//extern float threshold;

#endif
