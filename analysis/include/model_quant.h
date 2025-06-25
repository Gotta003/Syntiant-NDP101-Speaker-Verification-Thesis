#ifndef __MODEL_QUANT_H__
#define __MODEL_QUANT_H__

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdbool.h"
#include "stdint.h"
#include "time.h"
#include "model.h"

typedef struct Layer_Dense_Q8 {
    int8_t* in_data;
    int8_t* out_data;
    Shape_Dense input_shape;
    Shape_Dense output_shape;
    const int8_t* weights;
    const int32_t* biases;
    Activation activation;
} Layer_Dense_Q8;

typedef struct Model_Quant8_Dense {
    Model_Types model_type;
    int num_dense_layers;
    Layer_Dense_Q8* layers;
    float threshold;
    const int8_t (*dvectors)[MAX_DVECTOR_SIZE];
    int num_references;
    int d_vector_size;
} Model_Quant8_Dense;
//extern Model_Conv sv_256;

#endif
