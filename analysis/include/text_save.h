#ifndef __TEXT_SAVE_H__
#define __TEXT_SAVE_H__

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdbool.h"
#include "stdint.h"
#include "time.h"
#include "model.h"

#define KWS_MODEL "kws_model.csv"
#define SV_CONV_128 "sv_conv_128_model.csv"
#define SV_CONV_256 "sv_conv_256_model.csv"
#define SV_DENSE_128_256 "sv_dense_128_256_model.csv"
#define SV_DENSE_U256_128 "sv_dense_u256_128_model.csv"
#define SV_DENSE_256_192 "sv_dense_256_192_model.csv"
#define SV_DENSE_256_256 "sv_dense_256_256_model.csv"
#define SV_DENSE_256_240 "sv_dense_256_240_model.csv"

typedef struct FileStruct {
    char* filename;
    Model_Types type;
    int model_size;
    int model_output_size;
    char* neurons_dense;
    int num_refs;
    int method; //BENCHMARKING 0 MEANCOS 1
    int ntime;
    float prob;
    float threshold;
    int result;
} FileStruct;

extern FileStruct file;

void populateStruct_conv(char* filename, Model_Conv* model_conv, int ntime, float prob, int result);
void populateStruct_dense(char* filename, Model_Dense* model_dense, int ntime, float prob, int result);
void writeStruct();

#endif
