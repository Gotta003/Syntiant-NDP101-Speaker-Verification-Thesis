#include "kws/kws.h"
#include "sv_conv.h"
#include "svq4/d_vector_int4.h"
#include "svq4/sv_dense_int4.h"
#include "svq4/model_int4.h"
#include "spectrogram.h"
#ifndef __D_VECTOR_INT4__
#include <dirent.h>
#include <string.h>
#define NUM_FILES 16
#endif

#define SAMPLE_RATE 16000
#define AUDIO_WINDOW 0.032
#define DURATION_SECONDS 1
#define SIMILARITY_THRESHOLD 0.7

static int count_q=0;

float cosine_similarity_int(const int8_t vec1[DVECTORS], const int8_t vec2[DVECTORS]) {
    float dot_product=0.0f;
    float norm_vec1=0.0f;
    float norm_vec2=0.0f;
    for(int i=0; i<DVECTORS; i++) {
        dot_product+=vec1[i]*vec2[i];
        norm_vec1+=vec1[i]*vec1[i];
        norm_vec2+=vec2[i]*vec2[i];
    }
    norm_vec1=sqrtf(norm_vec1);
    norm_vec2=sqrtf(norm_vec2);
    if(norm_vec1==0 || norm_vec2==0) {
        return 0.0f;
    }
    return dot_product/(norm_vec1*norm_vec2);
}

float compute_similarity_int(const int8_t input_vector[DVECTORS], const int8_t d_vectors[][DVECTORS], int num_vectors) {
    float max_similarity=-1.0f;
    for(int i=0; i<num_vectors; i++) {
        float similarity=cosine_similarity_int(input_vector, d_vectors[i]);
        if(similarity>max_similarity) {
            max_similarity=similarity;
        }
    }
    return max_similarity;
}

void bestmatching_int(const int8_t input_vectors[][DVECTORS], const int8_t d_vectors[][DVECTORS], float y_prediction_prob[], int num_inputs, int vector_size) {
    for(int i=0; i<num_inputs; i++) {
        y_prediction_prob[i]=compute_similarity_int(input_vectors[i], d_vectors, vector_size);
    }
}

void mean_d_vector_int(const int8_t input_vectors[][DVECTORS], int8_t mean_output[DVECTORS], int num_vectors) {
    for(int i=0; i<DVECTORS; i++) {
        float sum=0.0f;
        for(int j=0; j<num_vectors; j++) {
            sum+=input_vectors[j][i];
        }
        mean_output[i]=(int8_t)(sum/num_vectors);
    }
}

void mean_cos_int(const int8_t input_vectors[][DVECTORS], const int8_t d_vectors[][DVECTORS], float y_prediction_prob[], int num_inputs, int vector_size) {
    int8_t mean_d_vect[DVECTORS];
    mean_d_vector_int(d_vectors, mean_d_vect, vector_size);
    for(int i=0; i<num_inputs; i++) {
        y_prediction_prob[i]=cosine_similarity_int(input_vectors[i], mean_d_vect);
    }
}

void quantization(const float input[INPUT_SIZE_SV_INT4], int8_t quantized_input[INPUT_SIZE_SV_INT4]) {
    float scale=0.003921539522707462;
    int dimension=-128;
    for(int i=0; i<INPUT_SIZE_SV_INT4; i++) {
        quantized_input[i]=(int8_t)(input[i]/scale-dimension);
    }
} 

void fully_connected_layer_int(const int8_t input[], int8_t output[],const int8_t weights[], const int32_t biases[], int input_size, int output_size) {
    //int weight_size=input_size*output_size;
    //int biases_size=output_size;
    for(int j=0; j<output_size; j++) {
        float z=0.0f;
        for(int i=0; i<input_size; i++) {
            z+=weights[j*input_size+i]*input[i];
        }
        z+=biases[j];
        output[j]=relu(z);
    }
} 

void features_extraction(const float input[INPUT_SIZE_SV_INT4], int8_t output[OUTPUT_SIZE_SV_INT4]) {
    int8_t input_q[INPUT_SIZE_SV_INT4];
    int8_t fc1[HIDDEN_LAYER_1_SIZE_SV_INT4];
    int8_t fc2[HIDDEN_LAYER_2_SIZE_SV_INT4];
    int8_t fc3[HIDDEN_LAYER_3_SIZE_SV_INT4];

    quantization(input, input_q);
    fully_connected_layer_int(input_q, fc1, sequential_dense_1_MatMul_SV_INT4, sequential_dense_1_BiasAdd_ReadVariableOp_SV_INT4, INPUT_SIZE_SV_INT4, HIDDEN_LAYER_1_SIZE_SV_INT4);
    fully_connected_layer_int(fc1, fc2, sequential_dense_2_MatMul_SV_INT4, sequential_dense_2_BiasAdd_ReadVariableOp_SV_INT4, HIDDEN_LAYER_1_SIZE_SV_INT4, HIDDEN_LAYER_2_SIZE_SV_INT4);
    fully_connected_layer_int(fc2, fc3, sequential_dense_3_MatMul_SV_INT4, sequential_dense_3_BiasAdd_ReadVariableOp_SV_INT4, HIDDEN_LAYER_2_SIZE_SV_INT4, HIDDEN_LAYER_3_SIZE_SV_INT4);
    fully_connected_layer_int(fc3, output, sequential_dense_4_MatMul_SV_INT4, sequential_dense_4_BiasAdd_ReadVariableOp_SV_INT4, HIDDEN_LAYER_3_SIZE_SV_INT4, OUTPUT_SIZE_SV_INT4);
}

int sv_dense_int4_neural_network(const float input[INPUT_SIZE_SV_INT4], int sv_elaborate) {
    int8_t output[OUTPUT_SIZE_SV_INT4];
    features_extraction(input, output);

    int num_inputs=1;
    float prob_0[num_inputs];
    int8_t input_vectors[1][DVECTORS];
    memcpy(input_vectors[0], output, sizeof(int8_t) * DVECTORS);
    if(sv_elaborate==0) {
        bestmatching_int(input_vectors, d_vector_array_16_SV_INT4, prob_0, num_inputs, 16);
    }
    else {
        if(sv_elaborate==1) {
            mean_cos_int(input_vectors, d_vector_array_16_SV_INT4, prob_0, num_inputs, 16);
        }
    }
    printf("prob 0: %.6f", prob_0[0]);
    return (prob_0[0]>SIMILARITY_THRESHOLD ? 0 : 1);
}
