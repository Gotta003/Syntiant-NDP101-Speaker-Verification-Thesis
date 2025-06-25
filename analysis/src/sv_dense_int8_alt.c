#include "kws/kws.h"
#include "svq8/qat/sv_dense_int8_alt.h"
#include "svq8/qat/model_int8_alt.h"
#include "svq8/qat/d_vectors_dense_int8_alt.h"
#include "assert.h"

void fully_connected_layer_int8_alt(const float input[], float output[], const int8_t weights[], const float biases[], int input_size, int output_size) {
    //int weight_size=input_size*output_size;
    //int biases_size=output_size;
    assert(input != NULL && output != NULL && weights != NULL && biases != NULL);
    assert(input_size > 0 && output_size > 0);
    for(int j=0; j<output_size; j++) {
        float z=0.0f;
        for(int i=0; i<input_size; i++) {
            z+=weights[j*input_size+i]*input[i];
        }
        z+=biases[j];
        output[j]=relu(z);
    }
}

float cosine_similarity_int8_alt(const float vec1[OUTPUT_SIZE_SV_INT8_ALT], const float vec2[OUTPUT_SIZE_SV_INT8_ALT]) {
    float dot_product=0.0f;
    float norm_vec1=0.0f;
    float norm_vec2=0.0f;
    for(int i=0; i<OUTPUT_SIZE_SV_INT8_ALT; i++) {
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

float compute_similarity_int8_alt(const float input_vector[OUTPUT_SIZE_SV_INT8_ALT], const float d_vectors[][OUTPUT_SIZE_SV_INT8_ALT], int num_vectors) {
    float max_similarity=-1.0f;
    for(int i=0; i<num_vectors; i++) {
        float similarity=cosine_similarity_int8_alt(input_vector, d_vectors[i]);
        if(similarity>max_similarity) {
            max_similarity=similarity;
        }
    }
    return max_similarity;
}

void bestmatching_int8_alt(const float input_vectors[][OUTPUT_SIZE_SV_INT8_ALT], const float d_vectors[][OUTPUT_SIZE_SV_INT8_ALT], float y_prediction_prob[], int num_inputs, int vector_size) {
    for(int i=0; i<num_inputs; i++) {
        y_prediction_prob[i]=compute_similarity_int8_alt(input_vectors[i], d_vectors, vector_size);
    }
}

void mean_d_vector_int8_alt(const float input_vectors[][OUTPUT_SIZE_SV_INT8_ALT], float mean_output[OUTPUT_SIZE_SV_INT8_ALT], int num_vectors) {
    for(int i=0; i<OUTPUT_SIZE_SV_INT8_ALT; i++) {
        float sum=0.0f;
        for(int j=0; j<num_vectors; j++) {
            sum+=input_vectors[j][i];
        }
        mean_output[i]=sum/num_vectors;
    }
}

void mean_cos_int8_alt(const float input_vectors[][OUTPUT_SIZE_SV_INT8_ALT], const float d_vectors[][OUTPUT_SIZE_SV_INT8_ALT], float y_prediction_prob[], int num_inputs, int vector_size) {
    float mean_d_vect[OUTPUT_SIZE_SV_INT8_ALT];
    mean_d_vector_int8_alt(d_vectors, mean_d_vect, vector_size);
    for(int i=0; i<num_inputs; i++) {
        y_prediction_prob[i]=cosine_similarity_int8_alt(input_vectors[i], mean_d_vect);
    }
}

int sv_dense_int8_alt_neural_network(const float input[INPUT_SIZE_SV_INT8_ALT], int sv_elaborate) {
    float fc1[HIDDEN_LAYER_1_SIZE_SV_INT8_ALT];
    float fc2[HIDDEN_LAYER_2_SIZE_SV_INT8_ALT];
    float fc3[HIDDEN_LAYER_3_SIZE_SV_INT8_ALT];
    float output[OUTPUT_SIZE_SV_INT8_ALT];
    
    fully_connected_layer_int8_alt(input, fc1, sequential_dense_1_MatMul_SV_INT8_ALT, sequential_dense_1_BiasAdd_ReadVariableOp_SV_INT8_ALT, INPUT_SIZE_SV_INT8_ALT, HIDDEN_LAYER_1_SIZE_SV_INT8_ALT);
    fully_connected_layer_int8_alt(fc1, fc2, sequential_dense_2_MatMul_SV_INT8_ALT, sequential_dense_2_BiasAdd_ReadVariableOp_SV_INT8_ALT, HIDDEN_LAYER_1_SIZE_SV_INT8_ALT, HIDDEN_LAYER_2_SIZE_SV_INT8_ALT);
    fully_connected_layer_int8_alt(fc2, fc3, sequential_dense_3_MatMul_SV_INT8_ALT, sequential_dense_3_BiasAdd_ReadVariableOp_SV_INT8_ALT, HIDDEN_LAYER_2_SIZE_SV_INT8_ALT, HIDDEN_LAYER_3_SIZE_SV_INT8_ALT);
    fully_connected_layer_int8_alt(fc3, output, sequential_dense_4_MatMul_SV_INT8_ALT, sequential_dense_4_BiasAdd_ReadVariableOp_SV_INT8_ALT, HIDDEN_LAYER_3_SIZE_SV_INT8_ALT, OUTPUT_SIZE_SV_INT8_ALT);
    
    int num_inputs=1;
    float prob_0[num_inputs];
    float input_vectors[num_inputs][OUTPUT_SIZE_SV_INT8_ALT];
    assert(sizeof(input_vectors[0]) == sizeof(float) * OUTPUT_SIZE_SV_INT8_ALT);
    memcpy(input_vectors[0], output, sizeof(float) * OUTPUT_SIZE_SV_INT8_ALT);
    if(sv_elaborate==0) {
        bestmatching_int8_alt(input_vectors, d_vectors_0_16_SV_INT8_ALT, prob_0, num_inputs, 16);
    }
    else {
        mean_cos_int8_alt(input_vectors, d_vectors_0_16_SV_INT8_ALT, prob_0, num_inputs, 16);
    }
    printf("PROB 0: %.6f", prob_0[0]);
    return (prob_0[0]>SIMILARITY_THRESHOLD_INT8_ALT ? 0 : 1);
}
