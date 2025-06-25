#include "kws/kws.h"
#include "svq8/pqt/sv_dense_int8.h"
#include "svq8/pqt/model_int8.h"
#include "svq8/pqt/d_vectors_dense_int8.h"
#include "assert.h"
#include "math.h"

static int layer=0;
void QuantizeMultiplier(double real_multiplier, int32_t* quantized_multiplier, int* shift) {
    if (real_multiplier == 0.0f) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }
    
    int exponent;
    float significand = frexpf(real_multiplier, &exponent);
    if(significand<0.5f) {
        significand *= 2.0f;  // Adjust for frexpf range [0.5, 1.0)
        exponent--;
    }
    
    int64_t q=(int64_t)round(significand * (1ll << 30));
    
    if (q == (1ll << 30)) {
        q /= 2;
        exponent++;
    }
    *quantized_multiplier=(int32_t)q;
    *shift=30-exponent;
    if(layer==0) {
        printf("real_multiplier=%.12f, significand=%.6f, quantized_multiplier=%d, shift=%d\n", 
           real_multiplier, significand, *quantized_multiplier, *shift);
    }
}

int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift) {
    int64_t result = (int64_t)x * (int64_t)quantized_multiplier;
    if(shift>0) {
        int64_t rounding=1LL << (shift-1);
        if(result>=0) {
            result+=rounding;
        }
        else {
            result-=rounding;
        }
        result>>=shift;
    }
    else if (shift<0) {
        result<<=-shift;
    }

    if(result>INT32_MAX) result=INT32_MAX;
    if(result<INT32_MIN) result=INT32_MIN;
    if(layer==0) {
        printf("x=%d, shift=%d, result=%lld\t", x, shift, result);
    }
    return (int32_t)result;
}

float switching_scales(Layer layer) {
    float scale = 0.0f;
    //240
    switch(layer) {
        case INPUT:
            scale = 0.003921568859368563f;
            break;
        case L1:
            scale = 0.07197827845811844f;
            break;
        case L2:
            scale = 0.08697842061519623f;
            break;
        case L3:
            scale = 0.10738003253936768f;
            break;
        case OUTPUT:
            scale = 0.37138763070106506f;
            break;
        default:
            scale = 1.0f;
            break;
    }
    return scale;
}

void fully_connected_layer_int8(const int8_t input[], int8_t output[], const int8_t weights[], const float weight_scales[], const int32_t biases[], int input_size, int output_size, Layer layer_input, Layer layer_output) {
    float input_scale = switching_scales(layer_input);
    float output_scale = switching_scales(layer_output);
    
    int32_t input_zero_point=-128;
    int32_t weight_zero_point=0;
    int32_t output_zero_point=-128;

    for(int j = 0; j < output_size; j++) {
        int32_t z = 0;
        for(int i = 0; i < input_size; i++) {
            int32_t input_val=(int32_t)input[i]-input_zero_point;
            int32_t weight_val=(int32_t)weights[j*input_size+i]-weight_zero_point;
            z += (input_val*weight_val);
        }
        z += biases[j];
        if(layer==0) {
            printf("z (acc): %d\t", z);
        }

        float real_multiplier = (input_scale * weight_scales[j]) / output_scale;
        int32_t output_multiplier;
        int shift;
        QuantizeMultiplier(real_multiplier, &output_multiplier, &shift);
        
        int32_t scaled = MultiplyByQuantizedMultiplier(z, output_multiplier, shift);
        scaled+=output_zero_point;
        // Clamp to int8 range
        scaled = scaled < -128 ? -128 : scaled;
        scaled = scaled > 127 ? 127 : scaled;
        
        output[j] = (int8_t)scaled;
        if(layer==0) {
            printf("Final: %d\n", scaled);
        }
    }
    layer++;
}


float cosine_similarity_int8(const float vec1[OUTPUT_SIZE_SV_INT8], const float vec2[OUTPUT_SIZE_SV_INT8]) {
    float dot_product=0.0f;
    float norm_vec1=0.0f;
    float norm_vec2=0.0f;
    for(int i=0; i<OUTPUT_SIZE_SV_INT8; i++) {
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

float compute_similarity_int8(const float input_vector[OUTPUT_SIZE_SV_INT8], const float d_vectors[][OUTPUT_SIZE_SV_INT8], int num_vectors) {
    float max_similarity=-1.0f;
    for(int i=0; i<num_vectors; i++) {
        float similarity=cosine_similarity_int8(input_vector, d_vectors[i]);
        if(similarity>max_similarity) {
            max_similarity=similarity;
        }
    }
    return max_similarity;
}

void bestmatching_int8(const float input_vectors[][OUTPUT_SIZE_SV_INT8], const float d_vectors[][OUTPUT_SIZE_SV_INT8], float y_prediction_prob[], int num_inputs, int vector_size) {
    for(int i=0; i<num_inputs; i++) {
        y_prediction_prob[i]=compute_similarity_int8(input_vectors[i], d_vectors, vector_size);
    }
}

void mean_d_vector_int8(const float input_vectors[][OUTPUT_SIZE_SV_INT8], float mean_output[OUTPUT_SIZE_SV_INT8], int num_vectors) {
    for(int i=0; i<OUTPUT_SIZE_SV_INT8; i++) {
        float sum=0.0f;
        for(int j=0; j<num_vectors; j++) {
            sum+=input_vectors[j][i];
        }
        mean_output[i]=sum/num_vectors;
    }
}

void mean_cos_int8(const float input_vectors[][OUTPUT_SIZE_SV_INT8], const float d_vectors[][OUTPUT_SIZE_SV_INT8], float y_prediction_prob[], int num_inputs, int vector_size) {
    float mean_d_vect[OUTPUT_SIZE_SV_INT8];
    mean_d_vector_int8(d_vectors, mean_d_vect, vector_size);
    for(int i=0; i<num_inputs; i++) {
        y_prediction_prob[i]=cosine_similarity_int8(input_vectors[i], mean_d_vect);
    }
}

void input_quantization_int8(const float* input, int8_t* q_input, int size, float scale, int32_t zero_point) {
    for(int i=0; i<size; i++) {
        int8_t val=(int8_t)round(input[i]/scale)+zero_point;
        val = val < -128.0f ? -128.0f : val;
        val = val > 127.0f ? 127.0f : val;
        q_input[i]=(int8_t)roundf(val);
    }
}

void input_dequantization_int32(int32_t* q_output, float* output, int size, float scale, int32_t zero_point) {
    for(int i=0; i<size; i++) {
        output[i]=scale*(q_output[i]-zero_point);
    }
}

void input_dequantization_int8(int8_t* q_output, float* output, int size, float scale, int32_t zero_point) {
    for(int i=0; i<size; i++) {
        output[i]=scale*(q_output[i]-zero_point);
    }
}

int sv_dense_int8_neural_network(const float input[INPUT_SIZE_SV_INT8], int sv_elaborate) {
    int8_t new_input[INPUT_SIZE_SV_INT8];
    int8_t fc1[HIDDEN_LAYER_1_SIZE_SV_INT8];
    int8_t fc2[HIDDEN_LAYER_2_SIZE_SV_INT8];
    int8_t fc3[HIDDEN_LAYER_3_SIZE_SV_INT8];
    int8_t output[OUTPUT_SIZE_SV_INT8];
    float new_output[OUTPUT_SIZE_SV_INT8];
    
    
    input_quantization_int8(input, new_input, INPUT_SIZE_SV_INT8, switching_scales(INPUT), -128);

    fully_connected_layer_int8(new_input, fc1, sequential_dense_1_MatMul_SV_INT8, scales_dense1_SV_INT8, sequential_dense_1_BiasAdd_ReadVariableOp_SV_INT8, INPUT_SIZE_SV_INT8, HIDDEN_LAYER_1_SIZE_SV_INT8, INPUT, L1);
    printf("\n\n\n\n");
    for(int i=0; i<HIDDEN_LAYER_1_SIZE_SV_INT8; i++) {
        printf("%d\t", fc1[i]);
        if(i%16==15) {
            printf("\n");
        }
    }
    fully_connected_layer_int8(fc1, fc2, sequential_dense_2_MatMul_SV_INT8, scales_dense2_SV_INT8, sequential_dense_2_BiasAdd_ReadVariableOp_SV_INT8, HIDDEN_LAYER_1_SIZE_SV_INT8, HIDDEN_LAYER_2_SIZE_SV_INT8, L1, L2);
    printf("\n\n\n\n");
    fully_connected_layer_int8(fc2, fc3, sequential_dense_3_MatMul_SV_INT8, scales_dense3_SV_INT8, sequential_dense_3_BiasAdd_ReadVariableOp_SV_INT8, HIDDEN_LAYER_2_SIZE_SV_INT8, HIDDEN_LAYER_3_SIZE_SV_INT8, L2, L3);
    printf("\n\n\n\n");
    fully_connected_layer_int8(fc3, output, sequential_dense_4_MatMul_SV_INT8, scales_output_SV_INT8, sequential_dense_4_BiasAdd_ReadVariableOp_SV_INT8, HIDDEN_LAYER_3_SIZE_SV_INT8, OUTPUT_SIZE_SV_INT8, L3, OUTPUT);
    printf("\n\n\n\n");
    for(int i=0; i<OUTPUT_SIZE_SV_INT8; i++) {
        printf("%d\t", output[i]);
        if(i%16==15) {
            printf("\n");
        }
    }
    
    input_dequantization_int8(output, new_output, OUTPUT_SIZE_SV_INT8, switching_scales(OUTPUT), -128);
    
    /*for(int i=0; i<OUTPUT_SIZE_SV_INT8; i++) {
        printf("%.6f\t", new_output[i]);
        if(i%8==7) {
            printf("\n");
        }
    }*/
    
    int num_inputs=1;
    float prob_0[num_inputs];
    float input_vectors[num_inputs][OUTPUT_SIZE_SV_INT8];
    assert(sizeof(input_vectors[0]) == sizeof(float) * OUTPUT_SIZE_SV_INT8);
    memcpy(input_vectors[0], new_output, sizeof(float) * OUTPUT_SIZE_SV_INT8);
    if(sv_elaborate==0) {
        bestmatching_int8(input_vectors, d_vectors_0_16_SV_INT8, prob_0, num_inputs, 16);
    }
    else {
        mean_cos_int8(input_vectors, d_vectors_0_16_SV_INT8, prob_0, num_inputs, 16);
    }
    printf("PROB 0: %.6f", prob_0[0]);
    return (prob_0[0]>SIMILARITY_THRESHOLD_INT8 ? 0 : 1);
}
