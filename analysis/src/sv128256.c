#include "sv_dense_network.h"
#include "svdense/sv128256/sv128256.h"
#include "svdense/sv128256/sv128256wb.h"
#include "svdense/sv128256/sv128256dvs.h"
#include "text_save.h"

void processing_sv128256_model(const float* input) {
    float prob;
    struct timespec start, end;
    long nsecs;
    int8_t result;
    init_sv128256_model();
    int8_t sv_temp=sv_elaborate;
    sv_elaborate=0;
    int8_t refs=num_refs;
    bool verification=true;
    for (int i=0; i<NUM_REFERENCES; i++) {
        num_refs=references[i];
        result=sv_dense_neural_network(sv_dense_128_256, input, &prob);
        if(prob>=0.99f){
            verification=false;
            break;
        } 
    }
    num_refs=refs;
    sv_elaborate=sv_temp;
    if(verification) {
        start_time_computing(&start);
        result=sv_dense_neural_network(sv_dense_128_256, input, &prob);
        end_time_computing(&start, &end, &nsecs);
        for(int i=0; i<NUM_THRESHOLDS; i++) {
            sv_dense_128_256->threshold=thresholds[i];
            result=(prob>sv_dense_128_256->threshold) ? 0 : 1;
            //save_results(nsecs, prob, result, sv_dense_128_256->threshold, model_size_computation_dense(sv_dense_128_256), SV_DENSE);
            populateStruct_dense(SV_DENSE_128_256, sv_dense_128_256, nsecs, prob, result);
            writeStruct();
        }
    }
}

void choose_references_SV128256() {
    switch(num_refs) {
        case 1:
            sv_dense_128_256->dvectors=d_vectors_0_1_SV128256;
            break;
        case 8:
            sv_dense_128_256->dvectors=d_vectors_0_8_SV128256;
            break;
        case 16:
            sv_dense_128_256->dvectors=d_vectors_0_16_SV128256;
            break;
        case 64:
            sv_dense_128_256->dvectors=d_vectors_0_64_SV128256;
            break;
        default:
            printf("Error Memory allocation");
            emergency();
            exit(1); 
            break;
    }
}

void init_sv128256_model() {
    allocate_dense_model(&sv_dense_128_256);
    sv_dense_128_256->model_type=SV_DENSE;
    sv_dense_128_256->num_dense_layers=4;
    choose_references_SV128256();
    sv_dense_128_256->num_references=num_refs;
    sv_dense_128_256->threshold=0.8;
    sv_dense_128_256->d_vector_size=128;
    adapt_shape_output_to_model(sv_dense_128_256->d_vector_size);
    allocate_layers_sv128256(&sv_dense_128_256->layers, sv_dense_128_256->num_dense_layers);
    allocate_intermediate_shapes_dense(sv_dense_128_256);
}

void allocate_layers_sv128256(Layer_Dense** layers, int num_layers) {
    *layers = malloc(sizeof(Layer_Dense) * num_layers);
    if(*layers == NULL) {
        printf("Error Memory allocation");
        emergency();
        exit(1);
    }
    
    for(int i = 0; i < num_layers; i++) {
        switch(i) {
            case 0:
                (*layers)[i] = (Layer_Dense) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .input_shape = shapes_dense[0],
                    .output_shape = shapes_dense[5],
                    .weights = sequential_dense_1_MatMul_SV128256,
                    .biases = sequential_dense_1_BiasAdd_ReadVariableOp_SV128256,
                    .activation = RELU
                };
                break;
            case 1:
                (*layers)[i] = (Layer_Dense) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .input_shape = shapes_dense[5],
                    .output_shape = shapes_dense[5],
                    .weights = sequential_dense_2_MatMul_SV128256,
                    .biases = sequential_dense_2_BiasAdd_ReadVariableOp_SV128256,
                    .activation = RELU
                };
                break;
            case 2:
                (*layers)[i] = (Layer_Dense) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .input_shape = shapes_dense[5],
                    .output_shape = shapes_dense[5],
                    .weights = sequential_dense_3_MatMul_SV128256,
                    .biases = sequential_dense_3_BiasAdd_ReadVariableOp_SV128256,
                    .activation = RELU
                };
                break;
            case 3:
                (*layers)[i] = (Layer_Dense) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .input_shape = shapes_dense[5],
                    .output_shape = shapes_dense[1],
                    .weights = sequential_dense_4_MatMul_SV128256,
                    .biases = sequential_dense_4_BiasAdd_ReadVariableOp_SV128256,
                    .activation = RELU
                };
                break;
            default:
                printf("ERROR OVERFLOW MAX_LAYERS SV_DENSE_128_256");
                emergency();
                exit(1);
        }
    }
}

