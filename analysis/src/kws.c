#include "dense_network.h"
#include "kws/kws.h"
#include "kws/kwswb.h"
#include "text_save.h"

int8_t processing_kws_model(const float* input) {
    float prob;
    struct timespec start, end;
    long nsecs;
    init_kws_model();
    start_time_computing(&start);
    int8_t result=dense_neural_network(kws, input, &prob);
    end_time_computing(&start, &end, &nsecs);
    if(bypass_kws==0) {
        for(int i=0; i<NUM_THRESHOLDS; i++) {
            kws->threshold=thresholds[i];
            int8_t temp_result=(prob>kws->threshold) ? 0 : 1;
            //save_results(nsecs, prob, result, kws->threshold, model_size_computation_dense(kws), KWS);
            populateStruct_dense(KWS_MODEL, kws, nsecs, prob, temp_result);
            writeStruct();
        }
    }
    return result;
}

void init_kws_model() {
    allocate_dense_model(&kws);
    kws->model_type=KWS;
    kws->num_dense_layers=4;
    kws->dvectors=NULL;
    kws->num_references=0;
    kws->threshold=0.8;
    kws->d_vector_size=0;
    adapt_shape_output_to_model(num_classes);
    allocate_layers_kws(&kws->layers, kws->num_dense_layers);
    allocate_intermediate_shapes_dense(kws);
}

void allocate_layers_kws(Layer_Dense** layers, int num_layers) {
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
                    .weights = sequential_dense_MatMul,
                    .biases = sequential_dense_BiasAdd_ReadVariableOp,
                    .activation = RELU
                };
                break;
            case 1:
                (*layers)[i] = (Layer_Dense) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .input_shape = shapes_dense[5],
                    .output_shape = shapes_dense[5],
                    .weights = sequential_dense_1_MatMul,
                    .biases = sequential_dense_1_BiasAdd_ReadVariableOp,
                    .activation = RELU
                };
                break;
            case 2:
                (*layers)[i] = (Layer_Dense) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .input_shape = shapes_dense[5],
                    .output_shape = shapes_dense[5],
                    .weights = sequential_dense_2_MatMul,
                    .biases = sequential_dense_2_BiasAdd_ReadVariableOp,
                    .activation = RELU
                };
                break;
            case 3:
                (*layers)[i] = (Layer_Dense) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .input_shape = shapes_dense[5],
                    .output_shape = shapes_dense[1],
                    .weights = sequential_y_pred_MatMul,
                    .biases = sequential_y_pred_BiasAdd_ReadVariableOp,
                    .activation = SOFTMAX
                };
                break;
            default:
                printf("ERROR OVERFLOW MAX_LAYERS kws");
                emergency();
                exit(1);
        }
    }
}

