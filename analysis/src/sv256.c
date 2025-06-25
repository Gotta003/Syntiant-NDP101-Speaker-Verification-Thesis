#include "sv_conv.h"
#include "svconv/sv256/sv256.h"
#include "svconv/sv256/sv256wb.h"
#include "svconv/sv256/sv256dvs.h"
#include "text_save.h"

void processing_sv_256_model(const float* input) {
    float prob;
    struct timespec start, end;
    long nsecs;
    int8_t result;
    init_sv_256_model();
    int8_t sv_temp=sv_elaborate;
    sv_elaborate=0;
    int8_t refs=num_refs;
    bool verification=true;
    for (int i=0; i<NUM_REFERENCES; i++) {
        num_refs=references[i];
        result=sv_conv_neural_network(sv_256, input, &prob);
        if(prob>=0.99f){
            verification=false;
            break;
        } 
    }
    num_refs=refs;
    sv_elaborate=sv_temp;
    if(verification) {
        start_time_computing(&start);
        result=sv_conv_neural_network(sv_256, input, &prob);
        end_time_computing(&start, &end, &nsecs);
        for(int i=0; i<NUM_THRESHOLDS; i++) {
            sv_256->threshold=thresholds[i];
            result=(prob>sv_256->threshold) ? 0 : 1;
            //save_results(nsecs, prob, result, sv_256->threshold, model_size_computation_conv(sv_256), SV_CONV);
            populateStruct_conv(SV_CONV_256, sv_256, nsecs, prob, result);
            writeStruct();
        }
    }
}

void choose_references_SV256() {
    switch(num_refs) {
        case 1:
            sv_256->dvectors=d_vectors_0_1;
            break;
        case 8:
            sv_256->dvectors=d_vectors_0_8;
            break;
        case 16:
            sv_256->dvectors=d_vectors_0_16;
            break;
        case 64:
            sv_256->dvectors=d_vectors_0_64;
            break;
        default:
            printf("Error Memory allocation");
            emergency();
            exit(1); 
            break;
    }
}

void init_sv_256_model() {
    allocate_conv_model(&sv_256);
    sv_256->model_type=SV_CONV;
    sv_256->num_conv_layers=6;
    choose_references_SV256();
    sv_256->num_references=num_refs;
    sv_256->threshold=0.8;
    sv_256->batch_norm=true;
    sv_256->beta=beta[0];
    sv_256->gamma=gamma[0];
    sv_256->d_vector_size=256;
    allocate_layers_sv_256(&sv_256->layers, sv_256->num_conv_layers);
    allocate_intermediate_shapes_conv(sv_256);
}


void allocate_layers_sv_256(Layer_Conv** layers, int num_layers) {
    *layers = malloc(sizeof(Layer_Conv) * num_layers);
    if(*layers == NULL) {
        printf("Error Memory allocation");
        emergency();
        exit(1);
    }
    
    for(int i = 0; i < num_layers; i++) {
        switch(i) {
            case 0:
                (*layers)[i] = (Layer_Conv) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .type_layer_conv = CONV2D,
                    .input_shape = shapes_conv[i],
                    .output_shape = shapes_conv[i+1],
                    .weights = conv_1_Weights,
                    .biases = conv_1_BiasAdd_ReadVariableOp,
                    .activation = RELU,
                    .padding_conv = SAME,
                    .kernel_size = kernel_size,
                    .stride = 1
                };
                break;
            case 1:
                (*layers)[i] = (Layer_Conv) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .type_layer_conv = MAXPOOL2D,
                    .input_shape = shapes_conv[i],
                    .output_shape = shapes_conv[i+1],
                    .weights = NULL,
                    .biases = NULL,
                    .activation = NONE,
                    .padding_conv = VALID,
                    .kernel_size = 3,
                    .stride = 3
                };
                break;
            case 2:
                (*layers)[i] = (Layer_Conv) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .type_layer_conv = CONV2D,
                    .input_shape = shapes_conv[i],
                    .output_shape = shapes_conv[i+1],
                    .weights = conv_2_Weights,
                    .biases = conv_2_BiasAdd_ReadVariableOp,
                    .activation = RELU,
                    .padding_conv = SAME,
                    .kernel_size = kernel_size,
                    .stride = 1
                };
                break;
            case 3:
                (*layers)[i] = (Layer_Conv){
                    .in_data = NULL,
                    .out_data = NULL,
                    .type_layer_conv = MAXPOOL2D,
                    .input_shape = shapes_conv[3],
                    .output_shape = shapes_conv[4],
                    .weights = NULL,
                    .biases = NULL,
                    .activation = NONE,
                    .padding_conv = VALID,
                    .kernel_size = 2,
                    .stride = 2
                };
                break;
            case 4:
                (*layers)[i] = (Layer_Conv) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .type_layer_conv = CONV2D,
                    .input_shape = shapes_conv[4],
                    .output_shape = shapes_conv[5],
                    .weights = conv_3_Weights,
                    .biases = conv_3_BiasAdd_ReadVariableOp,
                    .activation = RELU,
                    .padding_conv = SAME,
                    .kernel_size = kernel_size,
                    .stride = 2
                };
                break;
            case 5:
                (*layers)[i] = (Layer_Conv) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .type_layer_conv = CONV2D,
                    .input_shape = shapes_conv[5],
                    .output_shape = shapes_conv[6],
                    .weights = conv_4_Weights,
                    .biases = conv_4_BiasAdd_ReadVariableOp,
                    .activation = RELU,
                    .padding_conv = SAME,
                    .kernel_size = kernel_size,
                    .stride = 2
                };
                break;
            default:
                printf("ERROR OVERFLOW MAX_LAYERS SV_256");
                emergency();
                exit(1);
        }
    }
}
