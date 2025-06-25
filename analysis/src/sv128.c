#include "sv_conv.h"
#include "svconv/sv128/sv128.h"
#include "svconv/sv128/sv128wb.h"
#include "svconv/sv128/sv128dvs.h"
#include "text_save.h"

void processing_sv_128_model(const float* input) {
    float prob;
    struct timespec start, end;
    long nsecs;
    int8_t result;
    init_sv_128_model();
    int8_t sv_temp=sv_elaborate;
    sv_elaborate=0;
    int8_t refs=num_refs;
    bool verification=true;
    for (int i=0; i<NUM_REFERENCES; i++) {
        num_refs=references[i];
        result=sv_conv_neural_network(sv_128, input, &prob);
        if(prob>=0.99f){
            verification=false;
            break;
        } 
    }
    num_refs=refs;
    sv_elaborate=sv_temp;
    if(verification) {
        start_time_computing(&start);
        result=sv_conv_neural_network(sv_128, input, &prob);
        end_time_computing(&start, &end, &nsecs);
        for(int i=0; i<NUM_THRESHOLDS; i++) {
            sv_128->threshold=thresholds[i];
            result=(prob>sv_128->threshold) ? 0 : 1;
            //save_results(nsecs, prob, result, sv_128->threshold, model_size_computation_conv(sv_128), SV_CONV);
            populateStruct_conv(SV_CONV_128, sv_128, nsecs, prob, result);
            writeStruct();
        }
    }
}

void choose_references_SV128() {
    switch(num_refs) {
        case 1:
            sv_128->dvectors=d_vectors_0_1_sv128;
            break;
        case 8:
            sv_128->dvectors=d_vectors_0_8_sv128;
            break;
        case 16:
            sv_128->dvectors=d_vectors_0_16_sv128;
            break;
        case 64:
            sv_128->dvectors=d_vectors_0_64_sv128;
            break;
        default:
            printf("Error Memory allocation");
            emergency();
            exit(1); 
            break;
    }
}

void init_sv_128_model() {
    allocate_conv_model(&sv_128);
    sv_128->model_type=SV_CONV;
    sv_128->num_conv_layers=7;
    choose_references_SV128();
    sv_128->num_references=num_refs;
    sv_128->threshold=0.8;
    sv_128->batch_norm=true;
    sv_128->beta=beta_sv128[0];
    sv_128->gamma=gamma_sv128[0];
    sv_128->d_vector_size=128;
    allocate_layers_sv_128(&sv_128->layers, sv_128->num_conv_layers);
    allocate_intermediate_shapes_conv(sv_128);
}

void allocate_layers_sv_128(Layer_Conv** layers, int num_layers) {
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
                    .input_shape = shapes_conv[0],
                    .output_shape = shapes_conv[1],
                    .weights = conv_1_Weights_sv128,
                    .biases = conv_1_BiasAdd_ReadVariableOp_sv128,
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
                    .input_shape = shapes_conv[1],
                    .output_shape = shapes_conv[2],
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
                    .input_shape = shapes_conv[2],
                    .output_shape = shapes_conv[3],
                    .weights = conv_2_Weights_sv128,
                    .biases = conv_2_BiasAdd_ReadVariableOp_sv128,
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
                    .weights = conv_3_Weights_sv128,
                    .biases = conv_3_BiasAdd_ReadVariableOp_sv128,
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
                    .weights = conv_4_Weights_sv128,
                    .biases = conv_4_BiasAdd_ReadVariableOp_sv128,
                    .activation = RELU,
                    .padding_conv = SAME,
                    .kernel_size = kernel_size,
                    .stride = 2
                };
                break;
            case 6:
                (*layers)[i] = (Layer_Conv) {
                    .in_data = NULL,
                    .out_data = NULL,
                    .type_layer_conv = CONV2D,
                    .input_shape = shapes_conv[6],
                    .output_shape = shapes_conv[7],
                    .weights = conv_5_Weights_sv128,
                    .biases = conv_5_BiasAdd_ReadVariableOp_sv128,
                    .activation = RELU,
                    .padding_conv = SAME,
                    .kernel_size = kernel_size,
                    .stride = 2
                };
                break;
            default:
                printf("ERROR OVERFLOW MAX_LAYERS SV_128");
                emergency();
                exit(1);
        }
    }
}

