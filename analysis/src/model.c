#include "model.h"
#include "text_save.h"

Model_Dense* kws=NULL;
Model_Conv* sv_128=NULL;
Model_Conv* sv_256=NULL;
Model_Dense* sv_dense_128_256=NULL;
Model_Dense* sv_dense_256_unbalanced=NULL;
Model_Dense* sv_dense_256_192=NULL;
Model_Dense* sv_dense_256_240=NULL;
Model_Dense* sv_dense_256_256=NULL;

Shape_Conv* shapes_conv=NULL;
Shape_Dense* shapes_dense=NULL;

const int kernel_size=3;

int8_t mode=-1;
int8_t sv_elaborate=-1;
int16_t dvector_model=-1;
int8_t dense_neurons_mode=-1;
int8_t num_refs=-1;
int8_t bypass_kws=-1;
int8_t bypass_sv=-1;
float threshold=0.8;

void allocate_shapes_conv() {
    shapes_conv=malloc(sizeof(Shape_Conv)*MAX_CONV_LEVELS);
    if(shapes_conv==NULL) {
        printf("Error Memory allocation");
        emergency();
        exit(1);
    }
    for (int i=0; i<MAX_CONV_LEVELS; i++) {
        switch(i) {
            case 0:
                shapes_conv[i].batch=1;
                shapes_conv[i].height=40;
                shapes_conv[i].width=40;
                shapes_conv[i].channels=1;
                break;
            case 1:
                shapes_conv[i].batch=1;
                shapes_conv[i].height=40;
                shapes_conv[i].width=40;
                shapes_conv[i].channels=8;
                break;
            case 2:
                shapes_conv[i].batch=1;
                shapes_conv[i].height=13;
                shapes_conv[i].width=13;
                shapes_conv[i].channels=8;
                break;
            case 3:
                shapes_conv[i].batch=1;
                shapes_conv[i].height=13;
                shapes_conv[i].width=13;
                shapes_conv[i].channels=16;
                break;
            case 4:
                shapes_conv[i].batch=1;
                shapes_conv[i].height=6;
                shapes_conv[i].width=6;
                shapes_conv[i].channels=16;
                break;
            case 5:
                shapes_conv[i].batch=1;
                shapes_conv[i].height=3;
                shapes_conv[i].width=3;
                shapes_conv[i].channels=32;
                break;
            case 6:
                shapes_conv[i].batch=1;
                shapes_conv[i].height=2;
                shapes_conv[i].width=2;
                shapes_conv[i].channels=64;
                break;
            case 7:
                shapes_conv[i].batch=1;
                shapes_conv[i].height=1;
                shapes_conv[i].width=1;
                shapes_conv[i].channels=128;
                break;
            default:
                printf("ERROR - CONV LEVELS\n");
                emergency();
                exit(1);
        }
    }
}

void adapt_shape_output_to_model(int num_neurons) {
    if(num_neurons>0) {
        shapes_dense[1].size=num_neurons;
    }
}

void allocate_shapes_dense() {
    shapes_dense=malloc(sizeof(Shape_Dense)*MAX_DENSE_LEVELS);
    if(shapes_dense==NULL) {
        printf("Error Memory allocation");
        emergency();
        exit(1);
    }
    for (int i=0; i<MAX_DENSE_LEVELS; i++) {
        switch(i) {
            case 0:
                shapes_dense[i].batch=1;
                shapes_dense[i].size=1600;
                break;
            case 1:
                shapes_dense[i].batch=1;
                shapes_dense[i].size=0;
                break;
            case 2:
                shapes_dense[i].batch=1;
                shapes_dense[i].size=128;
                break;
            case 3:
                shapes_dense[i].batch=1;
                shapes_dense[i].size=192;
                break;
            case 4:
                shapes_dense[i].batch=1;
                shapes_dense[i].size=240;
                break;
            case 5:
                shapes_dense[i].batch=1;
                shapes_dense[i].size=256;
                break;
            
            default:
                printf("ERROR - DENSE LEVELS\n");
                emergency();
                exit(1);
        }
    }
}

void allocate_intermediate_shapes_dense(Model_Dense* dense) {
    int flatten_size=-1;
    for(int i=0; i<dense->num_dense_layers; i++) {
        int output_size=compute_flatten_size_dense(dense->layers[i].output_shape);
        allocate_float_array(&(dense->layers[i].out_data), output_size);
        if(i==0) {
            int input_size=compute_flatten_size_dense(dense->layers[i].input_shape);
            allocate_float_array(&(dense->layers[i].in_data), input_size);
        }
        else {
            dense->layers[i].in_data=dense->layers[i-1].out_data;
        }
    }
}

void allocate_intermediate_shapes_conv(Model_Conv* conv) {
    int flatten_size=-1;
    for(int i=0; i<conv->num_conv_layers; i++) {
        int output_size=compute_flatten_size_conv(conv->layers[i].output_shape);
        allocate_float_array(&(conv->layers[i].out_data), output_size);
        if(i==0) {
            int input_size=compute_flatten_size_conv(conv->layers[i].input_shape);
            allocate_float_array(&(conv->layers[i].in_data), input_size);
        }
        else {
            conv->layers[i].in_data=conv->layers[i-1].out_data;
        }
    }
}

void allocate_float_array(float** array, int16_t size) {
    *array=malloc(sizeof(float)*size);
    if(*array==NULL) {
        printf("Error Memory allocation");
        emergency();
        exit(1);
    }
}

void deallocate_float_array(float* array) {
    if(array!=NULL) {
        free(array);
    }
}

int compute_flatten_size_conv(Shape_Conv conv) {
    return conv.batch*conv.width*conv.height*conv.channels;
}

int compute_flatten_size_dense(Shape_Dense dense) {
    return dense.batch*dense.size;
}

void deallocate_conv_model(Model_Conv* conv) {
    if(conv==NULL){
        return;
    }
    
    for (int i = 0; i<conv->num_conv_layers; ++i) {
        if(i==0) {
            deallocate_float_array(conv->layers[i].in_data);
        }
        deallocate_float_array(conv->layers[i].out_data);
    }
    if(conv->layers!=NULL) {
        free(conv->layers);
    }
    free(conv);
}

void deallocate_dense_model (Model_Dense* dense) {
    if(dense==NULL) {
        return;
    }
    for(int i=0; i<dense->num_dense_layers; ++i) {
        if(i==0) {
            deallocate_float_array(dense->layers[i].in_data);
        }
        deallocate_float_array(dense->layers[i].out_data);
    }
    if(dense->layers!=NULL) {
        free(dense->layers);
    }
    free(dense);
}

void emergency() {
    if(kws!=NULL) {
        deallocate_dense_model(kws);
        kws=NULL;
    }
    if(sv_128!=NULL) {
        deallocate_conv_model(sv_128);
        sv_128=NULL;
    }
    if(sv_256!=NULL) {
        deallocate_conv_model(sv_256);
        sv_256=NULL;
    }
    if(sv_dense_128_256) {
        deallocate_dense_model(sv_dense_128_256);
        sv_dense_128_256=NULL;
    }
    if(sv_dense_256_unbalanced!=NULL) {
        deallocate_dense_model(sv_dense_256_unbalanced);
        sv_dense_256_unbalanced=NULL;
    }
    if(sv_dense_256_192!=NULL) {
        deallocate_dense_model(sv_dense_256_192);
        sv_dense_256_192=NULL;
    }
    if(sv_dense_256_240!=NULL) {
        deallocate_dense_model(sv_dense_256_240);
        sv_dense_256_240=NULL;
    }
    if(sv_dense_256_256!=NULL) {
        deallocate_dense_model(sv_dense_256_256);
        sv_dense_256_256=NULL;
    }
    if(shapes_conv!=NULL) {
        free(shapes_conv);
        shapes_conv=NULL;
    }
    if(shapes_dense!=NULL) {
        free(shapes_dense);
        shapes_dense=NULL;
    }
}

void allocate_conv_model(Model_Conv** conv) {
    *conv=malloc(sizeof(Model_Conv));
    if(*conv==NULL) {
        printf("Error Memory allocation");
        emergency();
        exit(1);
    }
}

void allocate_dense_model(Model_Dense** dense) {
    *dense=malloc(sizeof(Model_Dense));
    if(*dense==NULL) {
        printf("Error Memory allocation");
        emergency();
        exit(1);
    }
}

void start_time_computing (struct timespec* start) {
    timespec_get(start, TIME_UTC);
}

void end_time_computing (struct timespec* start, struct timespec* end, long* nsecs) {
    timespec_get(end, TIME_UTC);
    double seconds=difftime(end->tv_sec, start->tv_sec);
    if(seconds<0) {
        seconds=0;
    }
    *nsecs=end->tv_nsec-start->tv_nsec;
    (*nsecs)/=1000;
}

char* mode_output() {
    switch(sv_elaborate) {
        case 0:
            return "BEST";
            break;
        case 1:
            return "MEAN";
            break;
        case 2:
            return "GEOM_MEDIAN";
            break;
        default:
            printf("ERROR");
            emergency();
            exit(1);
    }
}

char* decompose_neurons() {
    switch(dense_neurons_mode) {
        case 1: 
            return "256";
            break;
        case 2:
            return "U256-128";
            break;
        case 3:
            return "192";
            break;
        case 4: 
            return "240";
            break;
        case 5:
            return "256";
            break;
        default:
            printf("ERROR - NEURONS");
            emergency();
            exit(1);
    }
}

void save_results(long nsecs, float prob, int8_t result, float threshold, int model_size, Model_Types model) {
    model_size/=1024;
    switch(model) {
        case KWS:
            printf("\nMODEL: KWS\nMODEL_SIZE: %d KB\nTIME: %ld us\nPROB: %.6f\nTHRESHOLD: %.2f\nRESULT: %s\n\n", model_size, nsecs,  prob, threshold, ((result==0) ? "OK" : "NO"));
            break;
        case SV_CONV:
            printf("\nMODEL: SV_CONV\nMODEL_SIZE: %d KB\nDVECTOR: %d\nNEURONS: %s\nNUM_REFERENCES: %d\nMETHOD: %s\nTIME: %ld us\nPROB: %.6f\nTHRESHOLD: %.2f\nRESULT: %s\n\n", model_size,  dvector_model, decompose_neurons(), num_refs, mode_output(), nsecs, prob, threshold, ((result==0) ? "OK" : "NO"));
            break;
        case SV_DENSE:
            printf("\nMODEL: SV_DENSE\nMODEL_SIZE: %d KB\nDVECTOR: %d\nNEURONS: %s\nNUM_REFERENCES: %d\nMETHOD: %s\nTIME: %ld us\nPROB: %.6f\nTHRESHOLD: %.2f\nRESULT: %s\n\n", model_size,  dvector_model, decompose_neurons(), num_refs,mode_output(), nsecs, prob, threshold, ((result==0) ? "OK" : "NO"));
            break;
        case SVQ8:
            printf("\nMODEL: SVQ8\nMODEL_SIZE: %d KB\nDVECTOR: %d\nNEURONS: %s\nNUM_REFERENCES: %d\nMETHOD: %s\nTIME: %ld us\nPROB: %.6f\nTHRESHOLD: %.2f\nRESULT: %s\n\n", model_size, dvector_model, decompose_neurons(),num_refs, mode_output(), nsecs, prob, threshold, ((result==0) ? "OK" : "NO"));
            break;
        case SVQ4:
            printf("\nMODEL: SVQ4\nMODEL_SIZE: %d KB\nDVECTOR: %d\nNEURONS: %s\nNUM_REFERENCES: %d\nMETHOD: %s\nTIME: %ld us\nPROB: %.6f\nTHRESHOLD: %.2f\nRESULT: %s\n\n", model_size, dvector_model, decompose_neurons(),num_refs, mode_output(), nsecs, prob, threshold, ((result==0) ? "OK" : "NO"));
            break;
    }
}

int model_size_computation_conv(Model_Conv* model) {
    int tot_bytes=0;
    int single_value_size=sizeof(float);
    for(int i=0; i<model->num_conv_layers; i++) {
        if(model->layers[i].type_layer_conv==CONV2D) {
            int input_size=compute_flatten_size_conv(model->layers[i].input_shape);
            int output_size=compute_flatten_size_conv(model->layers[i].output_shape);

            //WEIGHTS+BIASES=INPUT*OUTPUT+OUTPUT
            tot_bytes+=(model->layers[i].kernel_size*model->layers[i].kernel_size*model->layers[i].input_shape.channels*model->layers[i].output_shape.channels*single_value_size); //WEIGHTS
            tot_bytes+=(model->layers[i].output_shape.channels*single_value_size); //BIASES
        }
    }
    return tot_bytes;
}
int model_size_computation_dense(Model_Dense* model) {
    int tot_bytes=0;
    int single_value_size=sizeof(float);
    for(int i=0; i<model->num_dense_layers; i++) {
        int input_size=compute_flatten_size_dense(model->layers[i].input_shape);
        int output_size=compute_flatten_size_dense(model->layers[i].output_shape);

        //WEIGHTS+BIASES=INPUT*OUTPUT+OUTPUT
        tot_bytes+=(input_size*output_size*single_value_size); //WEIGHTS
        tot_bytes+=(output_size*single_value_size); //BIASES
    }
    return tot_bytes;
}
