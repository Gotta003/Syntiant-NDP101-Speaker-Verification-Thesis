#include "text_save.h"
#include "time.h"
#include "stdlib.h"

FileStruct file;

void populateStruct_conv(char* filename, Model_Conv* model_conv, int ntime, float prob, int result) {
    file.filename=filename;
    file.method=sv_elaborate;
    file.model_output_size=compute_flatten_size_conv(model_conv->layers[model_conv->num_conv_layers-1].output_shape);
    file.model_size=model_size_computation_conv(model_conv);
    file.neurons_dense=decompose_neurons();
    file.ntime=ntime;
    file.num_refs=model_conv->num_references;
    file.prob=prob;
    file.result=result;
    file.threshold=model_conv->threshold;
    file.type=model_conv->model_type;
}

void populateStruct_dense(char* filename, Model_Dense* model_dense, int ntime, float prob, int result) {
    file.filename=filename;  file.filename=filename;
    file.method=sv_elaborate;
    file.model_output_size=compute_flatten_size_dense(model_dense->layers[model_dense->num_dense_layers-1].output_shape);
    file.model_size=model_size_computation_dense(model_dense);
    file.ntime=ntime;
    file.neurons_dense=decompose_neurons();
    file.num_refs=model_dense->num_references;
    file.prob=prob;
    file.result=result;
    file.threshold=model_dense->threshold;
    file.type=model_dense->model_type;
}

void writeStruct() {
    FILE* to_file=fopen(file.filename, "a+");
    if(to_file==NULL) {
        printf("ERROR file opening");
        emergency();
        exit(1);
    }
    FILE* wc=NULL;
    char command[70];
    snprintf(command, sizeof(command), "wc -l < %s", file.filename);
    wc=popen(command, "r");
    if(wc==NULL) {
        printf("ERROR running wc -l command\n");
        fclose(to_file);
        emergency();
        exit(1);
    }
    int32_t lines=0;
    fscanf(wc, "%d", &lines);
    fclose(wc);
    if(lines==0) {
        lines=1;
    }
    fseek(to_file, 0, SEEK_END);
    long long file_size=ftell(to_file);
    if(file_size==0) {
        fprintf(to_file, "%-10s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s\n", "N°", "MODEL", "OUTPUT_SIZE", "NEURONS", "NUM_REFS", "MODEL_SIZE", "METHOD", "THRESHOLD", "PROBABILITY", "TIME(ns)", "RESULT");
    }
    char void_element='-';
    fprintf(to_file, "%-10d %-15d %-15d %-15s %-15d %-15d %-15s %-15.2lf %-15.6lf %-15d %-15s\n", lines, file.type, file.model_output_size, file.neurons_dense,file.num_refs, file.model_size,  mode_output(), file.threshold, file.prob, file.ntime, ((file.result==0) ? "OK" : "NO"));
    fclose(to_file);
    struct timespec ts;
    ts.tv_nsec=100000;
    ts.tv_sec=0;
    nanosleep(&ts, NULL);
}
