#include "dense_network.h"
#include "string.h"
#include "stdbool.h"

float relu(float x) {
    return x > 0 ? x : 0;
}

// z_j=∑_i(\omega_ij*x_i)+b_j
void fully_connected_layer(Layer_Dense* layer) {
    //int weight_size=input_size*output_size;
    //int biases_size=output_size;
    Shape_Dense input_shape=layer->input_shape;
    Shape_Dense output_shape=layer->output_shape;
    
    const float* input=layer->in_data;
    float* output=layer->out_data;
    const float* weights=layer->weights;
    const float* biases=layer->biases;
    int input_size=compute_flatten_size_dense(input_shape);
    int output_size=compute_flatten_size_dense(output_shape);
    
    for(int j=0; j<output_size; j++) {
        float z=0.0f;
        for(int i=0; i<input_size; i++) {
            z+=weights[j*input_size+i]*input[i];
        }
        z+=biases[j];
        output[j]=z;
        if(layer->activation==RELU || layer->activation==SOFTMAX) {
            output[j]=relu(output[j]);
        }
    }
    if(layer->activation==SOFTMAX) {
        softmax(output, output_size);
    }
}

void softmax(float input[], int size) {
   float sum=0.0f;
   for(int i=0; i<size; i++) {
        sum+=exp(input[i]);
   }
   for(int i=0; i<size; i++) {
        input[i]=exp(input[i])/sum;
   }
}

void elaborateResult(float output[], float* prob) {
    int printOrder[num_classes];
    for (int i=0; i<num_classes; i++) {
        printOrder[i]=i;
    }
    for(int i=0; i<num_classes-1; i++) {
        for(int j=0; j<num_classes-i-1; j++) {
            if(output[j+1]>output[j]) {
                float temp_output = output[j];
                output[j] = output[j+1];
                output[j+1] = temp_output;
                int temp=printOrder[j];
                printOrder[j]=printOrder[j+1];
                printOrder[j+1]=temp;
            }
        }
    }
    bool verify=false;
    for(int i=0; i<num_classes; i++) {
        //printf("%s=%f\n", class_names[printOrder[i]], output[i]);
        if(strcmp(class_names[printOrder[i]], "sheila")==0) {
            *prob=output[i];
            verify=true;
        }
    }
    if(!verify) {
        *prob=0.0;
    }
}

int dense_neural_network(Model_Dense* dense_model, const float* input, float* prob) {
    memcpy(dense_model->layers[0].in_data, input, sizeof(float)*compute_flatten_size_dense(dense_model->layers[0].input_shape));
    for(int i=0; i<dense_model->num_dense_layers; i++) {
        fully_connected_layer(&dense_model->layers[i]);
    }
    elaborateResult(dense_model->layers[dense_model->num_dense_layers-1].out_data, prob);
    return (*prob>dense_model->threshold) ? 0 : 1;
}

