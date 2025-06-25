#include "sv_conv.h"

int floor_div(int a, int b) {
    return (a - ((a % b) + b) % b) / b;
}

float relu_sv(float x) {
    return (x>0) ? x : 0.0;
}

void batch_normalization(Model_Conv* conv, const float* input) {
    int size=compute_flatten_size_conv(conv->layers[0].input_shape);
    for(int i=0; i<size; i++) {
        conv->layers[0].in_data[i]=input[i]*conv->gamma+conv->beta;
    }
}

void conv2d(Layer_Conv* layer) {
    Shape_Conv* input_shape = &layer->input_shape;
    Shape_Conv* output_shape = &layer->output_shape;
    
    const float* input = layer->in_data;
    float* output = layer->out_data;
    int in_height = input_shape->height;
    int in_width = input_shape->width;
    int in_channels = input_shape->channels;
    int out_channels = output_shape->channels;
    int kernel_size = layer->kernel_size;
    int stride = layer->stride;
    Padding_Conv padding = layer->padding_conv;
    const float* weights = layer->weights;
    const float* biases = layer->biases;
    
    int out_height;
    int out_width;
    int pad_top=0;
    int pad_bottom=0;
    int pad_left=0;
    int pad_right=0;

    if(padding==SAME) {
        out_height = floor_div(in_height + stride - 1, stride);
        out_width = floor_div(in_width + stride - 1, stride);
        int pad_h=(out_height-1)*stride+kernel_size-in_height;
        int pad_w=(out_width-1)*stride+kernel_size-in_width;
        pad_h=pad_h>0 ? pad_h : 0;
        pad_w=pad_w>0 ? pad_w : 0;
        pad_top=floor_div(pad_h,2);
        pad_bottom=pad_h-pad_top;
        pad_left=floor_div(pad_w,2);
        pad_right=pad_w-pad_left;
    }
    else if(padding==VALID) {
        out_height=floor_div(in_height-kernel_size, stride)+1;
        out_width=floor_div(in_width-kernel_size, stride)+1;
    }
   
    int padded_height=in_height+pad_top+pad_bottom;
    int padded_width=in_width+pad_left+pad_right;
    float* padded_input=(float*)calloc(padded_height*padded_width*in_channels, sizeof(float));
    
    for(int h=0; h<in_height; h++) {
        for(int w=0; w<in_width; w++) {
            for(int c=0; c<in_channels; c++) {
                int padded_h=h+pad_top;
                int padded_w=w+pad_left;
                const int in_idx = (h * in_width + w) * in_channels + c;
                const int pad_idx = (padded_h * padded_width + padded_w) * in_channels + c;
                padded_input[pad_idx] = input[in_idx];
            }
        }
    }
    
    for(int i=0; i<out_height; i++) {
        for(int j=0; j<out_width; j++) {
            for(int oc=0; oc<out_channels; oc++) {
                float sum=biases[oc];
                int h_s=i*stride;
                int w_s=j*stride;
                for(int kh=0; kh<kernel_size; kh++) {
                    for(int kw=0; kw<kernel_size; kw++) {
                        int h=h_s+kh;
                        int w=w_s+kw;
                        if(h<padded_height && w<padded_width) {
                            for(int ic=0; ic<in_channels; ic++) {
                                int input_idx=(h*padded_width+w)*in_channels+ic;
                                int weight_idx=((oc*kernel_size+kh)*kernel_size+kw)*in_channels+ic;
                                sum+=padded_input[input_idx]*weights[weight_idx];
                            }
                        }
                    }
                }
                if(layer->activation==RELU) {
                    output[(i*out_width+j)*out_channels+oc]=relu_sv(sum);
                }
            }
        }
    }
    free(padded_input);
}

void max_pool2d(Layer_Conv* layer) {
    Shape_Conv* input_shape = &layer->input_shape;
    Shape_Conv* output_shape = &layer->output_shape;
    
    const float* input = layer->in_data;
    float* output = layer->out_data;
    int in_height = input_shape->height;
    int in_width = input_shape->width;
    int channels = input_shape->channels;
    int pool_size = layer->kernel_size;
    int stride = layer->stride;
    Padding_Conv padding = layer->padding_conv;
    
    int out_height;
    int out_width;
    int pad_top=0;
    int pad_bottom=0;
    int pad_left=0;
    int pad_right=0;
    if (padding==VALID) {
        out_height = floor_div(in_height - pool_size, stride)+1;
        out_width = floor_div(in_width - pool_size, stride)+1;
    }
    else if (padding==SAME) {
        out_height = floor_div(in_height+stride-1, stride);
        out_width = floor_div(in_width+stride-1, stride);
        
        int pad_needed_height = (out_height - 1) * stride + pool_size - in_height;
        int pad_needed_width = (out_width - 1) * stride + pool_size - in_width;
        
        pad_top = floor_div(pad_needed_height, 2);
        pad_bottom = pad_needed_height - pad_top;
        pad_left = floor_div(pad_needed_width, 2);
        pad_right = pad_needed_width - pad_left;
    }

    for (int i=0; i<channels*out_height*out_width; i++) {
        output[i]=0.0f;
    }
    for(int c=0; c<channels; c++) {
        for(int h=0; h<out_height; h++) {
            for(int w=0; w<out_width; w++) {
                float max_val=-INFINITY;
                int h_s=h*stride-pad_top;
                int w_s=w*stride-pad_left;
                for(int kh=0; kh<pool_size; kh++) {
                    for(int kw=0; kw<pool_size; kw++) {
                        int h_in=h_s+kh;
                        int w_in=w_s+kw;
                        if(h_in>=0 && h_in<in_height && w_in>=0 && w_in<in_width) {
                            int input_idx=((h_in * in_width + w_in) * channels) + c;
                            float val=input[input_idx];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                }
                output[(h*out_width+w)*channels+c]=max_val;
            }
        }
    }
}

float cosine_similarity(const float* vec1, const float* vec2, int size) {
    float dot_product=0.0f;
    float norm_vec1=0.0f;
    float norm_vec2=0.0f;
    for(int i=0; i<size; i++) {
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

float bestmatching(const float* input_vector, const float d_vectors[][MAX_DVECTOR_SIZE], int num_vectors, int d_vector_size) {
    float max_similarity=-1.0f;
    for(int i=0; i<num_vectors; i++) {
        float similarity=cosine_similarity(input_vector, d_vectors[i], d_vector_size);
        if(similarity>max_similarity) {
            max_similarity=similarity;
        }
    }
    return max_similarity;
}

void mean_d_vector(const float d_vectors[][MAX_DVECTOR_SIZE], float* mean_output, int num_vectors, int d_vector_size) {
    for(int i=0; i<d_vector_size; i++) {
        float sum=0.0f;
        for(int j=0; j<num_vectors; j++) {
            sum+=d_vectors[j][i];
        }
        mean_output[i]=sum/num_vectors;
    }
}

float mean_cos(const float* input_vector, const float d_vectors[][MAX_DVECTOR_SIZE], int num_vectors, int dvector_size) {
    float mean_d_vect[MAX_DVECTOR_SIZE];
    mean_d_vector(d_vectors, mean_d_vect, num_vectors, dvector_size);
    return cosine_similarity(input_vector, mean_d_vect, dvector_size);
}

float distance(const float* a, const float* b, int dim) {
    float sum=0.0f;
    for(int i=0; i<dim; i++) {
        float diff=a[i]-b[i];
        sum+=diff*diff;
    }
    return sqrtf(sum);
}

void geometrical_median_computation(const float d_vectors[][MAX_DVECTOR_SIZE], float* geom_median_output, int num_vectors, int d_vector_size) {
    mean_d_vector(d_vectors, geom_median_output, num_vectors, d_vector_size);
    float temp[d_vector_size];
    for(int iter=0; iter<MAX_ITER; iter++) {
        float w_sum=0.0f;
        for(int j=0; j<d_vector_size; j++) {
            temp[j]=0.0f;
        }
        for(int i=0; i<num_vectors; i++) {
            float d=distance(geom_median_output, d_vectors[i], d_vector_size);
            if(d<EPSILON) {
                continue;
            }

            float w=1.0f/d;
            for(int j=0; j<d_vector_size; j++) {
                temp[j]+=w*d_vectors[i][j];
            }
            w_sum+=w;
        }
        if(w_sum==0.0f) {
            break;
        }

        for(int j=0; j<d_vector_size; j++) {
            temp[j]/=w_sum;
        }
        float shift=distance(temp, geom_median_output, d_vector_size);
        for(int j=0; j<d_vector_size; j++) {
            geom_median_output[j]=temp[j];
        }
        if(shift<EPSILON) {
            break;
        }
    }
}

float geometrical_median_vector(const float* input_vector, const float d_vectors[][MAX_DVECTOR_SIZE], int num_vectors, int dvector_size) {
    float geom_median_vector[MAX_DVECTOR_SIZE];
    geometrical_median_computation(d_vectors, geom_median_vector, num_vectors, dvector_size);
    return cosine_similarity(input_vector, geom_median_vector, dvector_size);
}

int8_t sv_conv_neural_network(Model_Conv* conv_model, const float mfe_input[], float* prob) {
    if(conv_model->batch_norm) {
        batch_normalization(conv_model, mfe_input);
    }
    else {
        memcpy(conv_model->layers[0].in_data, mfe_input,
              compute_flatten_size_conv(conv_model->layers[0].input_shape)*sizeof(float));
    }

    for(int i=0; i<conv_model->num_conv_layers; i++) {
        switch (conv_model->layers[i].type_layer_conv) {  // Changed from type_layer_conv to layer_type
            case CONV2D:
                conv2d(&conv_model->layers[i]);  // Added address-of operator
                break;
            case MAXPOOL2D:
                max_pool2d(&conv_model->layers[i]);  // Added address-of operator
                break;
            default:
                printf("INITIALIZATION ERROR, INVALID LAYER TYPE");
                emergency();
                exit(1);
        }
    }
    
    float input_vectors[MAX_DVECTOR_SIZE];
    int output_size = compute_flatten_size_conv(conv_model->layers[conv_model->num_conv_layers-1].output_shape);
    memcpy(input_vectors, conv_model->layers[conv_model->num_conv_layers-1].out_data,
          sizeof(float) * output_size);
    
    if(sv_elaborate==0) {
        *prob=bestmatching(input_vectors, conv_model->dvectors,
                                   conv_model->num_references, conv_model->d_vector_size);
    }
    else if(sv_elaborate==1) {
        *prob=mean_cos(input_vectors, conv_model->dvectors,
                               conv_model->num_references, conv_model->d_vector_size);
    }
    else if(sv_elaborate==2) {
        *prob=geometrical_median_vector(input_vectors, conv_model->dvectors, conv_model->num_references, conv_model->d_vector_size);
    }
    else {
        printf("ERROR - SV_Elaborate somehow modified, not 0 or 1\n");
        emergency();
        exit(1);
    }
    return (*prob>conv_model->threshold ? 0 : 1);
}
