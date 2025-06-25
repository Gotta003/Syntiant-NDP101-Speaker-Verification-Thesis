#include "sv_dense_network.h"

int sv_dense_neural_network(Model_Dense* dense_model, const float* input, float* prob) {
    memcpy(dense_model->layers[0].in_data, input, sizeof(float)*compute_flatten_size_dense(dense_model->layers[0].input_shape));
    for(int i=0; i<dense_model->num_dense_layers; i++) {
        fully_connected_layer(&dense_model->layers[i]);
    }

    float input_vectors[MAX_DVECTOR_SIZE];
    int output_size = compute_flatten_size_dense(dense_model->layers[dense_model->num_dense_layers-1].output_shape);
    memcpy(input_vectors, dense_model->layers[dense_model->num_dense_layers-1].out_data,
          sizeof(float) * output_size);
    
    if(sv_elaborate==0) {
        *prob=bestmatching(input_vectors, dense_model->dvectors, dense_model->num_references, dense_model->d_vector_size);
    }
    else if(sv_elaborate==1) {
        *prob=mean_cos(input_vectors, dense_model->dvectors, dense_model->num_references, dense_model->d_vector_size);
    }
    else if(sv_elaborate==2) {
        *prob=geometrical_median_vector(input_vectors, dense_model->dvectors, dense_model->num_references, dense_model->d_vector_size);
    }
    else {
        printf("ERROR - SV_Elaborate somehow modified, not 0 or 1\n");
        emergency();
        exit(1);
    }
    return (*prob>dense_model->threshold ? 0 : 1);
}
