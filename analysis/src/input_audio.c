#include "input_audio.h"
#include "audio_samples.h"
#include "sv_conv.h"
#include "kws/kws.h"
#include "svconv/sv128/sv128.h"
#include "svconv/sv256/sv256.h"
#include "svdense/sv128256/sv128256.h"
#include "svdense/sv256192/sv256192.h"
#include "svdense/sv256240/sv256240.h"
#include "svdense/sv256u/sv256u.h"
#include "svdense/sv256256/sv256256.h"
#include "svq8/pqt/sv_dense_int8.h"

static int audio_callback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {
    short* audio=(short*)inputBuffer;
    float spectrogram[NUM_FRAMES(framesPerBuffer)*FILTER_NUMBER];
    framesPerBuffer*=(1.0-AUDIO_WINDOW);
    for(int i=0; i<framesPerBuffer; i++) {
        printf("%d\t", audio[i]);
    }
    audio_processing(audio, framesPerBuffer);
    return paContinue;
}

void channel_setup() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        printf("PortAudio init failed: %s\n", Pa_GetErrorText(err));
        exit(1);
    }
    printf("PortAudio initialized successfully\n");
}

PaStreamParameters parameters_setup() {
    PaStreamParameters inputParams;
    inputParams.device = Pa_GetDefaultInputDevice();
    inputParams.channelCount = NUM_CHANNELS;
    inputParams.sampleFormat = paInt16;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = NULL;
    return inputParams;
}

void identify_device(PaStreamParameters inputParams) {
    const PaDeviceInfo *deviceInfo = Pa_GetDeviceInfo(inputParams.device);
    if (deviceInfo != NULL) {
        printf("Using input device: %s\n", deviceInfo->name);
        printf("Sample rate: %d\n", SAMPLE_RATE);
        printf("Max input channels: %d\n", deviceInfo->maxInputChannels);
        printf("Default sample rate: %f\n", deviceInfo->defaultSampleRate);
    } else {
        printf("No input device found\n");
        Pa_Terminate();
        exit(1);
    }
}

void setup_stream(PaStream** stream, PaStreamParameters inputParams) {
    PaError err = Pa_OpenStream(
        stream,
        &inputParams,
        NULL,
        SAMPLE_RATE,
        NUM_SAMPLES,
        paClipOff,
        audio_callback,
        NULL
    );
    if (err != paNoError) {
        printf("Failed to open stream: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        exit(1);
    }
    printf("PortAudio stream setup correctly\n");
}

void start_stream(PaStream* stream) {
    PaError err = Pa_StartStream(stream);
    if (err != paNoError) {
        printf("Failed to start stream: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        Pa_Terminate();
        exit(1);
    }
    printf("PortAudio stream started\n");
}

void stop_stream(PaStream* stream) {
    PaError err = Pa_StopStream(stream);
    if (err != paNoError) {
        printf("Error Stopping Stream: %s", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        Pa_Terminate();
        exit(1);
    }
    printf("STOP RECORDING...\n");
}

void close_stream(PaStream* stream) {
    PaError err = Pa_CloseStream(stream);
    if (err != paNoError) {
        printf("Error Closing Stream: %s", Pa_GetErrorText(err));
        Pa_Terminate();
        exit(1);
    }
    printf("PortAudio stream closing correctly\n");
}

void terminate_portaudio() {
    Pa_Terminate();
}

void live_sampling() {
    channel_setup();
    PaStream* stream=NULL;
    PaStreamParameters inputParams=parameters_setup();
    identify_device(inputParams);
    setup_stream(&stream, inputParams);
    start_stream(stream);
    //while(1) {
    Pa_Sleep((int)(DURATION_SECONDS*1024));
    //}
    stop_stream(stream);
    close_stream(stream);
    terminate_portaudio();
}
void model_processing(const float* input) {
    switch(dvector_model) {
        case 128:
            if(bypass_sv==0) {
                processing_sv_128_model(input);
            }
            if(dense_neurons_mode==1) {
                processing_sv128256_model(input);
            }
            break;
        case 256:
            if(bypass_sv==0) {
                processing_sv_256_model(input);
            }
            switch(dense_neurons_mode) {
                case 2: //unbalanced
                    processing_sv256u_model(input);
                    break;
                case 3: //192
                    processing_sv256192_model(input);
                    break;
                case 4: //240
                    processing_sv256240_model(input);
                    break;
                case 5: //not fit 256
                    processing_sv256256_model(input);
                    break;
                default:
                    printf("ERROR - Submode is not support. Distilled Model doesn't exist\n");
                    emergency();
                    exit(1);
            }
            break;
        default:
            printf("ERROR - DVector Model doesn't exist\n");
            emergency();
            exit(1);
    }
}

void audio_processing(short* audio_buffer, int framesNumber) {
    float output[1600];
    struct timespec start, end;
    float prob;
    compute_spectrogram(audio_buffer, output, framesNumber);
    int8_t result=processing_kws_model(output);
    if(result==0) {
        model_processing(output);
    }
    else {
        printf("\n\nNOT SHEILA WORD RECOGNIZED\n\n");
    }
    //sv_dense_int8_neural_network(output, sv_elaborate);
}

void process_wav_file(const char* filename) {
    int framesPerBuffer = SAMPLE_RATE * (DURATION_SECONDS - AUDIO_WINDOW);
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening WAV file\n");
        exit(1);
    }
    short* audio_buffer=(short*)malloc(sizeof(short)*framesPerBuffer);
    // Read and print WAV header (first 44 bytes)
    uint8_t header[44];
    fread(header, 1, sizeof(header), file);

    uint16_t audio_format = *(uint16_t*)(header + 20);
    uint16_t bits_per_sample = *(uint16_t*)(header + 34);
    uint16_t num_channels = *(uint16_t*)(header + 22);

    for (int i = 0; i < framesPerBuffer; i++) {
        int16_t bytes[4]; // Enough for 32-bit samples (stereo 16-bit or mono 32-bit)
        size_t bytes_read = fread(bytes, 1, num_channels * (bits_per_sample / 8), file);
        if (bytes_read != num_channels * (bits_per_sample / 8)) {
            printf("End of file reached\n");
            break;
        }
        audio_buffer[i]=*(short*)bytes;
        //printf("%d\t", audio_buffer[i]);
    }
    //printf("\n\n");
    fclose(file);
    allocate_shapes_conv();
    allocate_shapes_dense();
    audio_processing(audio_buffer, framesPerBuffer);
    free(audio_buffer);
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        printf("Usage:\n");
        printf("  For live sampling: %s 0\n", argv[0]);
        printf("  For file processing: %s 1 <audio_file.wav>\n", argv[0]);
        return 1;
    }
    mode = atoi(argv[1]);
    if (mode == 0) {
        // Live sampling mode
        if (argc != 3) {
            printf("Live sampling mode requires exactly 1 argument\n");
            return 1;
        }
        sv_elaborate=atoi(argv[2]);
        if(sv_elaborate<0 || sv_elaborate>2) {
            printf("ERROR - None sv elaboration\n");
            return 1;
        }
        live_sampling();
    }
    else if (mode == 1) {
        // File processing mode
        if (argc != 9) {
            printf("File processing mode requires exactly 5 arguments (mode, filename, d-vector model (128 or 256), dense neurons configuration (128->1, 256->2-3-4-5\n");
            return 1;
        }
        dvector_model=atoi(argv[3]);
        dense_neurons_mode=atoi(argv[4]);
        sv_elaborate=atoi(argv[5]);
        num_refs=atoi(argv[6]);
        bypass_kws=atoi(argv[7]);
        bypass_sv=atoi(argv[8]);
        if(dvector_model!=128 && dvector_model!=256) {
            printf("ERROR - dvector model has to be 128 or 256 - value inserted %d\n", dvector_model);
            return 1;
        }
        if(dense_neurons_mode==1) {
            if(dvector_model!=128) {
                printf("Model 256 doesn't support mode 1\n");
                return 1;
            }
        }
        else {
            if(dense_neurons_mode>=2 && dense_neurons_mode<=5) {
                if(dvector_model!=256) {
                    printf("Model 128 doesn't support modes from 2 to 5\n");
                    return 1;
                }
            }
            else {
                printf("ERROR - Mode not existing. Follow this: dense neurons configuration (128->1, 256->2-3-4-5\n");
            }
        }
        if(sv_elaborate<0 || sv_elaborate>2) {
            printf("ERROR - None sv elaboration\n");
            return 1;
        }
        const char* filename = argv[2];
        const char* extension = strrchr(filename, '.');
        
        if (!extension || strcmp(extension, ".wav") != 0) {
            printf("Unsupported file format. Please use .wav\n");
            return 1;
        }
        
        process_wav_file(filename);
    }
    /*else if (mode==2) {
        if (argc != 3) {
            printf("Live sampling mode requires exactly 1 argument\n");
            return 1;
        }
        sv_elaborate=atoi(argv[2]);
        if(sv_elaborate!=0 && sv_elaborate!=1) {
            printf("ERROR - None sv elaboration\n");
            return 1;
        }
        int framesPerBuffer = SAMPLE_RATE * (DURATION_SECONDS - AUDIO_WINDOW);
        //audio_processing(audio_sample, framesPerBuffer);
        int result=kws_neural_network(spectrogram_sample);
        //if(strcmp(class_names[result], "sheila")==0) {
        if(result==0) {
            if(sv_neural_network(spectrogram_sample, sv_elaborate)==0) {
                printf("\n\nHELLO MATTEO - CONV\n\n");
            }
            else {
                printf("\n\nUSER NOT ENROLLED - CONV\n\n");
            }
            if(sv_dense_neural_network(spectrogram_sample, sv_elaborate)==0) {
                printf("\n\nHELLO MATTEO - DENSE\n\n");
            }
            else {
                printf("\n\nUSER NOT ENROLLED - DENSE\n\n");
            }
            if(sv_dense_int8_neural_network(spectrogram_sample, sv_elaborate)==0) {
                printf("\n\nHELLO MATTEO - INT8 DENSE\n\n");
            }
            else {
                printf("\n\nUSER NOT ENROLLED - INT8 DENSE\n\n");
            }
            if(sv_dense_int8_alt_neural_network(output, sv_elaborate)==0) {
                printf("\n\nHELLO MATTEO - INT8 DENSE ALT\n\n");
            }
            else {
                printf("\n\nUSER NOT ENROLLED - INT8 DENSE ALT\n\n");
            }
            if(sv_dense_int4_neural_network(spectrogram_sample, sv_elaborate)==0) {
                printf("\n\nHELLO MATTEO - INT4 DENSE\n\n");
            }
            else {
                printf("\n\nUSER NOT ENROLLED - INT4 DENSE\n\n");
            }
        }
        else {
            printf("\n\nNOT SHEILA WORD RECOGNIZED\n\n");
        }
    }*/
    else {
        printf("Invalid mode. Use 0 for live sampling or 1 for file processing\n");
        return 1;
    }
    emergency();
    return 0;
}
