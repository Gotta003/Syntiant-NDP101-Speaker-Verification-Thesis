
#!/bin/bash
echo "Flashing to NDP101..."

if [ -z "$1" ]; then
    FILENAME="./wav_files/sheila_0.wav"
else
    FILENAME="$1"
fi

if [ "$2" -eq "0" ]; then
    gcc src/*.c -O0 -o ndp_model -I./include -I/opt/homebrew/include -L/opt/homebrew/lib -lportaudio -lfftw3  -lm
fi

MODE=1
DVECTORSIZE=(256)
DVECTORSIZE=(128 256)
DENSENEURONS=(1 2 3 4 5)
OPS=(0 1 2)
REFS=(1 8 16 64)
python3 ./scripts/jsonReader.py
count=0
bypass_kws=0
bypass_sv=0

function all_modes {
    for ref in "${REFS[@]}"; do
        for op in "${OPS[@]}"; do
            for dvector in "${DVECTORSIZE[@]}"; do
                for dn in "${DENSENEURONS[@]}"; do
                    case $dn in
                        1)
                            if [ "$dvector" -eq 128 ]; then
                                ./ndp_model $MODE "$FILENAME" $dvector $dn $op $ref $bypass_kws $bypass_sv 
                            fi
                            ;;
                        2|3|4|5)
                            if [ "$dvector" -eq 256 ]; then
                                ./ndp_model $MODE "$FILENAME" $dvector $dn $op $ref $bypass_kws $bypass_sv
                                bypass_sv=1
                            fi
                            ;;
                    esac
                    bypass_kws=1
                done
                bypass_sv=0
            done
        done
    done
    bypass_kws=0
}

all_modes

#************** MONO ACTIVATION ***********
#./ndp_model $MODE "$FILENAME" 128 1 0 16 0

#./ndp_model 0 0
#NDP_MODEL
#1. Mode -> 0 - live-sampling, 1 - .wav file specific modes 2. - .wav file all modes 3. values in header
#2. Filename
#3. Dvector model 128 - Dvector model 256
#4. Dense Neurons (128 -> 1 256 -> 2-3-4-5)
#5. 0 Bestmatching - 1 Mean cos Computation

#********TO DRAW SPECTROGRAM ********
#python3 ./scripts/spectrogram_draw.py
