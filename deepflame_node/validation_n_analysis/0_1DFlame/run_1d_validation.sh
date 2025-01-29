#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi

mkdir -p results

for num in "$@"
do
    for dir in ${num}_1DFlame_*
    do
        if [ -d "$dir" ]; then
            if [[ -f "inference.py" && -f "DNN_model.pt" ]]; then
                cp "inference.py" "$dir"
		cp DNN_model_NH3H2_interpolate.pt $dir/DNN_model.pt
            else
                echo "inference.py or DNN_model.pt missing."
            fi

            cd "$dir"
            echo "Entered $dir"

            python switchTorchOn.py
            ./Allrun
            # cp log.flameSpeed "../results/${dir}_log.flameSpeed"
            # ./Allclean

            cd ..
            echo "Exited $dir"
        else
            echo "Directory $dir does not exist"
        fi
    done
done
