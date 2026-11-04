#!/bin/bash

for num in 5 10 20 30 40 50 60 70 80 90 100 120 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850
do
    echo "$num"
    mkdir $num
    cd $num
    cp ../ref/*.py .
    python four.py --num_lines $num > log_$num 2>&1
    cd ..
done

