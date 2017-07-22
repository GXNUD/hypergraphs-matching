#!/bin/bash
FILES=./test/*
for image1 in $FILES; do
    for image2 in $FILES; do
        # echo "Processing $f file..."
        # take action on each file. $f store current file name
        # echo $image1 " with " $image2
        ./hyper.out $image1 $image2
    done
done
