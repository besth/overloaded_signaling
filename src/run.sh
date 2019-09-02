#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$pwd

goal_indices=(0 1)
worlds=("hands-free" "hands-tied")

{
for i in "${goal_indices[@]}"
do
    for j in "${worlds[@]}"
    do
        echo Testing with goal: ${i}, world: ${j}
        sleep 2
        python 2D_jp/main.py --goal-ind ${i} --world ${j} &
    done
done
}