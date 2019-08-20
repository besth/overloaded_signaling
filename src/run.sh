#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$pwd

goal_indices=(0 1 2)
env_types=(0 1)

{
for i in "${goal_indices[@]}"
do
    for j in "${env_types[@]}"
    do
        echo Testing with goal: ${i}, env-type: ${j}
        sleep 2
        python 2D_jp/main.py --goal-ind ${i} --env-type ${j} &
    done
done
}