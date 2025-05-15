#!/bin/bash

for k in 5 20 50
do
    python downstream_task_evaluation.py k_shot=$k
done