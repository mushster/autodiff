#ifndef IRIS_DATA_H
#define IRIS_DATA_H

#define IRIS_FEATURES 4
#define IRIS_CLASSES 3
#define IRIS_SAMPLES 150

typedef struct {
    double features[IRIS_FEATURES];
    int label;
} IrisData;

extern IrisData iris_dataset[IRIS_SAMPLES];

#endif