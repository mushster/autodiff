#ifndef DIFFERENTIABLE_OPERATION_H
#define DIFFERENTIABLE_OPERATION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct DifferentiableOperation DifferentiableOperation;
typedef enum { UNVISITED, VISITING, VISITED } VisitState;

struct DifferentiableOperation {
    void (*compute)(DifferentiableOperation*);
    void (*backward)(DifferentiableOperation*, double grad);
    double value;
    double grad;
    DifferentiableOperation** inputs;
    int num_inputs;
    VisitState visit_state;
};

DifferentiableOperation* create_variable(double value);
void free_operation(DifferentiableOperation* op);
void reset_visit_state(DifferentiableOperation* op);

#endif