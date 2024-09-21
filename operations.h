#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "differentiable_operation.h"

// Expose compute functions
void add_compute(DifferentiableOperation* op);
void mul_compute(DifferentiableOperation* op);
void exp_compute(DifferentiableOperation* op);
void softmax_compute(DifferentiableOperation* op);

// Creation functions
DifferentiableOperation* create_add_operation(DifferentiableOperation* a, DifferentiableOperation* b);
DifferentiableOperation* create_mul_operation(DifferentiableOperation* a, DifferentiableOperation* b);
DifferentiableOperation* create_exp_operation(DifferentiableOperation* input);
DifferentiableOperation* create_softmax_operation(DifferentiableOperation** inputs, int num_inputs);

#endif