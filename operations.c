#include "operations.h"
#include <string.h>

void add_compute(DifferentiableOperation* op) {
    op->value = op->inputs[0]->value + op->inputs[1]->value;
}

void add_backward(DifferentiableOperation* op, double grad) {
    op->inputs[0]->grad += grad;
    op->inputs[1]->grad += grad;
}

DifferentiableOperation* create_add_operation(DifferentiableOperation* a, DifferentiableOperation* b) {
    DifferentiableOperation* op = malloc(sizeof(DifferentiableOperation));
    op->inputs = malloc(2 * sizeof(DifferentiableOperation*));
    op->inputs[0] = a;
    op->inputs[1] = b;
    op->num_inputs = 2;
    op->compute = add_compute;
    op->backward = add_backward;
    op->visit_state = UNVISITED;
    return op;
}

void mul_compute(DifferentiableOperation* op) {
    op->value = op->inputs[0]->value * op->inputs[1]->value;
}

void mul_backward(DifferentiableOperation* op, double grad) {
    op->inputs[0]->grad += grad * op->inputs[1]->value;
    op->inputs[1]->grad += grad * op->inputs[0]->value;
}

DifferentiableOperation* create_mul_operation(DifferentiableOperation* a, DifferentiableOperation* b) {
    DifferentiableOperation* op = malloc(sizeof(DifferentiableOperation));
    op->inputs = malloc(2 * sizeof(DifferentiableOperation*));
    op->inputs[0] = a;
    op->inputs[1] = b;
    op->num_inputs = 2;
    op->compute = mul_compute;
    op->backward = mul_backward;
    op->visit_state = UNVISITED;
    return op;
}

void exp_compute(DifferentiableOperation* op) {
    op->value = exp(op->inputs[0]->value);
}

void exp_backward(DifferentiableOperation* op, double grad) {
    op->inputs[0]->grad += grad * op->value;
}

DifferentiableOperation* create_exp_operation(DifferentiableOperation* input) {
    DifferentiableOperation* op = malloc(sizeof(DifferentiableOperation));
    op->inputs = malloc(sizeof(DifferentiableOperation*));
    op->inputs[0] = input;
    op->num_inputs = 1;
    op->compute = exp_compute;
    op->backward = exp_backward;
    op->visit_state = UNVISITED;
    return op;
}

void softmax_compute(DifferentiableOperation* op) {
    double sum = 0.0;
    for (int i = 0; i < op->num_inputs; i++) {
        sum += op->inputs[i]->value;
    }
    op->value = op->inputs[0]->value / sum;
}

void softmax_backward(DifferentiableOperation* op, double grad) {
    double softmax = op->value;
    op->inputs[0]->grad += grad * softmax * (1 - softmax);
    for (int i = 1; i < op->num_inputs; i++) {
        op->inputs[i]->grad += grad * (-softmax * op->inputs[i]->value / op->inputs[0]->value);
    }
}

DifferentiableOperation* create_softmax_operation(DifferentiableOperation** inputs, int num_inputs) {
    DifferentiableOperation* op = malloc(sizeof(DifferentiableOperation));
    op->inputs = malloc(num_inputs * sizeof(DifferentiableOperation*));
    memcpy(op->inputs, inputs, num_inputs * sizeof(DifferentiableOperation*));
    op->num_inputs = num_inputs;
    op->compute = softmax_compute;
    op->backward = softmax_backward;
    op->visit_state = UNVISITED;
    return op;
}