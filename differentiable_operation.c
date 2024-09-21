#include "differentiable_operation.h"

DifferentiableOperation* create_variable(double value) {
    DifferentiableOperation* var = malloc(sizeof(DifferentiableOperation));
    var->value = value;
    var->grad = 0.0;
    var->inputs = NULL;
    var->num_inputs = 0;
    var->compute = NULL;
    var->backward = NULL;
    var->visit_state = UNVISITED;
    return var;
}

void free_operation(DifferentiableOperation* op) {
    if (op->visit_state == UNVISITED) {
        return;
    }
    op->visit_state = UNVISITED;
    for (int i = 0; i < op->num_inputs; ++i) {
        free_operation(op->inputs[i]);
    }
    if (op->inputs) {
        free(op->inputs);
    }
    free(op);
}

void reset_visit_state(DifferentiableOperation* op) {
    if (op->visit_state == UNVISITED) {
        return;
    }
    op->visit_state = UNVISITED;
    for (int i = 0; i < op->num_inputs; ++i) {
        reset_visit_state(op->inputs[i]);
    }
}