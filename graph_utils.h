#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include "differentiable_operation.h"

#define MAX_NODES 100

int collect_nodes(DifferentiableOperation* op);
void forward(DifferentiableOperation* op);
void backward_pass();
void generate_dot_file(DifferentiableOperation* root, const char* filename);

#endif