#include "graph_utils.h"
#include "operations.h"  // Add this line
#include <string.h>

int visited_nodes = 0;
DifferentiableOperation* nodes[MAX_NODES];

int collect_nodes(DifferentiableOperation* op) {
    if (op->visit_state == VISITED) {
        return 1;
    }
    if (op->visit_state == VISITING) {
        fprintf(stderr, "Error: Cycle detected in computation graph involving node at address %p.\n", (void*)op);
        return 0;
    }

    op->visit_state = VISITING;

    for (int i = 0; i < op->num_inputs; ++i) {
        if (!collect_nodes(op->inputs[i])) {
            return 0;
        }
    }

    op->visit_state = VISITED;
    nodes[visited_nodes++] = op;

    return 1;
}

void forward(DifferentiableOperation* op) {
    if (op->compute) {
        for (int i = 0; i < op->num_inputs; ++i) {
            forward(op->inputs[i]);
        }
        op->compute(op);
    }
}

void backward_pass() {
    printf("Starting backward pass...\n");
    for (int i = visited_nodes - 1; i >= 0; i--) {
        DifferentiableOperation* op = nodes[i];
        printf("Processing node %d: %p\n", i, (void*)op);
        if (op->backward) {
            printf("Calling backward function for node %d\n", i);
            op->backward(op, op->grad);
        } else {
            printf("No backward function for node %d\n", i);
        }
    }
    printf("Backward pass completed.\n");
}

void generate_dot_file(DifferentiableOperation* root, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }

    fprintf(file, "digraph ComputationGraph {\n");

    reset_visit_state(root);

    DifferentiableOperation* stack[MAX_NODES];
    int stack_size = 0;
    stack[stack_size++] = root;

    while (stack_size > 0) {
        DifferentiableOperation* op = stack[--stack_size];
        
        if (op->visit_state == VISITED) continue;
        op->visit_state = VISITED;

        fprintf(file, "    node%p [label=\"", (void*)op);
        if (op->compute == add_compute) {
            fprintf(file, "+\\nvalue: %.2f\\ngrad: %.2f", op->value, op->grad);
        } else if (op->compute == mul_compute) {
            fprintf(file, "*\\nvalue: %.2f\\ngrad: %.2f", op->value, op->grad);
        } else if (op->compute == exp_compute) {
            fprintf(file, "exp\\nvalue: %.2f\\ngrad: %.2f", op->value, op->grad);
        } else if (op->compute == softmax_compute) {
            fprintf(file, "softmax\\nvalue: %.2f\\ngrad: %.2f", op->value, op->grad);
        } else {
            fprintf(file, "var\\nvalue: %.2f\\ngrad: %.2f", op->value, op->grad);
        }
        fprintf(file, "\"];\n");

        for (int i = 0; i < op->num_inputs; i++) {
            fprintf(file, "    node%p -> node%p;\n", (void*)op->inputs[i], (void*)op);
            if (op->inputs[i]->visit_state == UNVISITED) {
                stack[stack_size++] = op->inputs[i];
            }
        }
    }

    fprintf(file, "}\n");
    fclose(file);

    reset_visit_state(root);
}