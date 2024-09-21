#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

typedef struct DifferentiableOperation DifferentiableOperation;
typedef enum { UNVISITED, VISITING, VISITED } VisitState;

struct DifferentiableOperation {
    // Function pointers to methods
    void (*compute)(DifferentiableOperation*);
    void (*backward)(DifferentiableOperation*, double grad);

    // Data members
    double value;  // Output value of the operation
    double grad;   // Gradient with respect to the output

    // Pointers to inputs (operands)
    DifferentiableOperation** inputs;
    int num_inputs;

    // Visitation state for cycle detection
    VisitState visit_state;
};

// Variable creation
DifferentiableOperation* create_variable(double value) {
    DifferentiableOperation* var = malloc(sizeof(DifferentiableOperation));
    var->value = value;
    var->grad = 0.0;

    // Variables have no inputs
    var->inputs = NULL;
    var->num_inputs = 0;

    // No compute or backward functions needed
    var->compute = NULL;
    var->backward = NULL;

    var->visit_state = UNVISITED;

    return var;
}

// Addition operation
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

// Multiplication operation
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

// Node collection with cycle detection
#define MAX_NODES 100

int visited_nodes = 0;
DifferentiableOperation* nodes[MAX_NODES];

int collect_nodes(DifferentiableOperation* op) {
    if (op->visit_state == VISITED) {
        return 1; // Node already processed
    }
    if (op->visit_state == VISITING) {
        fprintf(stderr, "Error: Cycle detected in computation graph involving node at address %p.\n", (void*)op);
        return 0;
    }

    op->visit_state = VISITING;

    // Process inputs recursively
    for (int i = 0; i < op->num_inputs; ++i) {
        if (!collect_nodes(op->inputs[i])) {
            // Cycle detected in a child
            return 0;
        }
    }

    op->visit_state = VISITED;
    nodes[visited_nodes++] = op;

    return 1; // Successful collection
}

// Forward pass
void forward(DifferentiableOperation* op) {
    if (op->compute) {
        // Compute inputs first
        for (int i = 0; i < op->num_inputs; ++i) {
            forward(op->inputs[i]);
        }
        // Then compute this operation
        op->compute(op);
    }
    // Variables have no compute function
}

// Backward pass
void backward_pass() {
    // Initialize gradients to zero
    for (int i = 0; i < visited_nodes; ++i) {
        nodes[i]->grad = 0.0;
    }

    // Set gradient of the output node
    DifferentiableOperation* output_node = nodes[visited_nodes - 1];
    output_node->grad = 1.0;  // ∂output/∂output = 1

    // Process nodes in reverse order
    for (int i = visited_nodes - 1; i >= 0; --i) {
        DifferentiableOperation* op = nodes[i];
        if (op->backward) {
            op->backward(op, op->grad);
        }
    }
}

// Reset visitation states
void reset_visit_state(DifferentiableOperation* op) {
    if (op->visit_state == UNVISITED) {
        return;
    }

    op->visit_state = UNVISITED;
    for (int i = 0; i < op->num_inputs; ++i) {
        reset_visit_state(op->inputs[i]);
    }
}

// Free operations
void free_operation(DifferentiableOperation* op) {
    if (op->visit_state == UNVISITED) {
        return;
    }
    op->visit_state = UNVISITED;

    // Free inputs recursively
    for (int i = 0; i < op->num_inputs; ++i) {
        free_operation(op->inputs[i]);
    }

    // Free the inputs array
    if (op->inputs) {
        free(op->inputs);
    }
    // Free the operation itself
    free(op);
}

// Add this function after the existing functions and before main()

void generate_dot_file(DifferentiableOperation* root, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }

    fprintf(file, "digraph ComputationGraph {\n");

    // Reset visit states
    reset_visit_state(root);

    // Use a stack to perform depth-first traversal
    DifferentiableOperation* stack[MAX_NODES];
    int stack_size = 0;
    stack[stack_size++] = root;

    while (stack_size > 0) {
        DifferentiableOperation* op = stack[--stack_size];
        
        if (op->visit_state == VISITED) continue;
        op->visit_state = VISITED;

        // Generate node
        fprintf(file, "    node%p [label=\"", (void*)op);
        if (op->compute == add_compute) {
            fprintf(file, "+\\nvalue: %.2f\\ngrad: %.2f", op->value, op->grad);
        } else if (op->compute == mul_compute) {
            fprintf(file, "*\\nvalue: %.2f\\ngrad: %.2f", op->value, op->grad);
        } else {
            fprintf(file, "var\\nvalue: %.2f\\ngrad: %.2f", op->value, op->grad);
        }
        fprintf(file, "\"];\n");

        // Generate edges
        for (int i = 0; i < op->num_inputs; i++) {
            fprintf(file, "    node%p -> node%p;\n", (void*)op->inputs[i], (void*)op);
            if (op->inputs[i]->visit_state == UNVISITED) {
                stack[stack_size++] = op->inputs[i];
            }
        }
    }

    fprintf(file, "}\n");
    fclose(file);

    // Reset visit states again
    reset_visit_state(root);
}

// Add these new operations
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

// In main(), replace the existing code with this softmax regression implementation:
int main() {
    // Input features
    DifferentiableOperation* x1 = create_variable(0.5);
    DifferentiableOperation* x2 = create_variable(-1.0);
    DifferentiableOperation* x3 = create_variable(0.2);

    // Weights (3 features x 4 classes)
    DifferentiableOperation* w[3][4];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            w[i][j] = create_variable((double)rand() / RAND_MAX * 0.2 - 0.1);
        }
    }

    // Biases (4 classes)
    DifferentiableOperation* b[4];
    for (int i = 0; i < 4; i++) {
        b[i] = create_variable((double)rand() / RAND_MAX * 0.2 - 0.1);
    }

    // Compute linear combinations
    DifferentiableOperation* z[4];
    for (int i = 0; i < 4; i++) {
        z[i] = create_add_operation(
            create_add_operation(
                create_add_operation(
                    create_mul_operation(x1, w[0][i]),
                    create_mul_operation(x2, w[1][i])
                ),
                create_mul_operation(x3, w[2][i])
            ),
            b[i]
        );
    }

    // Apply exp to each z
    DifferentiableOperation* exp_z[4];
    for (int i = 0; i < 4; i++) {
        exp_z[i] = create_exp_operation(z[i]);
    }

    // Compute softmax
    DifferentiableOperation* softmax[4];
    for (int i = 0; i < 4; i++) {
        softmax[i] = create_softmax_operation(exp_z, 4);
    }

    // Collect nodes and perform forward pass
    if (!collect_nodes(softmax[3])) {
        fprintf(stderr, "Cycle detected. Exiting.\n");
        return EXIT_FAILURE;
    }

    forward(softmax[3]);

    // Print softmax outputs
    printf("Softmax outputs:\n");
    for (int i = 0; i < 4; i++) {
        printf("Class %d: %f\n", i, softmax[i]->value);
    }

    // Backward pass (assuming class 2 is the correct label)
    for (int i = 0; i < 4; i++) {
        softmax[i]->grad = (i == 2) ? -1.0 / softmax[i]->value : 0;
    }
    backward_pass();

    // Print gradients of weights and biases
    printf("\nGradients:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("dL/dw[%d][%d] = %f\n", i, j, w[i][j]->grad);
        }
    }
    for (int i = 0; i < 4; i++) {
        printf("dL/db[%d] = %f\n", i, b[i]->grad);
    }

    // Generate DOT file
    generate_dot_file(softmax[3], "softmax_regression_graph.dot");
    printf("\nSoftmax regression graph saved to softmax_regression_graph.dot\n");

    // Free memory (you'll need to free all created operations)
    reset_visit_state(softmax[3]);
    free_operation(softmax[3]);

    return 0;
}