#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

int main() {
    // Create variables
    DifferentiableOperation* a = create_variable(2.0);
    DifferentiableOperation* b = create_variable(3.0);
    DifferentiableOperation* c = create_variable(4.0);

    // Build computation graph: f = (a + b) * c
    DifferentiableOperation* add_op = create_add_operation(a, b);
    DifferentiableOperation* mul_op = create_mul_operation(add_op, c);

    // Collect nodes with cycle detection
    if (!collect_nodes(mul_op)) {
        fprintf(stderr, "Cycle detected. Exiting.\n");
        return EXIT_FAILURE;
    }

    // Forward pass
    forward(mul_op);
    printf("Forward pass result: %f\n", mul_op->value);

    // Backward pass
    backward_pass();

    // Print gradients
    printf("Gradient w.r.t a: %f\n", a->grad);
    printf("Gradient w.r.t b: %f\n", b->grad);
    printf("Gradient w.r.t c: %f\n", c->grad);

    // Reset visitation states before freeing
    reset_visit_state(mul_op);

    // Free allocated memory
    free_operation(mul_op);

    return 0;
}