#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "differentiable_operation.h"
#include "operations.h"
#include "graph_utils.h"
#include "iris_data.h"

#define LEARNING_RATE 0.01
#define EPOCHS 1000
#define BATCH_SIZE 32

DifferentiableOperation* create_model() {
    printf("Creating model...\n");
    DifferentiableOperation* inputs[IRIS_FEATURES];
    DifferentiableOperation* weights[IRIS_FEATURES][IRIS_CLASSES];
    DifferentiableOperation* biases[IRIS_CLASSES];

    double xavier_init = sqrt(2.0 / (IRIS_FEATURES + IRIS_CLASSES));
    for (int i = 0; i < IRIS_FEATURES; i++) {
        inputs[i] = create_variable(0.0);
        printf("Created input %d: %p\n", i, (void*)inputs[i]);
        for (int j = 0; j < IRIS_CLASSES; j++) {
            weights[i][j] = create_variable(((double)rand() / RAND_MAX * 2 - 1) * xavier_init);
        }
    }

    for (int i = 0; i < IRIS_CLASSES; i++) {
        biases[i] = create_variable((double)rand() / RAND_MAX * 0.2 - 0.1);
    }

    DifferentiableOperation* z[IRIS_CLASSES];
    for (int i = 0; i < IRIS_CLASSES; i++) {
        z[i] = biases[i];
        for (int j = 0; j < IRIS_FEATURES; j++) {
            z[i] = create_add_operation(z[i], create_mul_operation(inputs[j], weights[j][i]));
        }
    }

    DifferentiableOperation* exp_z[IRIS_CLASSES];
    for (int i = 0; i < IRIS_CLASSES; i++) {
        exp_z[i] = create_exp_operation(z[i]);
    }

    DifferentiableOperation* softmax[IRIS_CLASSES];
    for (int i = 0; i < IRIS_CLASSES; i++) {
        softmax[i] = create_softmax_operation(exp_z, IRIS_CLASSES);
    }

    DifferentiableOperation* result = softmax[IRIS_CLASSES - 1];
    
    // Ensure the result has the correct inputs
    result->inputs = malloc(IRIS_FEATURES * sizeof(DifferentiableOperation*));
    result->num_inputs = IRIS_FEATURES;
    for (int i = 0; i < IRIS_FEATURES; i++) {
        result->inputs[i] = inputs[i];
    }

    printf("Model created successfully.\n");
    return result;
}

void reset_gradients(DifferentiableOperation* op) {
    if (op->visit_state == VISITED) return;
    op->visit_state = VISITED;

    op->grad = 0.0;

    for (int i = 0; i < op->num_inputs; i++) {
        reset_gradients(op->inputs[i]);
    }
}

void update_parameters(DifferentiableOperation* op, double learning_rate) {
    if (op->visit_state == VISITED) return;
    op->visit_state = VISITED;

    if (op->num_inputs == 0) {  // This is a variable (weight or bias)
        op->value -= learning_rate * op->grad;
    }

    for (int i = 0; i < op->num_inputs; i++) {
        update_parameters(op->inputs[i], learning_rate);
    }
}

int main() {
    printf("Starting program...\n");
    srand(time(NULL));

    printf("Creating model...\n");
    DifferentiableOperation* model = create_model();
    printf("Model created.\n");
    printf("Model address: %p\n", (void*)model);
    printf("Model inputs: %p\n", (void*)model->inputs);
    printf("Number of model inputs: %d\n", model->num_inputs);

    // Print information about each input
    for (int i = 0; i < model->num_inputs; i++) {
        printf("Input %d: %p\n", i, (void*)model->inputs[i]);
    }

    // Normalize features
    normalize_features();

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;
        int correct_predictions = 0;

        // Shuffle the dataset
        for (int i = 0; i < IRIS_SAMPLES; i++) {
            int j = i + rand() / (RAND_MAX / (IRIS_SAMPLES - i) + 1);
            IrisData temp = iris_dataset[j];
            iris_dataset[j] = iris_dataset[i];
            iris_dataset[i] = temp;
        }

        for (int batch_start = 0; batch_start < IRIS_SAMPLES; batch_start += BATCH_SIZE) {
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > IRIS_SAMPLES) batch_end = IRIS_SAMPLES;

            // Reset gradients
            reset_gradients(model);

            for (int i = batch_start; i < batch_end; i++) {
                // Forward pass
                for (int j = 0; j < IRIS_FEATURES; j++) {
                    model->inputs[j]->value = iris_dataset[i].features[j];
                }
                forward(model);

                // Compute loss and accuracy
                double loss = -log(model->inputs[iris_dataset[i].label]->value);
                total_loss += loss;

                int predicted_class = 0;
                double max_prob = -DBL_MAX;
                for (int j = 0; j < IRIS_CLASSES; j++) {
                    if (model->inputs[j]->value > max_prob) {
                        max_prob = model->inputs[j]->value;
                        predicted_class = j;
                    }
                }
                if (predicted_class == iris_dataset[i].label) {
                    correct_predictions++;
                }

                // Backward pass
                for (int j = 0; j < IRIS_CLASSES; j++) {
                    model->inputs[j]->grad += (j == iris_dataset[i].label) ? -1.0 / model->inputs[j]->value : 0;
                }
                backward_pass();
            }

            // Update parameters
            update_parameters(model, LEARNING_RATE * BATCH_SIZE / (batch_end - batch_start));
        }

        // Print epoch statistics
        if (epoch % 10 == 0 || epoch == EPOCHS - 1) {
            printf("Epoch %d: Loss = %f, Accuracy = %.2f%%\n", 
                   epoch, total_loss / IRIS_SAMPLES, 
                   100.0 * correct_predictions / IRIS_SAMPLES);
        }
    }

    // Test the model on the training set
    int correct_predictions = 0;
    for (int i = 0; i < IRIS_SAMPLES; i++) {
        // Set input values
        for (int j = 0; j < IRIS_FEATURES; j++) {
            model->inputs[j]->value = iris_dataset[i].features[j];
        }

        // Forward pass
        forward(model);

        // Predict class
        int predicted_class = 0;
        double max_prob = -DBL_MAX;
        for (int j = 0; j < IRIS_CLASSES; j++) {
            if (model->inputs[j]->value > max_prob) {
                max_prob = model->inputs[j]->value;
                predicted_class = j;
            }
        }
        if (predicted_class == iris_dataset[i].label) {
            correct_predictions++;
        }
    }

    printf("\nFinal test accuracy: %.2f%%\n", 100.0 * correct_predictions / IRIS_SAMPLES);

    // Generate DOT file for final model
    printf("Generating DOT file...\n");
    generate_dot_file(model, "iris_softmax_regression_graph.dot");
    printf("\nFinal model graph saved to iris_softmax_regression_graph.dot\n");

    // Free memory
    printf("Freeing memory...\n");
    reset_visit_state(model);
    free_operation(model);

    printf("Program completed successfully.\n");
    return 0;
}