#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_ROWS 1000
#define MAX_COLS 100
#define MAX_LINE_LENGTH 1024



typedef struct
{
    int no_of_nodes;
    int no_inputs;
    double *bias;
    double *inputs;
    double **weights;
    double *outputs;
    double *deltas;
} Layer;

typedef struct
{
    int no_of_layers;
    Layer *layer;
} NeuralNetwork;

// Activation Functions
double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

double relu(double z)
{
    return (z > 0) ? z : 0;
}

double relu_deriv(double z)
{
    return (z > 0) ? 1 : 0;
}

// Loss function with clipping to avoid log(0)
double binary_cross_entropy(double y, double y_hat)
{
    double epsilon = 1e-8;
    y_hat = fmax(epsilon, fmin(y_hat, 1 - epsilon));
    return -(y * log(y_hat) + (1 - y) * log(1 - y_hat));
}

// Modified to take std_dev as parameter
double rand_normal(double std_dev) {
    static int hasSpare = 0;
    static double spare;
    if (hasSpare) {
        hasSpare = 0;
        return spare * std_dev;
    }
    double u, v, s;
    do {
        u = (rand() / (RAND_MAX + 1.0)) * 2.0 - 1.0;
        v = (rand() / (RAND_MAX + 1.0)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    hasSpare = 1;
    return std_dev * (u * s);
}

void create_NN_layer(Layer* layer, int no_of_nodes, int is_output) {
    layer->no_of_nodes = no_of_nodes;
    layer->bias = malloc(no_of_nodes * sizeof(double));
    layer->weights = malloc(no_of_nodes * sizeof(double*));
    layer->outputs = malloc(no_of_nodes * sizeof(double));
    layer->deltas = malloc(no_of_nodes * sizeof(double));
    layer->inputs = malloc(layer->no_inputs * sizeof(double));
    
    for (int i = 0; i < no_of_nodes; i++) {
        layer->weights[i] = malloc(layer->no_inputs * sizeof(double));
        layer->bias[i] = 0.0;
        double std_dev;
        if (is_output) {
            // Xavier initialization for output layer (sigmoid)
            std_dev = sqrt(1.0 / layer->no_inputs);
        } else {
            // He initialization for hidden layers (ReLU)
            std_dev = sqrt(2.0 / layer->no_inputs);
        }
        for (int j = 0; j < layer->no_inputs; j++) {
            layer->weights[i][j] = rand_normal(std_dev);
        }
    }
    // Initialize input array to zero
    for (int i = 0; i < layer->no_inputs; i++) {
        layer->inputs[i] = 0.0;
    }
}

NeuralNetwork* createNN(int no_of_layers, int *no_of_nodes, int input_size)
{
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->no_of_layers = no_of_layers;
    nn->layer = malloc(no_of_layers * sizeof(Layer));
    for (int i = 0; i < nn->no_of_layers; i++) {
        if (i == 0) {
            nn->layer[i].no_inputs = input_size; 
        } else {
            nn->layer[i].no_inputs = nn->layer[i - 1].no_of_nodes;
        }
        int is_output = (i == nn->no_of_layers - 1);
        create_NN_layer(&nn->layer[i], no_of_nodes[i], is_output);
    }
    return nn;
}


// Gradient descent (Stochastic Gradient Descent)
void SGD(NeuralNetwork *nn, double learning_rate, double y_true)
{
    // Derivative for output layer (assuming a single output neuron)
    Layer *output_layer = &nn->layer[nn->no_of_layers - 1];
    double output = output_layer->outputs[0];
    output_layer->deltas[0] = (output - y_true);
    
    // Backpropagation for hidden layers
    for (int i = nn->no_of_layers - 2; i >= 0; i--) {
        Layer *current = &nn->layer[i];
        Layer *next = &nn->layer[i + 1];
        for (int j = 0; j < current->no_of_nodes; j++) {
            double error = 0.0;
            for (int k = 0; k < next->no_of_nodes; k++) {
                // FIX: Use next->deltas[k] (not next->deltas[j]) to accumulate error.
                error += next->weights[k][j] * next->deltas[k];
            }
            current->deltas[j] = error * relu_deriv(current->outputs[j]);
        }
    }
    
    // Update weights and biases for all layers.
    for (int i = 0; i < nn->no_of_layers; i++) {
        Layer *layer = &nn->layer[i];
        double *layer_input;
        if (i == 0) {
            layer_input = layer->inputs;
        } else {
            layer_input = nn->layer[i - 1].outputs;
        }
        for (int j = 0; j < layer->no_of_nodes; j++) {
            layer->bias[j] -= learning_rate * layer->deltas[j];
            for (int k = 0; k < layer->no_inputs; k++) {
                // FIX: Use layer_input[k] (not layer_input[j]) when updating weights.
                layer->weights[j][k] -= learning_rate * layer->deltas[j] * layer_input[k];
            }
        }
    }
}

// Forward pass for one training example
double* forward_pass(NeuralNetwork* nn, double* input) {
    // Set the input for the first layer.
    memcpy(nn->layer[0].inputs, input, nn->layer[0].no_inputs * sizeof(double));
    double* current_input = input;
    for (int i = 0; i < nn->no_of_layers; i++) {
        Layer* current = &nn->layer[i];
        for (int j = 0; j < current->no_of_nodes; j++) {
            double sum = current->bias[j];
            // FIX: Sum over all inputs (using correct indices) for the weighted sum.
            for (int k = 0; k < current->no_inputs; k++) {
                sum += current->weights[j][k] * current_input[k];
            }
            if (i < nn->no_of_layers - 1)
                current->outputs[j] = relu(sum);
            else
                current->outputs[j] = sigmoid(sum);
        }
        current_input = current->outputs;
    }
    return nn->layer[nn->no_of_layers - 1].outputs;
}

// Training function iterating over epochs and training examples
void train(NeuralNetwork *nn, int epochs, double lr, double** x_train, double** y_train, int num_examples, int input_dim) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        int correct = 0;
        for (int i = 0; i < num_examples; i++) {
            double* output = forward_pass(nn, x_train[i]);
            int prediction = (output[0] > 0.5) ? 1 : 0;
            epoch_loss += binary_cross_entropy(y_train[i][0], output[0]);
            if (prediction == (int)y_train[i][0]) {
                correct++;
            }
            SGD(nn, lr, y_train[i][0]);
        }
        double accuracy = (double)correct / num_examples * 100;
        printf("Epoch %d: Loss = %.4f, Accuracy = %.2f%%\n", epoch, epoch_loss / num_examples, accuracy);
    }
}

double** load_csv(const char* filename, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        return NULL;
    }
    *rows = 0;
    *cols = 0;
    char buffer[1024];
    int tmp_rows = 0;
    int tmp_cols = 0;
    int is_first_valid_line = 1;
    
    while (fgets(buffer, sizeof(buffer), file)) {
        int current_cols = 0;
        char* token;
        token = strtok(buffer, ",\n");
        while (token != NULL) {
            current_cols++;
            token = strtok(NULL, ",\n");
        }
        if (current_cols == 0) continue;
        if (is_first_valid_line) {
            tmp_cols = current_cols;
            tmp_rows = 1;
            is_first_valid_line = 0;
        } else {
            if (current_cols != tmp_cols) {
                fclose(file);
                return NULL; // Column count mismatch
            }
            tmp_rows++;
        }
    }
    if (tmp_rows == 0 || tmp_cols == 0) {
        fclose(file);
        return NULL;
    }
    
    double** data = malloc(tmp_rows * sizeof(double*));
    if (!data) {
        fclose(file);
        return NULL;
    }
    for (int i = 0; i < tmp_rows; i++) {
        data[i] = malloc(tmp_cols * sizeof(double));
        if (!data[i]) {
            for (int j = 0; j < i; j++)
                free(data[j]);
            free(data);
            fclose(file);
            return NULL;
        }
    }
    
    fseek(file, 0, SEEK_SET);
    int current_row = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        char* token;
        int current_col = 0;
        token = strtok(buffer, ",\n");
        if (token == NULL) continue;
        while (token != NULL && current_col < tmp_cols) {
            char* endptr;
            double value = strtod(token, &endptr);
            if (endptr == token) {
                for (int i = 0; i < tmp_rows; i++)
                    free(data[i]);
                free(data);
                fclose(file);
                return NULL;
            }
            data[current_row][current_col] = value;
            current_col++;
            token = strtok(NULL, ",\n");
        }
        if (current_col != tmp_cols) {
            for (int i = 0; i < tmp_rows; i++)
                free(data[i]);
            free(data);
            fclose(file);
            return NULL;
        }
        current_row++;
    }
    fclose(file);
    if (current_row != tmp_rows) {
        for (int i = 0; i < tmp_rows; i++)
            free(data[i]);
        free(data);
        return NULL;
    }
    *rows = tmp_rows;
    *cols = tmp_cols;
    return data;
}

int main()
{
    NeuralNetwork *nn;
    int no_of_layers = 3;
    .
    int no_of_nodes[] = {2, 3, 1};
    int x_rows, x_cols;
    int y_rows, y_cols;
    double** x_train = load_csv("x_train.csv", &x_rows, &x_cols);
    double** y_train = load_csv("y_train.csv", &y_rows, &y_cols);
    if (x_train == NULL || y_train == NULL) {
        printf("Error loading training data.\n");
        return 1;
    }
    double learning_rate = 0.0001;
    int epochs = 1000;
    
    nn = createNN(no_of_layers, no_of_nodes, x_cols);
    
    train(nn, epochs, learning_rate, x_train, y_train, x_rows, x_cols);
    return 0;
}
