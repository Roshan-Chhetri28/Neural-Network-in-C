#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_LINE_LENGTH 4096

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

// Box-Muller transform, scaled to the requested standard deviation
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

void free_NN(NeuralNetwork *nn) {
    if (!nn) return;
    for (int i = 0; i < nn->no_of_layers; i++) {
        Layer *l = &nn->layer[i];
        for (int j = 0; j < l->no_of_nodes; j++) free(l->weights[j]);
        free(l->weights);
        free(l->bias);
        free(l->outputs);
        free(l->deltas);
        free(l->inputs);
    }
    free(nn->layer);
    free(nn);
}

// Gradient descent (Stochastic Gradient Descent)
void SGD(NeuralNetwork *nn, double learning_rate, double y_true)
{
    // Sigmoid + binary cross-entropy collapse to (output - y) at the output layer
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

// Run the network over a dataset without updating weights.
void evaluate(NeuralNetwork *nn, double **x, double **y, int n,
              double *out_loss, double *out_accuracy) {
    double loss = 0.0;
    int correct = 0;
    for (int i = 0; i < n; i++) {
        double *output = forward_pass(nn, x[i]);
        double target = y[i][0];
        loss += binary_cross_entropy(target, output[0]);
        int prediction = (output[0] > 0.5) ? 1 : 0;
        if (prediction == (int)target) correct++;
    }
    *out_loss = loss / n;
    *out_accuracy = (double)correct / n * 100.0;
}

// Training function iterating over epochs and training examples
void train(NeuralNetwork *nn, int epochs, double lr,
           double **x_train, double **y_train, int num_examples,
           double **x_test, double **y_test, int num_test) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        int correct = 0;
        for (int i = 0; i < num_examples; i++) {
            double* output = forward_pass(nn, x_train[i]);
            double target = y_train[i][0];
            int prediction = (output[0] > 0.5) ? 1 : 0;
            epoch_loss += binary_cross_entropy(target, output[0]);
            if (prediction == (int)target) {
                correct++;
            }
            SGD(nn, lr, target);
        }
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            double accuracy = (double)correct / num_examples * 100;
            double test_loss, test_acc;
            evaluate(nn, x_test, y_test, num_test, &test_loss, &test_acc);
            printf("Epoch %4d: train loss = %.4f, train acc = %.2f%% | "
                   "test loss = %.4f, test acc = %.2f%%\n",
                   epoch, epoch_loss / num_examples, accuracy, test_loss, test_acc);
        }
    }
}

/*
 * Loads a numeric CSV into a row-major matrix.
 *
 * skip_header   : discard the first line (pandas writes a header row; parsing it
 *                 as data injects a garbage training example).
 * skip_first_col: discard column 0 (pandas writes an unnamed index column; as a
 *                 feature it runs 0..N and dominates every dot product, and as a
 *                 label it is not the label at all).
 */
double** load_csv(const char* filename, int* rows, int* cols,
                  int skip_header, int skip_first_col) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        return NULL;
    }

    char buffer[MAX_LINE_LENGTH];
    long data_start = 0;

    if (skip_header) {
        if (!fgets(buffer, sizeof(buffer), file)) {
            fclose(file);
            return NULL;
        }
        data_start = ftell(file);
    }

    // First pass: count rows, and take the column count from the first data row.
    int tmp_rows = 0;
    int tmp_cols = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        int current_cols = 0;
        for (char* t = strtok(buffer, ",\n\r"); t != NULL; t = strtok(NULL, ",\n\r")) {
            current_cols++;
        }
        if (current_cols == 0) continue;
        if (tmp_rows == 0) {
            tmp_cols = current_cols;
        } else if (current_cols != tmp_cols) {
            fclose(file);
            return NULL; // Column count mismatch
        }
        tmp_rows++;
    }

    int out_cols = skip_first_col ? tmp_cols - 1 : tmp_cols;
    if (tmp_rows == 0 || out_cols <= 0) {
        fclose(file);
        return NULL;
    }

    double** data = malloc(tmp_rows * sizeof(double*));
    if (!data) {
        fclose(file);
        return NULL;
    }
    for (int i = 0; i < tmp_rows; i++) {
        data[i] = malloc(out_cols * sizeof(double));
        if (!data[i]) {
            for (int j = 0; j < i; j++) free(data[j]);
            free(data);
            fclose(file);
            return NULL;
        }
    }

    // Second pass: parse values.
    fseek(file, data_start, SEEK_SET);
    int current_row = 0;
    while (fgets(buffer, sizeof(buffer), file) && current_row < tmp_rows) {
        int src_col = 0;
        int dst_col = 0;
        char* token = strtok(buffer, ",\n\r");
        if (token == NULL) continue;
        while (token != NULL && dst_col < out_cols) {
            if (skip_first_col && src_col == 0) {
                src_col++;
                token = strtok(NULL, ",\n\r");
                continue;
            }
            char* endptr;
            double value = strtod(token, &endptr);
            if (endptr == token) {
                for (int i = 0; i < tmp_rows; i++) free(data[i]);
                free(data);
                fclose(file);
                return NULL;
            }
            data[current_row][dst_col] = value;
            src_col++;
            dst_col++;
            token = strtok(NULL, ",\n\r");
        }
        if (dst_col != out_cols) {
            for (int i = 0; i < tmp_rows; i++) free(data[i]);
            free(data);
            fclose(file);
            return NULL;
        }
        current_row++;
    }
    fclose(file);

    *rows = tmp_rows;
    *cols = out_cols;
    return data;
}

void free_matrix(double** m, int rows) {
    if (!m) return;
    for (int i = 0; i < rows; i++) free(m[i]);
    free(m);
}

int main(void)
{
    srand(42); // fixed seed so runs are reproducible

    int x_rows, x_cols, y_rows, y_cols;
    int xt_rows, xt_cols, yt_rows, yt_cols;

    // Every file carries a header row and a leading index column; both are dropped.
    double** x_train = load_csv("x_train.csv", &x_rows, &x_cols, 1, 1);
    double** y_train = load_csv("y_train.csv", &y_rows, &y_cols, 1, 1);
    double** x_test  = load_csv("x_test.csv",  &xt_rows, &xt_cols, 1, 1);
    double** y_test  = load_csv("y_test.csv",  &yt_rows, &yt_cols, 1, 1);

    if (!x_train || !y_train || !x_test || !y_test) {
        printf("Error loading data.\n");
        return 1;
    }
    if (x_rows != y_rows || xt_rows != yt_rows) {
        printf("Feature/label row count mismatch.\n");
        return 1;
    }

    printf("Train: %d samples x %d features | Test: %d samples\n\n",
           x_rows, x_cols, xt_rows);

    int no_of_layers = 3;
    int no_of_nodes[] = {8, 4, 1};
    double learning_rate = 0.01;
    int epochs = 100;

    NeuralNetwork *nn = createNN(no_of_layers, no_of_nodes, x_cols);
    train(nn, epochs, learning_rate, x_train, y_train, x_rows, x_test, y_test, xt_rows);

    double test_loss, test_acc;
    evaluate(nn, x_test, y_test, xt_rows, &test_loss, &test_acc);
    printf("\nFinal test accuracy: %.2f%%\n", test_acc);

    free_NN(nn);
    free_matrix(x_train, x_rows);
    free_matrix(y_train, y_rows);
    free_matrix(x_test, xt_rows);
    free_matrix(y_test, yt_rows);
    return 0;
}
