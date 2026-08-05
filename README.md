# Neural Network in C

A feedforward neural network written from scratch in C, with no ML libraries. Forward and backward
propagation, stochastic gradient descent, ReLU and sigmoid activations, Xavier/He initialisation and
binary cross-entropy are all implemented directly, including the underlying loops.

I built this to understand the mechanics rather than call `model.fit()`. The whole thing is a single
file, `Nn.c`, in about 350 lines.

## Build and run

```bash
gcc -Wall -O2 Nn.c -o nn -lm
./nn
```

No dependencies beyond libc and libm.

## Architecture

| Component        | Implementation                                                        |
| ---------------- | --------------------------------------------------------------------- |
| Layers           | Fully connected, configurable via `no_of_nodes[]` in `main()`          |
| Hidden activation| ReLU                                                                   |
| Output activation| Sigmoid (single output neuron, binary classification)                  |
| Loss             | Binary cross-entropy, clipped at 1e-8 to avoid `log(0)`                |
| Initialisation   | He for hidden layers, Xavier for the output layer                      |
| Optimiser        | Stochastic gradient descent, one example at a time                     |
| Data loading     | CSV parser with header and index-column handling                       |

The default configuration is a 3-layer network (8 → 4 → 1) trained for 100 epochs at a learning rate
of 0.01, with a fixed RNG seed so runs are reproducible.

## Implementation notes

**The output-layer gradient is collapsed.** Rather than chaining the derivative of binary
cross-entropy with the derivative of sigmoid, the two cancel analytically:

$$\delta^{(L)} = a^{(L)} - y$$

which is what `SGD()` computes directly. This is both simpler and numerically better behaved than
applying the two derivatives separately.

**ReLU's derivative is evaluated on the post-activation output.** `relu_deriv(a)` rather than
`relu_deriv(z)`. This is valid specifically for ReLU, since `relu(z) > 0` exactly when `z > 0`, so
the two agree everywhere except at `z = 0`. It would not be valid for sigmoid or tanh.

**Xavier vs He are assigned per layer type.** He (variance `2/n_in`) for the ReLU hidden layers,
Xavier (variance `1/n_in`) for the sigmoid output layer. Normal samples come from a Box-Muller
transform in `rand_normal()`.

## A bug worth documenting

The network originally would not converge. Loss sat flat or ran away to NaN within a few epochs, and
the workaround at the time was to drop the learning rate to 0.0001, which masked the symptom without
fixing anything.

The cause was not in the network. It was in the data loading:

1. **The labels were wrong.** `y_train.csv` is written by pandas with an unnamed index column, so
   column 0 is the row index and column 1 is the actual label. The training loop was reading column
   0 — meaning the network was being asked to output the row index (0 to 711) through a sigmoid,
   scored against binary cross-entropy.
2. **The index column was also a feature.** `x_train.csv` carries the same index column. It ranges
   from 0 to 711 while every other feature sits roughly within [-2, 9], so it dominated every dot
   product into the first layer and produced saturating activations and exploding gradients.
3. **The header row was being trained on.** The CSV parser had no header handling, so the header line
   parsed cleanly as numbers and became training example #1.

The lesson generalises: the bug was in the part of the pipeline I never questioned, not in the
backprop I'd written myself and was least confident about. `load_csv()` now takes explicit
`skip_header` and `skip_first_col` flags rather than assuming the file is clean.

## Data

The dataset is a binary classification problem with 712 training and 179 test samples, 12 features
after dropping the index column. Features are pre-standardised, with several one-hot encoded columns.

The majority-class baseline is **62.4%** — any accuracy at or below that means the network is not
learning.

| File            | Contents                     |
| --------------- | ---------------------------- |
| `x_train.csv`   | Training features            |
| `y_train.csv`   | Training labels              |
| `x_test.csv`    | Test features                |
| `y_test.csv`    | Test labels                  |

Every file has a header row and a leading index column; `load_csv()` strips both.

## Results

Default configuration (8 → 4 → 1, lr 0.01, 100 epochs, seed 42):

```
Train: 712 samples x 12 features | Test: 179 samples

Epoch    0: train loss = 0.6823, train acc = 59.55% | test loss = 0.6860, test acc = 58.66%
Epoch   10: train loss = 0.4504, train acc = 81.74% | test loss = 0.4594, test acc = 81.56%
Epoch   20: train loss = 0.4144, train acc = 81.74% | test loss = 0.4468, test acc = 82.68%
Epoch   50: train loss = 0.3955, train acc = 82.87% | test loss = 0.4508, test acc = 81.01%
Epoch   99: train loss = 0.3877, train acc = 83.15% | test loss = 0.4519, test acc = 81.01%

Final test accuracy: 81.01%
```

**81.01% test accuracy against a 62.4% majority-class baseline.**

Worth noting: test loss bottoms out around epoch 20 at 0.4468 and then drifts upward while training
loss keeps falling. That is textbook overfitting, and with early stopping around epoch 20 the network
reaches 82.68%. The gap is small because the network is small — there isn't much capacity to overfit
with. Adding regularisation or early stopping would be the next thing to do.

## On performance

An equivalent network in TensorFlow trains roughly 2x slower on this problem, but that comparison
deserves a caveat. At this scale — a handful of neurons and a few hundred samples — the difference is
almost entirely framework overhead: graph construction and per-step Python dispatch, not arithmetic.
The matrix code here is plain nested loops with no blocking, vectorisation or BLAS. On any real
workload TensorFlow wins comfortably.

## Possible extensions

- Mini-batch gradient descent instead of per-sample SGD
- Additional activations (Leaky ReLU, softmax) and multi-class output
- Learning rate scheduling
- Gradient checking against numerical gradients as a built-in test

## References

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), Michael Nielsen

## Author

Roshan Chhetri — [github.com/Roshan-Chhetri28](https://github.com/Roshan-Chhetri28)

MIT License.
