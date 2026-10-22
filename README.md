# Deep Learning, Lab 1: Deep Neural Networks on Fashion-MNIST

Notebook: [`Lab_1_Deep_Neural_Networks.ipynb`](Lab_1_Deep_Neural_Networks.ipynb) (all outputs, figures and written conclusions are saved in it).

## What is inside

| Part | Content |
|---|---|
| 1 | L-layer neural network in pure NumPy: He/Xavier initialization, ReLU/Sigmoid/Softmax, forward and backward pass, cross-entropy. Verified with `assert` tests and a numerical gradient check |
| 2 | SGD with Momentum, RMSprop, Adam, L2 regularization, inverted Dropout (NumPy), tested by hand calculations, gradient checks and a training run |
| 3 | The same architecture in PyTorch (`nn.Linear`, `nn.BatchNorm1d`, `nn.ReLU`, `nn.Dropout`), SGD vs SGD+Momentum vs AdamW, convergence curves and analysis |
| 4 | Error analysis of 50 misclassified validation images, bias/variance diagnostics, five experiments, engineering summary and three action items |

Every conclusion in the notebook is placed after the cell that produces the numbers it relies on.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Lab_1_Deep_Neural_Networks.ipynb     # Kernel -> Restart & Run All (about 10 minutes on CPU)
```

The dataset is downloaded automatically into `./data` (not stored in the repository).
Random seeds are fixed (`SEED = 42`), two runs give identical numbers.

## Key results

| Model | Train error | Validation error |
|---|---|---|
| AdamW, 784-256-128-10, BatchNorm + Dropout 0.3, 10 epochs | 8.33 % | 11.35 % |
| Bigger network (512-256), 20 epochs | 5.09 % | 10.39 % |
