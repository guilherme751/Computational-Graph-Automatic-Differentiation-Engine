# ⚙️ Computational Graph & Automatic Differentiation Engine

A from-scratch implementation of automatic differentiation using computational graphs, similar to PyTorch's autograd system. This project builds the foundation of modern deep learning frameworks by implementing forward and backward propagation through dynamic computation graphs.

> **Note:** This was a course assignment for a university discipline. The original assignment guidelines and documentation are in Portuguese, which is why some comments in the notebook may be in Portuguese.

## 📋 Project Overview

This project implements a minimal but functional automatic differentiation (AD) engine that:
- Tracks operations in a dynamic computational graph
- Supports forward propagation for various mathematical operations
- Automatically computes gradients via reverse-mode differentiation (backpropagation)
- Enables gradient-based optimization (e.g., gradient descent)

**In essence, this is a simplified implementation of what PyTorch does under the hood.**

## 🎯 Objectives

- Implement a `Tensor` class to represent multi-dimensional arrays with gradient tracking
- Build a computational graph that records operations between tensors
- Implement automatic differentiation (backpropagation) through the graph
- Support common neural network operations and activation functions
- Use the AD system to solve optimization problems via gradient descent

## 🛠️ Core Components

### 1. Tensor Class
The fundamental data structure that:
- Stores numerical data as NumPy arrays
- Maintains references to parent tensors (computation graph)
- Accumulates gradients during backpropagation
- Provides a `.backward()` method to trigger gradient computation

### 2. Operation Interface (`Op`)
Abstract base class for all operations that:
- Defines `__call__()` for forward computation
- Defines `grad()` for gradient computation
- Automatically tracks parents in the computational graph

### 3. Implemented Operations

**Arithmetic Operations:**
- `Add` - Element-wise addition
- `Sub` - Element-wise subtraction  
- `Prod` - Element-wise multiplication
- `MatMul` - Matrix multiplication

**Activation Functions:**
- `ReLU` - Rectified Linear Unit
- `Sigmoid` - Logistic sigmoid
- `Tanh` - Hyperbolic tangent
- `Softmax` - Softmax normalization

**Mathematical Functions:**
- `Sin` - Element-wise sine
- `Cos` - Element-wise cosine
- `Exp` - Element-wise exponential
- `Square` - Element-wise square

**Reduction Operations:**
- `Sum` - Sum all elements
- `Mean` - Average of all elements

### 4. NameManager
A utility class that provides intuitive naming for tensors created by operations, making the computational graph easier to debug and visualize.

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy matplotlib seaborn
```

### Running the Project

1. **Open the notebook:**
   ```bash
   jupyter notebook especificacao-v1-resposta.ipynb
   ```

2. **Run all cells sequentially** - The notebook will:
   - Define the `Tensor` class and operation classes
   - Run unit tests for each operation
   - Demonstrate gradient descent optimization
   - Show examples of training simple functions

## 📚 How It Works

### Forward Pass
```python
# Create tensors
x = Tensor([1.0, 2.0, 3.0])
y = Tensor([4.0, 5.0, 6.0])

# Operations automatically build the graph
z = add(x, y)           # z = x + y
w = prod(z, 2.0)        # w = z * 2
loss = mean(w)          # scalar loss
```

### Backward Pass
```python
# Compute gradients automatically
loss.backward()

# Access gradients
print(x.grad)  # dLoss/dx
print(y.grad)  # dLoss/dy
```

### Training Example
```python
def objective_fn(inputs):
    x = inputs[0]
    return sin(add(prod(x, 2), 0.5))

x = Tensor(3.5)
history = gradient_descent(objective_fn, n_epochs=10, lr=0.2, inputs=[x])
```

## 🧮 Key Concepts Implemented

### 1. **Computational Graph**
- Each tensor maintains pointers to its parent tensors
- Operations create new tensors linked to their inputs
- Forms a Directed Acyclic Graph (DAG) of computations

### 2. **Reverse-Mode Automatic Differentiation**
- Starts from the output (loss) with gradient = 1
- Recursively applies chain rule backwards through the graph
- Each operation knows how to compute gradients w.r.t. its inputs

### 3. **Dynamic Graph Construction**
- Graph is built on-the-fly during forward pass
- Supports Python control flow (if/while statements)
- Similar to PyTorch's eager execution mode

## 📊 Operation Gradient Formulas

Each operation implements these gradient rules:

| Operation | Forward | Gradient (dL/dx) |
|-----------|---------|------------------|
| Add(x, y) | x + y | 1 · dL/dz |
| Sub(x, y) | x - y | 1 · dL/dz (for x), -1 · dL/dz (for y) |
| Prod(x, y) | x * y | y · dL/dz (for x), x · dL/dz (for y) |
| MatMul(A, B) | A @ B | dL/dC @ B^T (for A), A^T @ dL/dC (for B) |
| Sin(x) | sin(x) | cos(x) · dL/dz |
| Square(x) | x² | 2x · dL/dz |
| ReLU(x) | max(0, x) | (x > 0) · dL/dz |
| Sigmoid(x) | 1/(1+e^-x) | σ(x)(1-σ(x)) · dL/dz |

## 🎓 Assignment Details

**Course Assignment Specifications:**
- Delivery Date: November 28, 2025
- Points: 10 points
- Individual work
- Submission via Testr system

**Requirements Met:**
- ✅ Implemented `Tensor` class with gradient tracking
- ✅ Implemented all required operations with forward/backward passes
- ✅ Proper shape validation with assertions
- ✅ Dynamic computational graph construction
- ✅ Backpropagation through arbitrary computation graphs
- ✅ Gradient descent optimization implementation

## 🧪 Testing

The notebook includes comprehensive unit tests for each operation:

```python
# Example: Testing Add operation
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])
c = add(a, b)
c.backward()

print(a.grad)  # Expected: [[1.], [1.], [1.]]
print(b.grad)  # Expected: [[1.], [1.], [1.]]
```

Tests validate:
- Forward computation correctness
- Gradient computation accuracy
- Proper gradient backpropagation through chains of operations
- Shape compatibility checks

## 🎯 Key Features

- **Pure NumPy Implementation**: No external AD libraries used
- **Dynamic Graphs**: Supports arbitrary Python control flow
- **Educational**: Clear implementation showing AD internals
- **Extensible**: Easy to add new operations by inheriting from `Op`
- **Type Safety**: Assertions validate tensor shapes throughout

## 📐 Design Decisions

### Matrix-Only Tensors
**Simplification:** All tensors are stored as 2D matrices internally
- Scalars: `2` → `[[2]]`
- Vectors: `[1, 2, 3]` → `[[1], [2], [3]]` (column vector)

This simplifies shape compatibility checking and gradient computation.

### Gradient Accumulation
Gradients accumulate across multiple `.backward()` calls unless explicitly zeroed with `.zero_grad()`.

## 📈 Applications

The AD engine can be used for:
- **Function Optimization**: Finding minima using gradient descent
- **Neural Network Training**: Forward/backward passes for MLPs
- **Parameter Estimation**: Fitting models to data
- **Solving Differential Equations**: Physics-informed neural networks

## 🔬 Example: Simple Gradient Descent

```python
# Define objective function
def f(inputs):
    x = inputs[0]
    return sin(add(prod(x, 2), 0.5))

# Initialize variable
x = Tensor(3.5)

# Run gradient descent
for epoch in range(10):
    x.zero_grad()
    loss = f([x])
    loss.backward()
    
    # Update: x = x - lr * gradient
    x._arr = x._arr - 0.2 * x.grad
    
    print(f"Epoch {epoch}: x={x.numpy()}, loss={loss.numpy()}")
```

## 📚 References

**Primary Resources:**
- [Build Your Own PyTorch - Part 1: Computation Graphs](https://www.peterholderrieth.com/blog/2023/Build-Your-Own-Pytorch-1-Computation-Graphs/)
- [Build Your Own PyTorch - Part 2: Autograd](https://www.peterholderrieth.com/blog/2023/Build-Your-Own-Pytorch-2-Autograd/)
- [Build Your Own PyTorch - Part 3: Training Neural Networks](https://www.peterholderrieth.com/blog/2023/Build-Your-Own-Pytorch-3-Build-Classifier/)
- [PyTorch: A Gentle Introduction to torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

**Theoretical Background:**
- Automatic Differentiation
- Computational Graphs
- Backpropagation Algorithm
- Chain Rule in Multivariable Calculus

## 🔧 Extending the Framework

To add a new operation:

```python
class MyOp(Op):
    def __call__(self, *args, **kwargs) -> Tensor:
        args = self._ts(args)  # Convert to tensors
        # Compute forward pass
        result = custom_computation(args[0].numpy())
        return Tensor(result, parents=args, 
                     name=NameManager.new('myop'),
                     operation=self)
    
    def grad(self, back_grad: Tensor, *args, **kwargs) -> list[Tensor]:
        # Compute gradients w.r.t. each parent
        # Apply chain rule with back_grad
        return [Tensor(local_grad * back_grad, 
                      name=NameManager.new('myop_grad'))]
```

## 💡 Learning Outcomes

After working through this project, you will understand:
- How automatic differentiation works internally
- The relationship between computational graphs and gradient computation
- How PyTorch/TensorFlow implement backpropagation
- The mechanics of the chain rule in reverse-mode AD
- How to build differentiable programs from scratch

## 📝 Notes

- The implementation prioritizes clarity over performance
- All operations use NumPy for numerical computation
- Gradients are computed using reverse-mode AD (efficient for scalar outputs)
- The system supports arbitrary nesting of operations

## 🤝 Contributing

This was an academic project completed as part of coursework. Feel free to fork and extend with additional operations or optimizations.

## 📄 License

This project is part of academic coursework.

---

**Developed as part of university coursework - 2025**
