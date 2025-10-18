# TODO - Nutorch Implementation Status

This file tracks the implementation status and quality of all PyTorch methods in
Nutorch. Each method must meet standardized quality criteria before being
considered complete.

## Core Design Principles

### Dual Input Pattern: PyTorch API Parity + Nushell Idioms

**This is the fundamental design principle that makes Nutorch feel native to both PyTorch and Nushell users.**

Nutorch commands mirror PyTorch's dual API (method vs function form) while embracing Nushell's pipeline philosophy:

#### Binary Operations (add, sub, mul, div, mm, maximum)
- **Pipeline + Argument**: `$t1 | torch add $t2` ≈ `tensor1.add(tensor2)`
- **Two Arguments**: `torch add $t1 $t2` ≈ `torch.add(tensor1, tensor2)`

**Example**:
```nushell
# Pipeline + Argument (feels like tensor1.add(tensor2))
([1] | torch tensor) | torch add ([2] | torch tensor)

# Two Arguments (feels like torch.add(tensor1, tensor2))
torch add ([1] | torch tensor) ([2] | torch tensor)
```

**Python PyTorch equivalent**:
```python
# Method form
torch.tensor([1]).add(torch.tensor([2]))

# Function form
torch.add(torch.tensor([1]), torch.tensor([2]))
```

#### Unary Operations (sin, exp, neg, mean, softmax, shape, etc.)
- **Pipeline**: `$t | torch sin` ≈ `tensor.sin()`
- **Argument**: `torch sin $t` ≈ `torch.sin(tensor)`

#### List Operations (cat, stack)
- **Pipeline**: `[$t1 $t2] | torch cat` ≈ N/A (PyTorch only has function form)
- **Argument**: `torch cat [$t1 $t2]` ≈ `torch.cat([tensor1, tensor2])`

### Why This Flexibility Matters

This dual input pattern enables:
1. **Imperative style** (like Python): `torch add $a $b`
2. **Functional pipelines** (like Nushell): `$a | torch add $b | torch mean`
3. **Natural composition**: Mix styles for readable complex expressions
4. **Gradual learning**: Start with imperative, adopt pipelines as comfortable

**Every command must implement the appropriate dual input pattern for its operation type.**

---

## Quality Checklist Legend

For each method, we track:

- **Test Coverage**: Has at least one test file
- **Error Tests**: Tests include error/edge cases
- **Helper Usage**: Uses centralized helper functions from `lib.rs`
- **Dual Input**: Implements correct dual input pattern for operation type:
  - Binary ops: BOTH `$t1 | torch op $t2` AND `torch op $t1 $t2`
  - Unary ops: BOTH `$t | torch op` AND `torch op $t`
  - List ops: BOTH `[$ts] | torch op` AND `torch op [$ts]`
- **Examples**: Has comprehensive examples in signature
- **Validation**: Validates inputs (dimensions, shapes, etc.)
- **Documentation**: Has clear description and parameter docs

## Status Summary

- **Total Methods Implemented**: 39
- **Methods with Tests**: 33 (85%)
- **Methods Meeting All Quality Criteria**: 17 (44%)

---

## Tensor Creation Operations

### `torch.tensor(data, dtype=None, device=None, requires_grad=False)`

**Command**: `torch tensor`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (device, dtype, requires_grad)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `torch.full(size, fill_value, dtype=None, device=None, requires_grad=False)`

**Command**: `torch full`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `torch.randn(*size, dtype=None, device=None, requires_grad=False)`

**Command**: `torch randn`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `torch.linspace(start, end, steps, dtype=None, device=None, requires_grad=False)`

**Command**: `torch linspace`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `torch.arange(start, end, step, dtype=None, device=None)`

**Command**: `torch arange`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

---

## Binary Element-wise Operations

### `tensor.add(other, alpha=1)`

**Command**: `torch add`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (N/A - binary ops don't need creation helpers)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `tensor.sub(other)`

**Command**: `torch sub`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (N/A - binary ops don't need creation helpers)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `tensor.mul(other)`

**Command**: `torch mul`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (N/A - binary ops don't need creation helpers)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `tensor.div(other)`

**Command**: `torch div`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (N/A - binary ops don't need creation helpers)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `torch.maximum(input, other)`

**Command**: `torch maximum`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (N/A - binary ops don't need creation helpers)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

---

## Unary Element-wise Operations

### `tensor.neg()`

**Command**: `torch neg`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (N/A - unary ops don't need creation helpers)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `tensor.sin()`

**Command**: `torch sin`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (N/A - unary ops don't need creation helpers)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `tensor.exp()`

**Command**: `torch exp`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (N/A - unary ops don't need creation helpers)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `tensor.detach()`

**Command**: `torch detach`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (N/A - unary ops don't need creation helpers)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

---

## Reduction Operations

### `tensor.mean(dim=None, keepdim=False, dtype=None)`

**Command**: `torch mean`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (dtype)
- [x] Dual Input
- [x] Examples
- [x] Validation (dimension)
- [x] Documentation

### `tensor.max(dim=None, keepdim=False)`

**Command**: `torch max`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (N/A - reduction ops don't need creation helpers)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

### `tensor.argmax(dim=None, keepdim=False)`

**Command**: `torch argmax`

- [x] Test Coverage
- [x] Error Tests
- [x] Helper Usage (N/A - reduction ops don't need creation helpers)
- [x] Dual Input
- [x] Examples
- [x] Validation
- [x] Documentation

---

## Matrix Operations

### `tensor.mm(mat2)`

**Command**: `torch mm`

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [ ] Validation (shape compatibility)
- [ ] Documentation

### `tensor.t()`

**Command**: `torch t`

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [ ] Validation (2D only)
- [ ] Documentation

---

## Shape Manipulation

### `tensor.shape` (property)

**Command**: `torch shape`

- [ ] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [ ] Dual Input
- [x] Examples
- [ ] Validation
- [ ] Documentation

### `tensor.squeeze(dim=None)`

**Command**: `torch squeeze`

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [ ] Validation
- [ ] Documentation

### `tensor.unsqueeze(dim)`

**Command**: `torch unsqueeze`

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [x] Validation (dimension)
- [ ] Documentation

### `tensor.reshape(*shape)`

**Command**: `torch reshape`

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [ ] Validation
- [ ] Documentation

### `tensor.repeat(*sizes)`

**Command**: `torch repeat`

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [ ] Validation
- [ ] Documentation

### `tensor.repeat_interleave(repeats, dim=None)`

**Command**: `torch repeat_interleave`

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [ ] Validation
- [ ] Documentation

### `torch.cat(tensors, dim=0)`

**Command**: `torch cat`

- [x] Test Coverage
- [x] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [x] Validation (shapes, dimension)
- [ ] Documentation

### `torch.stack(tensors, dim=0)`

**Command**: `torch stack`

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [x] Validation (shapes, dimension)
- [ ] Documentation

---

## Indexing & Selection

### `tensor.gather(dim, index)`

**Command**: `torch gather`

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [x] Validation (dimension, shapes)
- [ ] Documentation

---

## Neural Network Operations

### `tensor.softmax(dim, dtype=None)`

**Command**: `torch softmax`

- [x] Test Coverage
- [ ] Error Tests
- [x] Helper Usage (dtype)
- [x] Dual Input
- [x] Examples
- [x] Validation (dimension)
- [ ] Documentation

### `tensor.log_softmax(dim, dtype=None)`

**Command**: `torch log_softmax`

- [x] Test Coverage
- [ ] Error Tests
- [x] Helper Usage (dtype)
- [x] Dual Input
- [x] Examples
- [x] Validation (dimension)
- [ ] Documentation

---

## Autograd Operations

### `tensor.backward(gradient=None, retain_graph=False)`

**Command**: `torch backward`

- [x] Test Coverage
- [x] Error Tests (non-scalar)
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [x] Validation (scalar only)
- [ ] Documentation

### `tensor.grad` (property)

**Command**: `torch grad`

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [ ] Validation
- [ ] Documentation

### `tensor.detach()`

See "Unary Element-wise Operations" section above

### `tensor.requires_grad_(requires_grad=True)` (partial)

**Note**: Currently only available as flag on tensor creation, not as standalone
command

- N/A - Implemented as `--requires_grad` flag on creation commands

---

## Optimizer Operations (Custom)

### Custom: SGD Step

**Command**: `torch sgd_step` **PyTorch Equivalent**: `optimizer.step()` for SGD

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [ ] Validation
- [ ] Documentation

### Custom: Zero Gradients

**Command**: `torch zero_grad` **PyTorch Equivalent**: `optimizer.zero_grad()`
or `tensor.grad.zero_()`

- [x] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [ ] Validation
- [ ] Documentation

---

## Utility Operations

### Custom: Convert to Nushell Value

**Command**: `torch value` **PyTorch Equivalent**: `tensor.tolist()` or
`tensor.numpy()`

- [ ] Test Coverage
- [ ] Error Tests
- [x] Helper Usage (uses `tensor_to_value`)
- [ ] Dual Input
- [x] Examples
- [ ] Validation
- [ ] Documentation

### Custom: Free Tensor from Registry

**Command**: `torch free` **PyTorch Equivalent**: `del tensor` (garbage
collection)

- [ ] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [x] Dual Input
- [x] Examples
- [ ] Validation
- [ ] Documentation

### `torch.manual_seed(seed)`

**Command**: `torch manual_seed`

- [ ] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [ ] Dual Input (N/A)
- [x] Examples
- [ ] Validation
- [ ] Documentation

### Custom: List Available Devices

**Command**: `torch devices` **PyTorch Equivalent**:
`torch.cuda.device_count()`, etc.

- [ ] Test Coverage
- [ ] Error Tests
- [ ] Helper Usage
- [ ] Dual Input (N/A)
- [x] Examples
- [ ] Validation
- [ ] Documentation

### Custom: Main torch Command

**Command**: `torch` **Purpose**: Entry point / help command

- [ ] Test Coverage (N/A)
- [ ] Error Tests (N/A)
- [ ] Helper Usage (N/A)
- [ ] Dual Input (N/A)
- [ ] Examples
- [ ] Validation (N/A)
- [ ] Documentation

---

## Not Yet Implemented (High Priority PyTorch Methods)

### Tensor Creation

- [ ] `torch.zeros()`
- [ ] `torch.ones()`
- [ ] `torch.zeros_like()`
- [ ] `torch.ones_like()`
- [ ] `torch.empty()`
- [ ] `torch.eye()`
- [ ] `torch.rand()` (uniform distribution)
- [ ] `torch.randint()`
- [ ] `torch.randperm()`

### Binary Operations

- [ ] `tensor.pow()`
- [ ] `tensor.sqrt()`
- [ ] `tensor.abs()`
- [ ] `tensor.clamp()`

### Reduction Operations

- [ ] `tensor.sum()`
- [ ] `tensor.min()`
- [ ] `tensor.argmin()`
- [ ] `tensor.std()`
- [ ] `tensor.var()`

### Matrix Operations

- [ ] `tensor.matmul()` / `tensor @ other`
- [ ] `tensor.bmm()` (batch matrix multiply)
- [ ] `tensor.transpose()`

### Shape Operations

- [ ] `tensor.view()`
- [ ] `tensor.permute()`
- [ ] `tensor.flatten()`
- [ ] `tensor.split()`
- [ ] `tensor.chunk()`

### Neural Network

- [ ] `F.relu()`
- [ ] `F.sigmoid()`
- [ ] `F.tanh()`
- [ ] `F.dropout()`
- [ ] `F.batch_norm()`
- [ ] `F.layer_norm()`
- [ ] `F.conv2d()`
- [ ] `F.max_pool2d()`
- [ ] `F.linear()`
- [ ] `F.embedding()`
- [ ] `F.cross_entropy()`
- [ ] `F.mse_loss()`

### Comparison Operations

- [ ] `tensor.eq()`
- [ ] `tensor.ne()`
- [ ] `tensor.gt()`
- [ ] `tensor.lt()`
- [ ] `tensor.ge()`
- [ ] `tensor.le()`

### Logical Operations

- [ ] `tensor.logical_and()`
- [ ] `tensor.logical_or()`
- [ ] `tensor.logical_not()`

### Type Conversions

- [ ] `tensor.float()`
- [ ] `tensor.int()`
- [ ] `tensor.long()`
- [ ] `tensor.double()`
- [ ] `tensor.to(device)`

### Advanced Indexing

- [ ] `tensor[...]` (slicing)
- [ ] `tensor.index_select()`
- [ ] `tensor.masked_select()`
- [ ] `tensor.where()`

---

## Progress Metrics

### Implementation Progress

- Tensor Creation: 5/14 (36%)
- Binary Operations: 5/9 (56%)
- Unary Operations: 4/8 (50%)
- Reduction Operations: 3/8 (38%)
- Matrix Operations: 2/3 (67%)
- Shape Manipulation: 8/14 (57%)
- Neural Network: 2/15 (13%)
- Autograd: 3/4 (75%)

### Quality Progress (Implemented Methods Only)

- Test Coverage: 26/39 (67%)
- Error Tests: 2/39 (5%)
- Helper Usage: 7/39 (18%)
- Dual Input Support: 31/35 (89%) (4 N/A)
- Examples: 39/39 (100%)
- Validation: 7/39 (18%)
- Documentation: 0/39 (0%)

### API Design Compliance (Dual Input Pattern)

- Binary Ops with Full Dual Input: 5/5 (100%)
  - All support both `$t1 | torch op $t2` AND `torch op $t1 $t2`
- Unary Ops with Full Dual Input: 8/11 (73%)
  - Missing: sin, exp, max (need to verify argument form)
- List Ops with Full Dual Input: 2/2 (100%)
  - cat, stack both support dual input

### Path to 1.0

To reach version 1.0, all currently implemented methods must achieve:

- [x] 100% Test Coverage (currently 67%)
- [ ] 80%+ Error Test Coverage (currently 5%)
- [ ] 100% Helper Usage (currently 18%)
- [ ] 100% Dual Input Support (currently 89%)
- [x] 100% Examples (currently 100%)
- [ ] 80%+ Validation (currently 18%)
- [ ] 100% Documentation (currently 0%)

---

## Notes

1. **Helper Usage**: Should use `get_device_from_call()`,
   `get_kind_from_call()`, `add_grad_from_call()` where applicable, and new
   input validation helpers once created.

2. **Dual Input Pattern - CORE DESIGN PRINCIPLE**: This is NOT just a feature,
   it's the fundamental design principle that bridges PyTorch and Nushell
   paradigms. Every command must support the appropriate dual input pattern for
   its operation type. This makes Nutorch feel native to both PyTorch users (who
   expect method/function duality) and Nushell users (who expect pipeline
   composition). Missing dual input support breaks the API contract.

3. **Error Tests**: Should test invalid dimensions, mismatched shapes, device
   conflicts, and other error conditions.

4. **Validation**: Commands should validate inputs before calling tch-rs to
   provide clear error messages.

5. **Documentation**: All public functions in `lib.rs` and all commands should
   have clear docstrings.
