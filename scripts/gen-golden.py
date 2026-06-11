#!/usr/bin/env python
"""Golden-test generator (issue 0005).

Run with the repo venv (the EXACT torch the daemon links):

    .venv-torch/bin/python scripts/gen-golden.py

Emits nutorchd/tests/golden.json. Every input tensor is constructed with the
dtype the daemon would assign (float32 default — Python's int64 inference
must never leak in) on device=mps, so expected outputs are bitwise
comparable with the daemon's MPS results.

Known documented deviation: nutorch's `mean` reduces in float32 regardless of
input dtype (v1 fidelity); PyTorch's mean errors on int inputs. The int-mean
case below encodes nutorch's documented semantics (t.float().mean()).

randn: the daemon generates on the seeded CPU generator and transfers to MPS
(tch cannot reach the MPS generator); the golden mirrors exactly that.
"""

import json
import pathlib

import torch

assert torch.backends.mps.is_available(), "golden generation requires MPS"
DEV = "mps"
DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}

cases = []


def make(spec):
    dtype = DTYPES[spec.get("dtype", "float32")]
    return torch.tensor(spec["data"], dtype=dtype, device=DEV)


def ok(name, op, tensors, params, compute):
    """compute(real_tensors) -> list of output tensors."""
    outs = compute([make(t) for t in tensors])
    cases.append(
        {
            "name": name,
            "op": op,
            "tensors": tensors,
            "params": params,
            "expect": {"values": [o.cpu().tolist() for o in outs]},
        }
    )


def ok_value(name, op, tensors, params, value):
    cases.append(
        {
            "name": name,
            "op": op,
            "tensors": tensors,
            "params": params,
            "expect": {"value": value},
        }
    )


def err(name, op, tensors, params, code):
    cases.append(
        {
            "name": name,
            "op": op,
            "tensors": tensors,
            "params": params,
            "expect": {"error": code},
        }
    )


def t(data, dtype="float32"):
    return {"data": data, "dtype": dtype}


# --- pointwise ---
ok("add_broadcast", "add", [t([[1, 2, 3], [4, 5, 6]]), t([10, 20, 30])], {},
   lambda ts: [ts[0] + ts[1]])
err("add_not_broadcastable", "add", [t([[1, 2, 3], [4, 5, 6]]), t([1, 2, 3, 4])], {},
    "shape_mismatch")
ok("sub", "sub", [t([5, 7, 9]), t([1, 2, 3])], {}, lambda ts: [ts[0] - ts[1]])
ok("sin", "sin", [t([0.0, 0.5, 1.0, 1.5707963267948966])], {},
   lambda ts: [torch.sin(ts[0])])
ok("pow_int_exponent", "pow", [t([1, 2, 3])], {"exponent": 3},
   lambda ts: [torch.pow(ts[0], 3)])
ok("pow_float_exponent", "pow", [t([1, 4, 9])], {"exponent": 0.5},
   lambda ts: [torch.pow(ts[0], 0.5)])
ok("clamp_both_bounds", "clamp", [t([-5, 0, 5])], {"min": -1, "max": 1},
   lambda ts: [torch.clamp(ts[0], -1, 1)])
ok("clamp_min_only", "clamp", [t([-5, 0, 5])], {"min": 0},
   lambda ts: [torch.clamp(ts[0], min=0)])

# --- reductions ---
ok("sum_all", "sum", [t([[1, 2], [3, 4]])], {}, lambda ts: [ts[0].sum()])
ok("sum_dim0", "sum", [t([[1, 2], [3, 4]])], {"dim": 0},
   lambda ts: [ts[0].sum(dim=0)])
ok("sum_dim1_keepdim", "sum", [t([[1, 2], [3, 4]])], {"dim": 1, "keepdim": True},
   lambda ts: [ts[0].sum(dim=1, keepdim=True)])
ok("mean_float", "mean", [t([1, 2, 3, 4])], {}, lambda ts: [ts[0].mean()])
# Documented deviation: nutorch mean reduces ints in float32 (v1 fidelity).
ok("mean_int_input_is_float32", "mean", [t([1, 2, 3, 4], "int64")], {},
   lambda ts: [ts[0].float().mean()])

# --- comparison ---
ok("eq", "eq", [t([1, 2, 3]), t([1, 0, 3])], {}, lambda ts: [ts[0] == ts[1]])
ok_value("allclose_true", "allclose", [t([1.0, 2.0]), t([1.0, 2.0])], {}, True)
ok_value("allclose_false_default_tol", "allclose", [t([1.0, 2.0]), t([1.01, 2.0])], {}, False)
ok_value("allclose_true_loose_rtol", "allclose", [t([1.0, 2.0]), t([1.01, 2.0])],
         {"rtol": 0.1}, True)
ok("sort_default", "sort", [t([3, 1, 2])], {},
   lambda ts: list(torch.sort(ts[0])))
ok("sort_descending", "sort", [t([[3, 1, 2], [6, 5, 4]])], {"dim": 1, "descending": True},
   lambda ts: list(torch.sort(ts[0], dim=1, descending=True)))

# --- linalg ---
ok("mm", "mm", [t([[1, 2, 3], [4, 5, 6]]), t([[7, 8], [9, 10], [11, 12]])], {},
   lambda ts: [ts[0] @ ts[1]])
err("mm_rank1", "mm", [t([1, 2, 3]), t([1, 2, 3])], {}, "shape_mismatch")
err("mm_inner_mismatch", "mm", [t([[1, 2], [3, 4]]), t([[1, 2], [3, 4], [5, 6]])], {},
    "shape_mismatch")

# --- shape ---
ok("cat_dim0", "cat", [t([1]), t([2]), t([3])], {}, lambda ts: [torch.cat(ts)])
ok("cat_dim1", "cat", [t([[1, 2]]), t([[3, 4]])], {"dim": 1},
   lambda ts: [torch.cat(ts, dim=1)])

# --- creation ---
ok("full", "full", [], {"shape": [2, 3], "value": 7},
   lambda ts: [torch.full([2, 3], 7, dtype=torch.float32, device=DEV)])
ok("full_float_value", "full", [], {"shape": [2], "value": 0.5},
   lambda ts: [torch.full([2], 0.5, dtype=torch.float32, device=DEV)])
err("randn_int_dtype", "randn", [], {"shape": [2], "dtype": "int64"}, "bad_dtype")
err("randn_float64", "randn", [], {"shape": [2], "dtype": "float64"}, "bad_dtype")

# randn with a seed: mirror the daemon (CPU generator, then transfer).
torch.manual_seed(42)
seeded = torch.randn([2, 3], dtype=torch.float32, device="cpu").to(DEV)
cases.append(
    {
        "name": "randn_seeded",
        "op": "randn",
        "tensors": [],
        "params": {"shape": [2, 3]},
        "seed": 42,
        "expect": {"values": [seeded.cpu().tolist()]},
    }
)

# --- pointwise sweep (issue 0005 exp 2): data-driven ---
# Domain-aware samples per unary op.
UNARY_SAMPLES = {
    "abs": [-2.5, 0.0, 3.0],
    "acos": [-0.9, 0.0, 0.9],
    "acosh": [1.0, 1.5, 3.0],
    "asin": [-0.9, 0.0, 0.9],
    "asinh": [-2.0, 0.0, 2.0],
    "atan": [-2.0, 0.0, 2.0],
    "atanh": [-0.9, 0.0, 0.9],
    "ceil": [-1.5, 0.2, 2.7],
    "cos": [0.0, 1.0, 3.14159],
    "cosh": [-1.0, 0.0, 1.0],
    "deg2rad": [0.0, 90.0, 180.0],
    "digamma": [0.5, 1.0, 3.5],
    "erf": [-1.0, 0.0, 1.0],
    "erfc": [-1.0, 0.0, 1.0],
    "exp": [-1.0, 0.0, 2.0],
    "exp2": [-1.0, 0.0, 3.0],
    "expm1": [-0.5, 0.0, 0.5],
    "floor": [-1.5, 0.2, 2.7],
    "frac": [-1.75, 0.25, 2.5],
    "i0": [0.0, 1.0, 2.0],
    "lgamma": [0.5, 1.0, 4.0],
    "log": [0.5, 1.0, 10.0],
    "log10": [0.1, 1.0, 100.0],
    "log1p": [-0.5, 0.0, 1.0],
    "log2": [0.5, 1.0, 8.0],
    "logit": [0.1, 0.5, 0.9],
    "neg": [-2.0, 0.0, 3.0],
    "rad2deg": [0.0, 1.5707963267948966, 3.141592653589793],
    "reciprocal": [0.5, 1.0, 4.0],
    "relu": [-2.0, 0.0, 3.0],
    "round": [-1.5, 0.4, 2.5],
    "rsqrt": [0.25, 1.0, 4.0],
    "sgn": [-2.0, 0.0, 3.0],
    "sigmoid": [-2.0, 0.0, 2.0],
    "sign": [-2.0, 0.0, 3.0],
    "sinc": [-0.5, 0.0, 0.5],
    "sinh": [-1.0, 0.0, 1.0],
    "sqrt": [0.0, 1.0, 9.0],
    "square": [-2.0, 0.0, 3.0],
    "tan": [-0.5, 0.0, 0.5],
    "tanh": [-1.0, 0.0, 1.0],
    "trunc": [-1.7, 0.3, 2.7],
}
for name, sample in UNARY_SAMPLES.items():
    ok(f"pw_{name}", name, [t(sample)], {},
       lambda ts, f=getattr(torch, name): [f(ts[0])])

ok("pw_softmax", "softmax", [t([[1, 2, 3], [1, 1, 1]])], {"dim": 1},
   lambda ts: [torch.softmax(ts[0], dim=1)])
ok("pw_log_softmax", "log_softmax", [t([[1, 2, 3]])], {"dim": 1},
   lambda ts: [torch.log_softmax(ts[0], dim=1)])
# nan_to_num: golden inputs must be finite (bare NaN/Infinity are not valid
# JSON); this case covers param plumbing + identity-on-finite. The real
# NaN/inf replacement semantics live in a Rust dispatch unit test, which can
# construct non-finite tensors directly.
ok("pw_nan_to_num_finite", "nan_to_num", [t([1.0, -2.0, 3.0])],
   {"nan": 0.5, "posinf": 100.0, "neginf": -100.0},
   lambda ts: [torch.nan_to_num(ts[0], nan=0.5, posinf=100.0, neginf=-100.0)])

# Binary, broadcasting: (a, b) samples chosen in-domain.
BINARY_SAMPLES = {
    "mul": ([1.5, -2.0, 3.0], [2.0, 0.5, -1.0]),
    "div": ([3.0, -8.0, 1.0], [2.0, 4.0, -0.5]),
    "maximum": ([1.0, 5.0, -3.0], [2.0, 4.0, -1.0]),
    "minimum": ([1.0, 5.0, -3.0], [2.0, 4.0, -1.0]),
    "atan2": ([1.0, -1.0, 0.5], [1.0, 1.0, -0.5]),
    "fmod": ([5.0, -7.0, 9.5], [3.0, 3.0, 2.0]),
    "remainder": ([5.0, -7.0, 9.5], [3.0, 3.0, 2.0]),
    "floor_divide": ([5.0, -7.0, 9.5], [3.0, 3.0, 2.0]),
    "hypot": ([3.0, 5.0, 8.0], [4.0, 12.0, 15.0]),
    "copysign": ([1.5, -2.5, 3.5], [-1.0, 1.0, -1.0]),
    "xlogy": ([0.0, 2.0, 3.0], [1.0, 2.0, 0.5]),
    "logaddexp": ([-1.0, 0.0, 2.0], [1.0, 0.0, -2.0]),
}
for name, (a, b) in BINARY_SAMPLES.items():
    ok(f"pw_{name}", name, [t(a), t(b)], {},
       lambda ts, f=getattr(torch, name): [f(ts[0], ts[1])])
ok("pw_mul_broadcast", "mul", [t([[1, 2], [3, 4]]), t([10, 20])], {},
   lambda ts: [ts[0] * ts[1]])

# --alpha: signs differ between add and sub (PyTorch semantics).
ok("pw_add_alpha", "add", [t([10.0, 10.0]), t([1.0, 2.0])], {"alpha": 2},
   lambda ts: [torch.add(ts[0], ts[1], alpha=2)])
ok("pw_sub_alpha", "sub", [t([10.0, 10.0]), t([1.0, 2.0])], {"alpha": 2},
   lambda ts: [torch.sub(ts[0], ts[1], alpha=2)])
ok("pw_add_alpha_float", "add", [t([1.0, 2.0]), t([3.0, 4.0])], {"alpha": 0.5},
   lambda ts: [torch.add(ts[0], ts[1], alpha=0.5)])

# --- reductions + comparison sweep (issue 0005 exp 3) ---
M = [[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]]
ok("rc_prod", "prod", [t([1.5, 2.0, 3.0])], {}, lambda ts: [ts[0].prod()])
ok("rc_prod_dim", "prod", [t(M)], {"dim": 1}, lambda ts: [ts[0].prod(dim=1)])
ok("rc_amax", "amax", [t(M)], {"dim": 0}, lambda ts: [torch.amax(ts[0], dim=0)])
ok("rc_amin", "amin", [t(M)], {"dim": 1}, lambda ts: [torch.amin(ts[0], dim=1)])
ok("rc_max_all", "max", [t(M)], {}, lambda ts: [ts[0].max()])
ok("rc_max_dim", "max", [t(M)], {"dim": 1},
   lambda ts: list(ts[0].max(dim=1)))
ok("rc_min_all", "min", [t(M)], {}, lambda ts: [ts[0].min()])
ok("rc_min_dim", "min", [t(M)], {"dim": 0},
   lambda ts: list(ts[0].min(dim=0)))
ok("rc_median_all", "median", [t([3.0, 1.0, 2.0])], {}, lambda ts: [ts[0].median()])
ok("rc_median_dim", "median", [t(M)], {"dim": 1},
   lambda ts: list(ts[0].median(dim=1)))
ok("rc_argmax", "argmax", [t(M)], {"dim": 1}, lambda ts: [ts[0].argmax(dim=1)])
ok("rc_argmin", "argmin", [t(M)], {}, lambda ts: [ts[0].argmin()])
ok("rc_all", "all", [t([1.0, 1.0, 0.0])], {}, lambda ts: [ts[0].bool().all()])
ok("rc_any", "any", [t([0.0, 0.0, 1.0])], {}, lambda ts: [ts[0].bool().any()])
ok("rc_std", "std", [t([1.0, 2.0, 3.0, 4.0])], {},
   lambda ts: [ts[0].std(correction=1)])
ok("rc_std_corr0", "std", [t([1.0, 2.0, 3.0, 4.0])], {"correction": 0},
   lambda ts: [ts[0].std(correction=0)])
ok("rc_var_dim", "var", [t(M)], {"dim": 1, "correction": 1},
   lambda ts: [ts[0].var(dim=1, correction=1)])
ok("rc_nansum", "nansum", [t([1.0, 2.0, 3.0])], {}, lambda ts: [ts[0].nansum()])
ok("rc_logsumexp", "logsumexp", [t(M)], {"dim": 1},
   lambda ts: [torch.logsumexp(ts[0], dim=1)])
ok("rc_count_nonzero", "count_nonzero", [t([1.0, 0.0, 3.0, 0.0])], {},
   lambda ts: [torch.count_nonzero(ts[0])])
ok("rc_cumsum", "cumsum", [t([1.0, 2.0, 3.0])], {"dim": 0},
   lambda ts: [ts[0].cumsum(dim=0)])
ok("rc_cumprod", "cumprod", [t([1.0, 2.0, 3.0])], {"dim": 0},
   lambda ts: [ts[0].cumprod(dim=0)])
ok("rc_norm", "norm", [t([3.0, 4.0])], {}, lambda ts: [ts[0].norm()])
ok("rc_norm_p1_dim", "norm", [t(M)], {"p": 1.0, "dim": 1},
   lambda ts: [ts[0].norm(p=1, dim=1)])

A, B = [1.0, 5.0, 3.0], [2.0, 4.0, 3.0]
for name in ["gt", "lt", "ge", "le", "ne"]:
    ok(f"rc_{name}", name, [t(A), t(B)], {},
       lambda ts, f=getattr(torch, name): [f(ts[0], ts[1])])
ok("rc_logical_and", "logical_and", [t([1.0, 0.0, 1.0]), t([1.0, 1.0, 0.0])], {},
   lambda ts: [torch.logical_and(ts[0], ts[1])])
ok("rc_logical_or", "logical_or", [t([1.0, 0.0, 0.0]), t([0.0, 0.0, 1.0])], {},
   lambda ts: [torch.logical_or(ts[0], ts[1])])
ok("rc_logical_xor", "logical_xor", [t([1.0, 0.0, 1.0]), t([1.0, 1.0, 0.0])], {},
   lambda ts: [torch.logical_xor(ts[0], ts[1])])
ok("rc_logical_not", "logical_not", [t([1.0, 0.0, 2.0])], {},
   lambda ts: [torch.logical_not(ts[0])])
ok("rc_isclose", "isclose", [t([1.0, 2.0]), t([1.001, 2.0])], {"rtol": 0.01},
   lambda ts: [torch.isclose(ts[0], ts[1], rtol=0.01)])
# Predicates on FINITE inputs only (golden constraint); the non-finite TRUE
# path is guarded by a Rust dispatch unit test.
for name in ["isnan", "isinf", "isfinite", "isposinf", "isneginf"]:
    ok(f"rc_{name}_finite", name, [t([1.0, -2.0, 0.0])], {},
       lambda ts, f=getattr(torch, name): [f(ts[0])])
ok_value("rc_equal_true", "equal", [t([1.0, 2.0]), t([1.0, 2.0])], {}, True)
ok_value("rc_equal_false", "equal", [t([1.0, 2.0]), t([1.0, 3.0])], {}, False)
ok("rc_topk", "topk", [t([1.0, 5.0, 3.0, 4.0])], {"k": 2},
   lambda ts: list(torch.topk(ts[0], 2)))
ok("rc_topk_smallest", "topk", [t([1.0, 5.0, 3.0, 4.0])], {"k": 2, "smallest": True},
   lambda ts: list(torch.topk(ts[0], 2, largest=False)))
ok("rc_argsort", "argsort", [t([3.0, 1.0, 2.0])], {"descending": True},
   lambda ts: [torch.argsort(ts[0], descending=True)])

# --- linalg + shape sweep (issue 0005 exp 4) ---
M2 = [[1.0, 2.0], [3.0, 4.0]]
ok("ls_matmul", "matmul", [t(M2), t(M2)], {}, lambda ts: [ts[0] @ ts[1]])
ok("ls_bmm", "bmm", [t([M2, M2]), t([M2, M2])], {},
   lambda ts: [torch.bmm(ts[0], ts[1])])
ok("ls_dot", "dot", [t([1.0, 2.0, 3.0]), t([4.0, 5.0, 6.0])], {},
   lambda ts: [torch.dot(ts[0], ts[1])])
ok("ls_outer", "outer", [t([1.0, 2.0]), t([3.0, 4.0, 5.0])], {},
   lambda ts: [torch.outer(ts[0], ts[1])])
ok("ls_einsum_mm", "einsum", [t(M2), t(M2)], {"equation": "ij,jk->ik"},
   lambda ts: [torch.einsum("ij,jk->ik", ts[0], ts[1])])
ok("ls_einsum_trace", "einsum", [t(M2)], {"equation": "ii"},
   lambda ts: [torch.einsum("ii", ts[0])])
M3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
ok("ls_tril", "tril", [t(M3)], {}, lambda ts: [torch.tril(ts[0])])
ok("ls_triu_diag1", "triu", [t(M3)], {"diagonal": 1},
   lambda ts: [torch.triu(ts[0], diagonal=1)])
ok("ls_diag_extract", "diag", [t(M3)], {}, lambda ts: [torch.diag(ts[0])])
ok("ls_diag_build", "diag", [t([1.0, 2.0, 3.0])], {},
   lambda ts: [torch.diag(ts[0])])
ok("ls_trace", "trace", [t(M2)], {}, lambda ts: [torch.trace(ts[0])])
INV = [[4.0, 7.0], [2.0, 6.0]]
ok("ls_det", "det", [t(INV)], {}, lambda ts: [torch.det(ts[0])])
ok("ls_inverse", "inverse", [t(INV)], {}, lambda ts: [torch.inverse(ts[0])])
ok("ls_svd", "svd", [t(M2)], {},
   lambda ts: list(torch.svd(ts[0], some=False, compute_uv=True)))
ok("ls_solve", "solve", [t(INV), t([[1.0], [2.0]])], {},
   lambda ts: [torch.linalg.solve(ts[0], ts[1])])
V6 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
ok("ls_reshape", "reshape", [t(V6)], {"shape": [2, 3]},
   lambda ts: [ts[0].reshape(2, 3)])
ok("ls_reshape_infer", "reshape", [t(V6)], {"shape": [3, -1]},
   lambda ts: [ts[0].reshape(3, -1)])
ok("ls_permute", "permute", [t([M2])], {"dims": [2, 0, 1]},
   lambda ts: [ts[0].permute(2, 0, 1)])
ok("ls_transpose", "transpose", [t(M3)], {"dim0": 0, "dim1": 1},
   lambda ts: [ts[0].transpose(0, 1)])
ok("ls_t", "t", [t(M2)], {}, lambda ts: [ts[0].t()])
ok("ls_squeeze", "squeeze", [t([[1.0, 2.0]])], {}, lambda ts: [ts[0].squeeze()])
ok("ls_unsqueeze", "unsqueeze", [t([1.0, 2.0])], {"dim": 0},
   lambda ts: [ts[0].unsqueeze(0)])
ok("ls_flatten", "flatten", [t([M2])], {}, lambda ts: [ts[0].flatten()])
ok("ls_stack", "stack", [t([1.0, 2.0]), t([3.0, 4.0]), t([5.0, 6.0])], {"dim": 1},
   lambda ts: [torch.stack(ts, dim=1)])
ok("ls_split", "split", [t(V6)], {"split_size": 2},
   lambda ts: list(torch.split(ts[0], 2)))
ok("ls_chunk", "chunk", [t(V6)], {"chunks": 3},
   lambda ts: list(torch.chunk(ts[0], 3)))
ok("ls_gather", "gather", [t(M2), t([[0, 0], [1, 0]], "int64")], {"dim": 1},
   lambda ts: [torch.gather(ts[0], 1, ts[1])])
ok("ls_index_select", "index_select", [t(M3), t([0, 2], "int64")], {"dim": 0},
   lambda ts: [torch.index_select(ts[0], 0, ts[1])])
ok("ls_masked_select", "masked_select", [t([1.0, 2.0, 3.0]), t([1.0, 0.0, 1.0])], {},
   lambda ts: [torch.masked_select(ts[0], ts[1] != 0)])
ok("ls_where", "where", [t([1.0, 0.0, 1.0]), t([10.0, 20.0, 30.0]), t([-1.0, -2.0, -3.0])], {},
   lambda ts: [torch.where(ts[0] != 0, ts[1], ts[2])])
ok("ls_narrow", "narrow", [t(V6)], {"dim": 0, "start": 1, "length": 3},
   lambda ts: [ts[0].narrow(0, 1, 3)])
ok("ls_flip", "flip", [t(M2)], {"dims": [0]}, lambda ts: [ts[0].flip(0)])
ok("ls_roll", "roll", [t(V6)], {"shifts": [2]}, lambda ts: [ts[0].roll(2)])
ok("ls_repeat", "repeat", [t([1.0, 2.0])], {"repeats": [2, 3]},
   lambda ts: [ts[0].repeat(2, 3)])
ok("ls_repeat_interleave", "repeat_interleave", [t([1.0, 2.0])], {"repeats": 3},
   lambda ts: [ts[0].repeat_interleave(3)])
ok("ls_movedim", "movedim", [t([M2])], {"source": 0, "destination": 2},
   lambda ts: [ts[0].movedim(0, 2)])

# --- creation + remainder sweep (issue 0005 exp 5) ---
ok("cr_zeros", "zeros", [], {"shape": [2, 2]},
   lambda ts: [torch.zeros(2, 2, dtype=torch.float32, device=DEV)])
ok("cr_ones_int", "ones", [], {"shape": [3], "dtype": "int64"},
   lambda ts: [torch.ones(3, dtype=torch.int64, device=DEV)])
ok("cr_eye", "eye", [], {"n": 3},
   lambda ts: [torch.eye(3, dtype=torch.float32, device=DEV)])
ok("cr_eye_m", "eye", [], {"n": 2, "m": 4},
   lambda ts: [torch.eye(2, 4, dtype=torch.float32, device=DEV)])
ok("cr_arange", "arange", [], {"end": 5},
   lambda ts: [torch.arange(5, dtype=torch.float32, device=DEV)])
ok("cr_arange_start_step", "arange", [], {"end": 7, "start": 1, "step": 2},
   lambda ts: [torch.arange(1, 7, 2, dtype=torch.float32, device=DEV)])
ok("cr_linspace", "linspace", [], {"start": 0, "end": 1, "steps": 5},
   lambda ts: [torch.linspace(0, 1, 5, dtype=torch.float32, device=DEV)])
torch.manual_seed(7)
rand_expected = torch.rand([2, 2], dtype=torch.float32, device="cpu").to(DEV)
cases.append({"name": "cr_rand_seeded", "op": "rand", "tensors": [],
              "params": {"shape": [2, 2]}, "seed": 7,
              "expect": {"values": [rand_expected.cpu().tolist()]}})
torch.manual_seed(9)
ri_expected = torch.randint(2, 10, [2, 3], dtype=torch.int64, device="cpu").to(DEV)
cases.append({"name": "cr_randint_seeded", "op": "randint", "tensors": [],
              "params": {"high": 10, "shape": [2, 3], "low": 2}, "seed": 9,
              "expect": {"values": [ri_expected.cpu().tolist()]}})
ok("cr_zeros_like", "zeros_like", [t([[1.0, 2.0], [3.0, 4.0]])], {},
   lambda ts: [torch.zeros_like(ts[0])])
ok("cr_ones_like", "ones_like", [t([1, 2, 3], "int64")], {},
   lambda ts: [torch.ones_like(ts[0])])
ok("cr_full_like", "full_like", [t([1.0, 2.0])], {"value": 7},
   lambda ts: [torch.full_like(ts[0], 7)])
torch.manual_seed(11)
rl_expected = torch.rand([2, 2], dtype=torch.float32, device="cpu").to(DEV)
cases.append({"name": "cr_rand_like_seeded", "op": "rand_like",
              "tensors": [t([[1.0, 2.0], [3.0, 4.0]])], "params": {}, "seed": 11,
              "expect": {"values": [rl_expected.cpu().tolist()]}})
torch.manual_seed(13)
rnl_expected = torch.randn([3], dtype=torch.float32, device="cpu").to(DEV)
cases.append({"name": "cr_randn_like_seeded", "op": "randn_like",
              "tensors": [t([1.0, 2.0, 3.0])], "params": {}, "seed": 13,
              "expect": {"values": [rnl_expected.cpu().tolist()]}})
# HandleOrScalar: tensor bounds / exponents / weights
ok("cr_clamp_tensor_bounds", "clamp",
   [t([-5.0, 0.0, 5.0]), t([-1.0, -1.0, -1.0]), t([1.0, 2.0, 3.0])],
   {"min": "T1", "max": "T2"},
   lambda ts: [torch.clamp(ts[0], ts[1], ts[2])])
ok("cr_pow_tensor_exponent", "pow", [t([2.0, 3.0, 4.0]), t([2.0, 1.0, 0.5])],
   {"exponent": "T1"},
   lambda ts: [torch.pow(ts[0], ts[1])])
ok("cr_lerp_scalar", "lerp", [t([0.0, 10.0]), t([10.0, 20.0])], {"weight": 0.5},
   lambda ts: [torch.lerp(ts[0], ts[1], 0.5)])
ok("cr_lerp_tensor", "lerp", [t([0.0, 10.0]), t([10.0, 20.0]), t([0.25, 0.75])],
   {"weight": "T2"},
   lambda ts: [torch.lerp(ts[0], ts[1], ts[2])])
ok("cr_addcmul", "addcmul", [t([1.0, 2.0]), t([3.0, 4.0]), t([5.0, 6.0])],
   {"value": 2},
   lambda ts: [torch.addcmul(ts[0], ts[1], ts[2], value=2)])
ok("cr_addcdiv", "addcdiv", [t([1.0, 2.0]), t([6.0, 8.0]), t([2.0, 4.0])], {},
   lambda ts: [torch.addcdiv(ts[0], ts[1], ts[2])])
ok("cr_cross", "cross", [t([1.0, 0.0, 0.0]), t([0.0, 1.0, 0.0])], {},
   lambda ts: [torch.linalg.cross(ts[0], ts[1])])
ok("cr_kron", "kron", [t([1.0, 2.0]), t([3.0, 4.0])], {},
   lambda ts: [torch.kron(ts[0], ts[1])])
ok("cr_tensordot", "tensordot", [t([[1.0, 2.0], [3.0, 4.0]]), t([[5.0, 6.0], [7.0, 8.0]])],
   {"dims": 2},
   lambda ts: [torch.tensordot(ts[0], ts[1], dims=2)])
ok("cr_take_along_dim", "take_along_dim",
   [t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), t([[0, 2], [1, 0]], "int64")],
   {"dim": 1},
   lambda ts: [torch.take_along_dim(ts[0], ts[1], dim=1)])
ok("cr_searchsorted", "searchsorted", [t([1.0, 3.0, 5.0, 7.0]), t([2.0, 6.0])], {},
   lambda ts: [torch.searchsorted(ts[0], ts[1])])
ok("cr_bucketize", "bucketize", [t([2.0, 6.0]), t([1.0, 3.0, 5.0, 7.0])], {},
   lambda ts: [torch.bucketize(ts[0], ts[1])])
ok("cr_msort", "msort", [t([[3.0, 1.0], [2.0, 4.0]])], {},
   lambda ts: [torch.msort(ts[0])])
ok("cr_diff", "diff", [t([1.0, 4.0, 9.0, 16.0])], {},
   lambda ts: [torch.diff(ts[0])])
ok("cr_scatter", "scatter",
   [t([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), t([[0, 1], [1, 2]], "int64"),
    t([[10.0, 20.0], [30.0, 40.0]])],
   {"dim": 1},
   lambda ts: [torch.scatter(ts[0], 1, ts[1], ts[2])])
ok("cr_bitwise_and", "bitwise_and", [t([12, 10], "int64"), t([10, 6], "int64")], {},
   lambda ts: [torch.bitwise_and(ts[0], ts[1])])
ok("cr_bitwise_or", "bitwise_or", [t([12, 10], "int64"), t([10, 6], "int64")], {},
   lambda ts: [torch.bitwise_or(ts[0], ts[1])])
ok("cr_bitwise_xor", "bitwise_xor", [t([12, 10], "int64"), t([10, 6], "int64")], {},
   lambda ts: [torch.bitwise_xor(ts[0], ts[1])])
ok("cr_bitwise_not", "bitwise_not", [t([0, 1, 7], "int64")], {},
   lambda ts: [torch.bitwise_not(ts[0])])
ok("cr_left_shift", "bitwise_left_shift", [t([1, 2], "int64"), t([2, 3], "int64")], {},
   lambda ts: [torch.bitwise_left_shift(ts[0], ts[1])])
ok("cr_right_shift", "bitwise_right_shift", [t([8, 16], "int64"), t([2, 3], "int64")], {},
   lambda ts: [torch.bitwise_right_shift(ts[0], ts[1])])
ok("cr_unique", "unique", [t([3.0, 1.0, 2.0, 1.0, 3.0])], {},
   lambda ts: [torch.unique(ts[0])])
# Rank-2 case pins the flattening default (result-review finding: a 1-D-only
# golden let a non-flattening implementation pass vacuously).
ok("cr_unique_rank2", "unique", [t([[3.0, 1.0], [2.0, 1.0]])], {},
   lambda ts: [torch.unique(ts[0])])

# --- gradient goldens (issue 0008): per-op backward verification ---
# Each case: x (requires_grad, post-transfer leaf on MPS) -> op -> sum ->
# backward -> x.grad. The op list is the MPS-backward oracle. Inputs are
# chosen in-domain and chains avoid degenerate losses (softmax.sum() == 1
# would have an identically-zero gradient — a broken backward passes
# vacuously; it goes through softmax*softmax instead).
def grad_case(name, op, data, params=None, with_self=False, square_loss=False):
    x = torch.tensor(data, dtype=torch.float32, device=DEV).requires_grad_(True)
    if with_self:
        y = getattr(torch, op)(x, x)
    elif params:
        y = getattr(torch, op)(x, **params)
    else:
        y = getattr(torch, op)(x)
    loss = (y * y).sum() if square_loss else y.sum()
    loss.backward()
    cases.append({
        "name": name,
        "grad_op": op,
        "input": {"data": data, "dtype": "float32"},
        "params": params or {},
        "with_self": with_self,
        "square_loss": square_loss,
        "expect_grad": x.grad.cpu().tolist(),
    })

grad_case("ag_sin", "sin", [0.3, 1.0, 2.0])
grad_case("ag_exp", "exp", [-1.0, 0.0, 1.0])
grad_case("ag_sigmoid", "sigmoid", [-2.0, 0.0, 2.0])
grad_case("ag_tanh", "tanh", [-1.0, 0.0, 1.0])
grad_case("ag_sqrt", "sqrt", [1.0, 4.0, 9.0])
grad_case("ag_relu", "relu", [-2.0, 0.5, 3.0])
grad_case("ag_log", "log", [0.5, 1.0, 4.0])
grad_case("ag_mul_self", "mul", [1.0, 2.0, 3.0], with_self=True)
grad_case("ag_mm_self", "mm", [[1.0, 2.0], [3.0, 4.0]], with_self=True)
grad_case("ag_pow2", "pow", [1.0, 2.0, 3.0], params={"exponent": 2})
grad_case("ag_mean", "mean", [1.0, 2.0, 3.0, 4.0])
grad_case("ag_sum_dim", "sum", [[1.0, 2.0], [3.0, 4.0]], params={"dim": 1})
grad_case("ag_softmax_sq", "softmax", [1.0, 2.0, 3.0], params={"dim": 0},
          square_loss=True)

# --- nn module goldens (issue 0009 exp 2): explicit weights, forward +
# parameter gradients vs torch.nn.functional on MPS. Init parity is NOT
# golden-tested (different draw sequences); these cases LOAD weights.
def nn_linear_case(name, x, weight, bias, chain=None):
    """chain: optional list of activation names applied after linear."""
    xt = torch.tensor(x, dtype=torch.float32, device=DEV)
    wt = torch.tensor(weight, dtype=torch.float32, device=DEV).requires_grad_(True)
    bt = (torch.tensor(bias, dtype=torch.float32, device=DEV).requires_grad_(True)
          if bias is not None else None)
    y = torch.nn.functional.linear(xt, wt, bt)
    for act in (chain or []):
        y = getattr(torch, act)(y) if act != "gelu" else torch.nn.functional.gelu(y)
    y.sum().backward()
    cases.append({
        "name": name,
        "nn_linear_forward": True,
        "input": x,
        "weight": weight,
        "bias": bias,
        "chain": chain or [],
        "expect_output": y.detach().cpu().tolist(),
        "expect_weight_grad": wt.grad.cpu().tolist(),
        "expect_bias_grad": bt.grad.cpu().tolist() if bt is not None else None,
    })

W = [[0.5, -0.25], [1.5, 2.0], [-1.0, 0.75]]
B = [0.1, -0.2, 0.3]
X = [[1.0, 2.0], [3.0, -1.0]]
nn_linear_case("nn_linear_bias", X, W, B)
nn_linear_case("nn_linear_no_bias", X, W, None)
nn_linear_case("nn_linear_relu", X, W, B, chain=["relu"])
nn_linear_case("nn_linear_gelu", X, W, B, chain=["gelu"])
nn_linear_case("nn_linear_sigmoid", X, W, B, chain=["sigmoid"])

# --- losses (issue 0009 exp 3) vs torch.nn.functional on MPS ---
import torch.nn.functional as F

P2 = [[0.2, 0.8], [0.6, 0.4]]
T2 = [[0.0, 1.0], [1.0, 0.0]]
ok("loss_mse", "mse_loss", [t(P2), t(T2)], {},
   lambda ts: [F.mse_loss(ts[0], ts[1])])
ok("loss_mse_sum", "mse_loss", [t(P2), t(T2)], {"reduction": "sum"},
   lambda ts: [F.mse_loss(ts[0], ts[1], reduction="sum")])
ok("loss_mse_none", "mse_loss", [t(P2), t(T2)], {"reduction": "none"},
   lambda ts: [F.mse_loss(ts[0], ts[1], reduction="none")])
ok("loss_l1", "l1_loss", [t(P2), t(T2)], {},
   lambda ts: [F.l1_loss(ts[0], ts[1])])
ok("loss_smooth_l1", "smooth_l1_loss", [t(P2), t(T2)], {"beta": 0.5},
   lambda ts: [F.smooth_l1_loss(ts[0], ts[1], beta=0.5)])
ok("loss_huber", "huber_loss", [t(P2), t(T2)], {"delta": 0.5},
   lambda ts: [F.huber_loss(ts[0], ts[1], delta=0.5)])
LOGITS = [[2.0, 0.5, -1.0], [0.1, 1.5, 0.3]]
ok("loss_cross_entropy", "cross_entropy", [t(LOGITS), t([0, 2], "int64")], {},
   lambda ts: [F.cross_entropy(ts[0], ts[1])])
ok("loss_nll", "nll_loss",
   [t([[ -0.5, -1.2, -2.0], [-1.0, -0.3, -2.5]]), t([1, 0], "int64")], {},
   lambda ts: [F.nll_loss(ts[0], ts[1])])
ok("loss_bce", "binary_cross_entropy", [t([0.8, 0.2, 0.6]), t([1.0, 0.0, 1.0])], {},
   lambda ts: [F.binary_cross_entropy(ts[0], ts[1])])
ok("loss_bce_logits", "binary_cross_entropy_with_logits",
   [t([1.5, -0.5, 0.2]), t([1.0, 0.0, 1.0])], {},
   lambda ts: [F.binary_cross_entropy_with_logits(ts[0], ts[1])])
ok("loss_kl_div", "kl_div",
   [t([[-1.0, -0.5, -2.0]]), t([[0.3, 0.5, 0.2]])], {},
   lambda ts: [F.kl_div(ts[0], ts[1])])

# Gradient through mse_loss: a DISTINCT non-grad target; only x.grad compared.
x = torch.tensor([0.5, 1.5, -0.5], dtype=torch.float32, device=DEV).requires_grad_(True)
tgt = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32, device=DEV)
F.mse_loss(x, tgt).backward()
cases.append({
    "name": "loss_mse_grad",
    "grad_op": "mse_loss",
    "input": {"data": [0.5, 1.5, -0.5], "dtype": "float32"},
    "target": {"data": [1.0, 1.0, 0.0], "dtype": "float32"},
    "params": {},
    "with_self": False,
    "square_loss": False,
    "skip_sum": True,
    "expect_grad": x.grad.cpu().tolist(),
})

out = pathlib.Path(__file__).resolve().parent.parent / "nutorchd" / "tests" / "golden.json"







out.write_text(json.dumps(cases, indent=2) + "\n")
print(f"wrote {len(cases)} cases to {out}")
