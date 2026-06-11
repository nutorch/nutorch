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

out = pathlib.Path(__file__).resolve().parent.parent / "nutorchd" / "tests" / "golden.json"

out.write_text(json.dumps(cases, indent=2) + "\n")
print(f"wrote {len(cases)} cases to {out}")
