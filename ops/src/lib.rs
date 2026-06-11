//! The declarative op table (issue 0005): one row per tensor operation,
//! read by the daemon (dispatch/validation) and the thin client (argument
//! parsing, usage, `torch ops`). Deliberately has no tch dependency.
//!
//! Table-level invariant: variadic-tensor ops take ALL non-tensor parameters
//! as flags, never trailing positionals — with unbounded tensor slots there
//! is no way to tell where tensors end and a positional scalar begins.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    Exactly(usize),
    AtLeast(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// Integer (e.g. --dim 0, seed)
    Int,
    /// Float (e.g. --rtol 1e-5)
    Float,
    /// Number, int or float (e.g. a fill value, an exponent)
    Scalar,
    /// JSON list of integers (e.g. a shape: '[2,3]')
    IntList,
    /// Presence-only flag (e.g. --keepdim, --descending)
    Bool,
    /// String (e.g. --dtype float32; einsum's equation later)
    Str,
    /// A number (scalar) OR a tensor handle string (issue 0005 exp 5).
    HandleOrScalar,
}

#[derive(Debug, Clone, Copy)]
pub struct ParamSpec {
    pub name: &'static str,
    pub kind: ParamKind,
    /// Positional params follow the tensor slots, in spec order; the rest
    /// are flags (`--name value`, or presence-only for Bool).
    pub positional: bool,
    pub required: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultKind {
    /// N new registry handles (1 for most ops; sort returns 2).
    Handles(usize),
    /// Output count depends on params (max/min/median: 1 without --dim,
    /// values+indices with it). Issue 0005 exp 3 spec extension.
    VariableHandles,
    /// A plain JSON value (e.g. allclose's bool).
    Value,
    /// Nothing (e.g. manual_seed).
    None,
}

#[derive(Debug, Clone, Copy)]
pub struct OpSpec {
    pub name: &'static str,
    pub category: &'static str,
    pub tensors: Arity,
    pub params: &'static [ParamSpec],
    pub results: ResultKind,
    /// Elementwise ops that follow PyTorch broadcasting get a Rust-side
    /// broadcastability pre-check for quality errors.
    pub broadcasts: bool,
    pub summary: &'static str,
}

const fn flag(name: &'static str, kind: ParamKind) -> ParamSpec {
    ParamSpec {
        name,
        kind,
        positional: false,
        required: false,
    }
}

const fn pos(name: &'static str, kind: ParamKind) -> ParamSpec {
    ParamSpec {
        name,
        kind,
        positional: true,
        required: true,
    }
}

const fn req_flag(name: &'static str, kind: ParamKind) -> ParamSpec {
    ParamSpec {
        name,
        kind,
        positional: false,
        required: true,
    }
}

const fn unary(name: &'static str, summary: &'static str) -> OpSpec {
    OpSpec {
        name,
        category: "pointwise",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary,
    }
}

const REDUCTION_ONLY: &[ParamSpec] = &[flag("reduction", ParamKind::Str)];

const fn loss(name: &'static str, summary: &'static str) -> OpSpec {
    OpSpec {
        name,
        category: "loss",
        tensors: Arity::Exactly(2),
        params: REDUCTION_ONLY,
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary,
    }
}

const fn binary_compare(name: &'static str, summary: &'static str) -> OpSpec {
    OpSpec {
        name,
        category: "comparison",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary,
    }
}

const fn unary_predicate(name: &'static str, summary: &'static str) -> OpSpec {
    OpSpec {
        name,
        category: "comparison",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary,
    }
}

const fn binary_bcast(name: &'static str, summary: &'static str) -> OpSpec {
    OpSpec {
        name,
        category: "pointwise",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary,
    }
}

pub static OPS: &[OpSpec] = &[
    OpSpec {
        name: "add",
        category: "pointwise",
        tensors: Arity::Exactly(2),
        params: &[flag("alpha", ParamKind::Scalar)],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary: "a + alpha*b (broadcasting; --alpha default 1)",
    },
    OpSpec {
        name: "sub",
        category: "pointwise",
        tensors: Arity::Exactly(2),
        params: &[flag("alpha", ParamKind::Scalar)],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary: "a - alpha*b (broadcasting; --alpha default 1)",
    },
    OpSpec {
        name: "sin",
        category: "pointwise",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "elementwise sine",
    },
    OpSpec {
        name: "pow",
        category: "pointwise",
        tensors: Arity::Exactly(1),
        params: &[pos("exponent", ParamKind::HandleOrScalar)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "elementwise power (scalar or tensor exponent)",
    },
    OpSpec {
        name: "clamp",
        category: "pointwise",
        tensors: Arity::Exactly(1),
        params: &[
            flag("min", ParamKind::HandleOrScalar),
            flag("max", ParamKind::HandleOrScalar),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "clamp into [min, max] (scalar or tensor bounds; one required)",
    },
    OpSpec {
        name: "sum",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "sum over all elements, or along --dim",
    },
    OpSpec {
        name: "mean",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "mean over all elements, or along --dim (float32, v1 fidelity)",
    },
    OpSpec {
        name: "eq",
        category: "comparison",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary: "elementwise equality (returns a Bool tensor)",
    },
    OpSpec {
        name: "allclose",
        category: "comparison",
        tensors: Arity::Exactly(2),
        params: &[
            flag("rtol", ParamKind::Float),
            flag("atol", ParamKind::Float),
        ],
        results: ResultKind::Value,
        broadcasts: true,
        summary: "true if all elements are close (returns a JSON bool)",
    },
    OpSpec {
        name: "sort",
        category: "comparison",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("descending", ParamKind::Bool),
        ],
        results: ResultKind::Handles(2),
        broadcasts: false,
        summary: "sort along --dim (default last); returns values and indices",
    },
    OpSpec {
        name: "mm",
        category: "linalg",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "matrix multiply of two 2-D tensors",
    },
    OpSpec {
        name: "cat",
        category: "shape",
        tensors: Arity::AtLeast(2),
        params: &[flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "concatenate tensors along --dim (default 0)",
    },
    OpSpec {
        name: "full",
        category: "creation",
        tensors: Arity::Exactly(0),
        params: &[
            pos("shape", ParamKind::IntList),
            pos("value", ParamKind::Scalar),
            flag("dtype", ParamKind::Str),
            flag("requires_grad", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "a tensor of the given shape filled with a value",
    },
    OpSpec {
        name: "randn",
        category: "creation",
        tensors: Arity::Exactly(0),
        params: &[
            pos("shape", ParamKind::IntList),
            flag("dtype", ParamKind::Str),
            flag("requires_grad", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "standard-normal random tensor (float kinds only)",
    },
    // --- pointwise sweep (issue 0005 exp 2) ---
    unary("abs", "elementwise absolute value"),
    unary("acos", "elementwise arccosine"),
    unary("acosh", "elementwise inverse hyperbolic cosine"),
    unary("asin", "elementwise arcsine"),
    unary("asinh", "elementwise inverse hyperbolic sine"),
    unary("atan", "elementwise arctangent"),
    unary("atanh", "elementwise inverse hyperbolic tangent"),
    unary("ceil", "elementwise ceiling"),
    unary("cos", "elementwise cosine"),
    unary("cosh", "elementwise hyperbolic cosine"),
    unary("deg2rad", "degrees to radians"),
    unary("digamma", "elementwise digamma"),
    unary("erf", "elementwise error function"),
    unary("erfc", "elementwise complementary error function"),
    unary("exp", "elementwise e^x"),
    unary("exp2", "elementwise 2^x"),
    unary("expm1", "elementwise e^x - 1"),
    unary("floor", "elementwise floor"),
    unary("frac", "elementwise fractional part"),
    unary("i0", "elementwise modified Bessel function I0"),
    unary("lgamma", "elementwise log-gamma"),
    unary("log", "elementwise natural log"),
    unary("log10", "elementwise log base 10"),
    unary("log1p", "elementwise log(1+x)"),
    unary("log2", "elementwise log base 2"),
    unary("logit", "elementwise logit (inverse sigmoid)"),
    unary("neg", "elementwise negation"),
    unary("rad2deg", "radians to degrees"),
    unary("reciprocal", "elementwise 1/x"),
    unary("relu", "elementwise max(x, 0)"),
    unary("round", "elementwise round to nearest"),
    unary("rsqrt", "elementwise 1/sqrt(x)"),
    unary("sgn", "elementwise sign (complex-aware)"),
    unary("sigmoid", "elementwise sigmoid"),
    unary("sign", "elementwise sign"),
    unary("sinc", "elementwise normalized sinc"),
    unary("sinh", "elementwise hyperbolic sine"),
    unary("sqrt", "elementwise square root"),
    unary("square", "elementwise x^2"),
    unary("tan", "elementwise tangent"),
    unary("tanh", "elementwise hyperbolic tangent"),
    unary("trunc", "elementwise truncation toward zero"),
    OpSpec {
        name: "softmax",
        category: "pointwise",
        tensors: Arity::Exactly(1),
        params: &[req_flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "softmax along --dim (float32)",
    },
    OpSpec {
        name: "log_softmax",
        category: "pointwise",
        tensors: Arity::Exactly(1),
        params: &[req_flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "log-softmax along --dim (float32)",
    },
    OpSpec {
        name: "nan_to_num",
        category: "pointwise",
        tensors: Arity::Exactly(1),
        params: &[
            flag("nan", ParamKind::Float),
            flag("posinf", ParamKind::Float),
            flag("neginf", ParamKind::Float),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "replace NaN/inf (--nan/--posinf/--neginf)",
    },
    binary_bcast("mul", "elementwise product (broadcasting)"),
    binary_bcast("div", "elementwise true division (broadcasting)"),
    binary_bcast("maximum", "elementwise maximum (broadcasting)"),
    binary_bcast("minimum", "elementwise minimum (broadcasting)"),
    binary_bcast("atan2", "elementwise atan2(a, b) (broadcasting)"),
    binary_bcast("fmod", "elementwise C-style remainder (broadcasting)"),
    binary_bcast(
        "remainder",
        "elementwise Python-style remainder (broadcasting)",
    ),
    binary_bcast("floor_divide", "elementwise floor division (broadcasting)"),
    binary_bcast("hypot", "elementwise hypotenuse (broadcasting)"),
    binary_bcast("copysign", "magnitude of a, sign of b (broadcasting)"),
    binary_bcast("xlogy", "elementwise x*log(y) (broadcasting)"),
    binary_bcast("logaddexp", "elementwise log(e^a + e^b) (broadcasting)"),
    // --- reductions + comparison sweep (issue 0005 exp 3) ---
    OpSpec {
        name: "prod",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "product over all elements, or along --dim",
    },
    OpSpec {
        name: "amax",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "max values over all elements, or along --dim",
    },
    OpSpec {
        name: "amin",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "min values over all elements, or along --dim",
    },
    OpSpec {
        name: "max",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::VariableHandles,
        broadcasts: false,
        summary: "max of all elements; with --dim also returns indices",
    },
    OpSpec {
        name: "min",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::VariableHandles,
        broadcasts: false,
        summary: "min of all elements; with --dim also returns indices",
    },
    OpSpec {
        name: "median",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::VariableHandles,
        broadcasts: false,
        summary: "median of all elements; with --dim also returns indices",
    },
    OpSpec {
        name: "argmax",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "index of the max, overall or along --dim",
    },
    OpSpec {
        name: "argmin",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "index of the min, overall or along --dim",
    },
    OpSpec {
        name: "all",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "true if all elements are true (Bool tensor)",
    },
    OpSpec {
        name: "any",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "true if any element is true (Bool tensor)",
    },
    OpSpec {
        name: "std",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
            flag("correction", ParamKind::Int),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "standard deviation (--correction, default 1)",
    },
    OpSpec {
        name: "var",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
            flag("correction", ParamKind::Int),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "variance (--correction, default 1)",
    },
    OpSpec {
        name: "nansum",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "sum treating NaN as zero",
    },
    OpSpec {
        name: "logsumexp",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            req_flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "log(sum(exp(x))) along --dim",
    },
    OpSpec {
        name: "count_nonzero",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "count of nonzero elements, overall or along --dim",
    },
    OpSpec {
        name: "cumsum",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[req_flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "cumulative sum along --dim",
    },
    OpSpec {
        name: "cumprod",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[req_flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "cumulative product along --dim",
    },
    OpSpec {
        name: "norm",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[
            flag("p", ParamKind::Float),
            flag("dim", ParamKind::Int),
            flag("keepdim", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "p-norm (--p default 2), overall or along --dim",
    },
    binary_compare("gt", "elementwise a > b (Bool, broadcasting)"),
    binary_compare("lt", "elementwise a < b (Bool, broadcasting)"),
    binary_compare("ge", "elementwise a >= b (Bool, broadcasting)"),
    binary_compare("le", "elementwise a <= b (Bool, broadcasting)"),
    binary_compare("ne", "elementwise a != b (Bool, broadcasting)"),
    binary_compare(
        "logical_and",
        "elementwise logical AND (Bool, broadcasting)",
    ),
    binary_compare("logical_or", "elementwise logical OR (Bool, broadcasting)"),
    binary_compare(
        "logical_xor",
        "elementwise logical XOR (Bool, broadcasting)",
    ),
    OpSpec {
        name: "isclose",
        category: "comparison",
        tensors: Arity::Exactly(2),
        params: &[
            flag("rtol", ParamKind::Float),
            flag("atol", ParamKind::Float),
        ],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary: "elementwise closeness (Bool; --rtol/--atol)",
    },
    unary_predicate("isnan", "elementwise NaN test (Bool)"),
    unary_predicate("isinf", "elementwise infinity test (Bool)"),
    unary_predicate("isfinite", "elementwise finiteness test (Bool)"),
    unary_predicate("isposinf", "elementwise +inf test (Bool)"),
    unary_predicate("isneginf", "elementwise -inf test (Bool)"),
    unary_predicate("logical_not", "elementwise logical NOT (Bool)"),
    OpSpec {
        name: "equal",
        category: "comparison",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Value,
        broadcasts: false,
        summary: "whole-tensor equality (returns a JSON bool)",
    },
    OpSpec {
        name: "topk",
        category: "comparison",
        tensors: Arity::Exactly(1),
        params: &[
            pos("k", ParamKind::Int),
            flag("dim", ParamKind::Int),
            flag("smallest", ParamKind::Bool),
        ],
        results: ResultKind::Handles(2),
        broadcasts: false,
        summary: "top-k values+indices (--smallest = PyTorch largest=False, a nutorch-ism)",
    },
    OpSpec {
        name: "argsort",
        category: "comparison",
        tensors: Arity::Exactly(1),
        params: &[
            flag("dim", ParamKind::Int),
            flag("descending", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "indices that would sort along --dim (default last)",
    },
    // --- linalg + shape/indexing sweep (issue 0005 exp 4) ---
    OpSpec {
        name: "matmul",
        category: "linalg",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "general matrix product (batched, PyTorch broadcasting)",
    },
    OpSpec {
        name: "bmm",
        category: "linalg",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "batched matrix multiply of two 3-D tensors",
    },
    OpSpec {
        name: "dot",
        category: "linalg",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "dot product of two 1-D tensors",
    },
    OpSpec {
        name: "outer",
        category: "linalg",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "outer product of two 1-D tensors",
    },
    OpSpec {
        name: "einsum",
        category: "linalg",
        tensors: Arity::AtLeast(1),
        params: &[req_flag("equation", ParamKind::Str)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "Einstein summation over --equation",
    },
    OpSpec {
        name: "tril",
        category: "linalg",
        tensors: Arity::Exactly(1),
        params: &[flag("diagonal", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "lower triangle (--diagonal offset)",
    },
    OpSpec {
        name: "triu",
        category: "linalg",
        tensors: Arity::Exactly(1),
        params: &[flag("diagonal", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "upper triangle (--diagonal offset)",
    },
    OpSpec {
        name: "diag",
        category: "linalg",
        tensors: Arity::Exactly(1),
        params: &[flag("diagonal", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "diagonal of a matrix, or diagonal matrix from a vector",
    },
    OpSpec {
        name: "trace",
        category: "linalg",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "sum of the main diagonal of a 2-D tensor",
    },
    OpSpec {
        name: "det",
        category: "linalg",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "determinant of a square matrix",
    },
    OpSpec {
        name: "inverse",
        category: "linalg",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "inverse of a square matrix",
    },
    OpSpec {
        name: "svd",
        category: "linalg",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(3),
        broadcasts: false,
        summary: "singular value decomposition (U, S, V)",
    },
    OpSpec {
        name: "solve",
        category: "linalg",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "solve AX = B for X",
    },
    OpSpec {
        name: "reshape",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[pos("shape", ParamKind::IntList)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "reshape to the given shape (-1 infers one dim)",
    },
    OpSpec {
        name: "permute",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[pos("dims", ParamKind::IntList)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "permute dimensions",
    },
    OpSpec {
        name: "transpose",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[pos("dim0", ParamKind::Int), pos("dim1", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "swap two dimensions",
    },
    OpSpec {
        name: "t",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "transpose a 2-D tensor",
    },
    OpSpec {
        name: "squeeze",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "drop size-1 dims (all, or --dim)",
    },
    OpSpec {
        name: "unsqueeze",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[pos("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "insert a size-1 dim",
    },
    OpSpec {
        name: "flatten",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[
            flag("start_dim", ParamKind::Int),
            flag("end_dim", ParamKind::Int),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "flatten dims (--start_dim/--end_dim)",
    },
    OpSpec {
        name: "stack",
        category: "shape",
        tensors: Arity::AtLeast(2),
        params: &[flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "stack tensors along a NEW --dim (default 0)",
    },
    OpSpec {
        name: "split",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[
            pos("split_size", ParamKind::Int),
            flag("dim", ParamKind::Int),
        ],
        results: ResultKind::VariableHandles,
        broadcasts: false,
        summary: "split into chunks of split_size along --dim",
    },
    OpSpec {
        name: "chunk",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[pos("chunks", ParamKind::Int), flag("dim", ParamKind::Int)],
        results: ResultKind::VariableHandles,
        broadcasts: false,
        summary: "split into N chunks along --dim",
    },
    OpSpec {
        name: "gather",
        category: "shape",
        tensors: Arity::Exactly(2),
        params: &[req_flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "gather values along --dim using an int64 index tensor",
    },
    OpSpec {
        name: "index_select",
        category: "shape",
        tensors: Arity::Exactly(2),
        params: &[req_flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "select rows/cols along --dim by an int64 index tensor",
    },
    OpSpec {
        name: "masked_select",
        category: "shape",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "select by mask (numeric mask cast via != 0, a nutorch-ism)",
    },
    OpSpec {
        name: "where",
        category: "shape",
        tensors: Arity::Exactly(3),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary: "cond ? x : y (numeric cond cast via != 0, a nutorch-ism)",
    },
    OpSpec {
        name: "narrow",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[
            pos("dim", ParamKind::Int),
            pos("start", ParamKind::Int),
            pos("length", ParamKind::Int),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "slice: length elements from start along dim",
    },
    OpSpec {
        name: "flip",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[pos("dims", ParamKind::IntList)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "reverse along the given dims",
    },
    OpSpec {
        name: "roll",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[
            pos("shifts", ParamKind::IntList),
            flag("dims", ParamKind::IntList),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "roll elements by shifts (optionally along --dims)",
    },
    OpSpec {
        name: "repeat",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[pos("repeats", ParamKind::IntList)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "tile the tensor by repeats per dim",
    },
    OpSpec {
        name: "repeat_interleave",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[pos("repeats", ParamKind::Int), flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "repeat each element N times (optionally along --dim)",
    },
    OpSpec {
        name: "movedim",
        category: "shape",
        tensors: Arity::Exactly(1),
        params: &[
            pos("source", ParamKind::Int),
            pos("destination", ParamKind::Int),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "move a dim to a new position",
    },
    // --- creation + remainder sweep (issue 0005 exp 5) ---
    OpSpec {
        name: "zeros",
        category: "creation",
        tensors: Arity::Exactly(0),
        params: &[
            pos("shape", ParamKind::IntList),
            flag("dtype", ParamKind::Str),
            flag("requires_grad", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "a tensor of zeros",
    },
    OpSpec {
        name: "ones",
        category: "creation",
        tensors: Arity::Exactly(0),
        params: &[
            pos("shape", ParamKind::IntList),
            flag("dtype", ParamKind::Str),
            flag("requires_grad", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "a tensor of ones",
    },
    OpSpec {
        name: "eye",
        category: "creation",
        tensors: Arity::Exactly(0),
        params: &[pos("n", ParamKind::Int), flag("m", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "identity matrix (n x n, or n x --m)",
    },
    OpSpec {
        name: "arange",
        category: "creation",
        tensors: Arity::Exactly(0),
        params: &[
            pos("end", ParamKind::Scalar),
            flag("start", ParamKind::Scalar),
            flag("step", ParamKind::Scalar),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "range [--start, end) by --step (CLI reshape of PyTorch overloads)",
    },
    OpSpec {
        name: "linspace",
        category: "creation",
        tensors: Arity::Exactly(0),
        params: &[
            pos("start", ParamKind::Scalar),
            pos("end", ParamKind::Scalar),
            pos("steps", ParamKind::Int),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "steps evenly spaced points in [start, end]",
    },
    OpSpec {
        name: "rand",
        category: "creation",
        tensors: Arity::Exactly(0),
        params: &[
            pos("shape", ParamKind::IntList),
            flag("requires_grad", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "uniform [0,1) random tensor (seeded CPU generator)",
    },
    OpSpec {
        name: "randint",
        category: "creation",
        tensors: Arity::Exactly(0),
        params: &[
            pos("high", ParamKind::Int),
            pos("shape", ParamKind::IntList),
            flag("low", ParamKind::Int),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "random int64s in [--low, high) (seeded CPU generator)",
    },
    OpSpec {
        name: "zeros_like",
        category: "creation",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "zeros with the input's shape and dtype",
    },
    OpSpec {
        name: "ones_like",
        category: "creation",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "ones with the input's shape and dtype",
    },
    OpSpec {
        name: "full_like",
        category: "creation",
        tensors: Arity::Exactly(1),
        params: &[pos("value", ParamKind::Scalar)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "a value-filled tensor with the input's shape and dtype",
    },
    OpSpec {
        name: "rand_like",
        category: "creation",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "uniform random with the input's shape (seeded CPU generator)",
    },
    OpSpec {
        name: "randn_like",
        category: "creation",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "normal random with the input's shape (seeded CPU generator)",
    },
    OpSpec {
        name: "lerp",
        category: "pointwise",
        tensors: Arity::Exactly(2),
        params: &[pos("weight", ParamKind::HandleOrScalar)],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary: "a + weight*(b - a) (scalar or tensor weight)",
    },
    OpSpec {
        name: "addcmul",
        category: "pointwise",
        tensors: Arity::Exactly(3),
        params: &[flag("value", ParamKind::Scalar)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "a + value * b * c",
    },
    OpSpec {
        name: "addcdiv",
        category: "pointwise",
        tensors: Arity::Exactly(3),
        params: &[flag("value", ParamKind::Scalar)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "a + value * b / c",
    },
    OpSpec {
        name: "cross",
        category: "linalg",
        tensors: Arity::Exactly(2),
        params: &[flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "vector cross product along --dim",
    },
    OpSpec {
        name: "kron",
        category: "linalg",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "Kronecker product",
    },
    OpSpec {
        name: "tensordot",
        category: "linalg",
        tensors: Arity::Exactly(2),
        params: &[flag("dims", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "tensor contraction over the last/first --dims dims (default 2)",
    },
    OpSpec {
        name: "take_along_dim",
        category: "shape",
        tensors: Arity::Exactly(2),
        params: &[req_flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "gather along --dim with a broadcastable int64 index",
    },
    OpSpec {
        name: "searchsorted",
        category: "comparison",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "insertion indices: searchsorted(sorted_seq, values)",
    },
    OpSpec {
        name: "bucketize",
        category: "comparison",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "bucket indices: bucketize(values, boundaries)",
    },
    OpSpec {
        name: "msort",
        category: "comparison",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "sort along the first dimension (values only)",
    },
    OpSpec {
        name: "diff",
        category: "reduction",
        tensors: Arity::Exactly(1),
        params: &[flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "forward differences along --dim (default last)",
    },
    OpSpec {
        name: "scatter",
        category: "shape",
        tensors: Arity::Exactly(3),
        params: &[req_flag("dim", ParamKind::Int)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "non-inplace scatter: input, int64 index, src along --dim",
    },
    OpSpec {
        name: "bitwise_and",
        category: "pointwise",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary: "bitwise AND of int tensors (broadcasting)",
    },
    OpSpec {
        name: "bitwise_or",
        category: "pointwise",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary: "bitwise OR of int tensors (broadcasting)",
    },
    OpSpec {
        name: "bitwise_xor",
        category: "pointwise",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary: "bitwise XOR of int tensors (broadcasting)",
    },
    OpSpec {
        name: "bitwise_not",
        category: "pointwise",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "bitwise NOT of an int tensor",
    },
    OpSpec {
        name: "bitwise_left_shift",
        category: "pointwise",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary: "left shift of int tensors (broadcasting)",
    },
    OpSpec {
        name: "bitwise_right_shift",
        category: "pointwise",
        tensors: Arity::Exactly(2),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: true,
        summary: "right shift of int tensors (broadcasting)",
    },
    OpSpec {
        name: "unique",
        category: "comparison",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "sorted unique values",
    },
    // --- autograd surface (issue 0008) ---
    OpSpec {
        name: "backward",
        category: "autograd",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::None,
        broadcasts: false,
        summary: "backpropagate from a scalar loss (gradients accumulate)",
    },
    OpSpec {
        name: "grad",
        category: "autograd",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "snapshot of a tensor's accumulated gradient",
    },
    OpSpec {
        name: "detach",
        category: "autograd",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "a graph-free reference to the same data",
    },
    OpSpec {
        name: "zero_grad",
        category: "autograd",
        tensors: Arity::Exactly(1),
        params: &[],
        results: ResultKind::None,
        broadcasts: false,
        summary: "zero a tensor's accumulated gradient in place",
    },
    // --- losses (issue 0009 exp 3) ---
    loss("mse_loss", "mean squared error (--reduction mean|sum|none)"),
    loss("l1_loss", "mean absolute error (--reduction)"),
    OpSpec {
        name: "smooth_l1_loss",
        category: "loss",
        tensors: Arity::Exactly(2),
        params: &[
            flag("reduction", ParamKind::Str),
            flag("beta", ParamKind::Float),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "smooth L1 loss (--beta, default 1.0)",
    },
    OpSpec {
        name: "huber_loss",
        category: "loss",
        tensors: Arity::Exactly(2),
        params: &[
            flag("reduction", ParamKind::Str),
            flag("delta", ParamKind::Float),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "Huber loss (--delta, default 1.0)",
    },
    loss(
        "cross_entropy",
        "cross entropy over logits vs int64 class indices",
    ),
    loss(
        "nll_loss",
        "negative log likelihood (log-prob inputs, int64 targets)",
    ),
    loss("binary_cross_entropy", "BCE over probabilities in [0,1]"),
    loss(
        "binary_cross_entropy_with_logits",
        "BCE over logits (the stable form)",
    ),
    OpSpec {
        name: "kl_div",
        category: "loss",
        tensors: Arity::Exactly(2),
        params: &[
            flag("reduction", ParamKind::Str),
            flag("log_target", ParamKind::Bool),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "KL divergence (--log_target if target is log-space)",
    },
    OpSpec {
        name: "manual_seed",
        category: "utility",
        tensors: Arity::Exactly(0),
        params: &[pos("seed", ParamKind::Int)],
        results: ResultKind::None,
        broadcasts: false,
        summary: "seed the random number generator",
    },
];

pub fn find(name: &str) -> Option<&'static OpSpec> {
    OPS.iter().find(|spec| spec.name == name)
}

/// Categories in display order (each category appears once).
pub fn categories() -> Vec<&'static str> {
    let mut seen = Vec::new();
    for spec in OPS {
        if !seen.contains(&spec.category) {
            seen.push(spec.category);
        }
    }
    seen
}

impl OpSpec {
    pub fn positional_params(&self) -> impl Iterator<Item = &'static ParamSpec> {
        self.params.iter().filter(|p| p.positional)
    }

    pub fn usage(&self) -> String {
        let mut usage = format!("usage: torch {}", self.name);
        match self.tensors {
            Arity::Exactly(n) => {
                for i in 0..n {
                    usage.push_str(&format!(" <t{}>", i + 1));
                }
            }
            Arity::AtLeast(n) => usage.push_str(&format!(" <t1>... (at least {n})")),
        }
        for param in self.positional_params() {
            usage.push_str(&format!(" <{}>", param.name));
        }
        for param in self.params.iter().filter(|p| !p.positional) {
            match param.kind {
                ParamKind::Bool => usage.push_str(&format!(" [--{}]", param.name)),
                _ => usage.push_str(&format!(" [--{} <{:?}>]", param.name, param.kind)),
            }
        }
        usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_names_are_unique_and_findable() {
        for spec in OPS {
            assert_eq!(find(spec.name).unwrap().name, spec.name);
        }
        let mut names: Vec<_> = OPS.iter().map(|s| s.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), OPS.len(), "duplicate op names");
    }

    #[test]
    fn variadic_ops_have_no_positional_params() {
        for spec in OPS {
            if matches!(spec.tensors, Arity::AtLeast(_)) {
                assert_eq!(
                    spec.positional_params().count(),
                    0,
                    "{}: variadic ops take all non-tensor params as flags",
                    spec.name
                );
            }
        }
    }

    #[test]
    fn table_is_at_least_the_experiment_one_baseline() {
        assert!(OPS.len() >= 15);
    }
}
