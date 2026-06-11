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
        params: &[pos("exponent", ParamKind::Scalar)],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "elementwise power with a scalar exponent",
    },
    OpSpec {
        name: "clamp",
        category: "pointwise",
        tensors: Arity::Exactly(1),
        params: &[
            flag("min", ParamKind::Scalar),
            flag("max", ParamKind::Scalar),
        ],
        results: ResultKind::Handles(1),
        broadcasts: false,
        summary: "clamp values into [min, max] (at least one bound required)",
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
