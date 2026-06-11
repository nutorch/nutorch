//! Neural-network modules (issue 0009): daemon-resident objects composed
//! from the shell. Own parameter management, no tch VarStore — the
//! decision record lives in issues/0009-nn-optim/02-module-foundation.md
//! (composition over handles needs parameter identity under our control;
//! optimizers will hold shallow-clone references to these tensors).

use tch::Tensor;

use crate::convert::tch_error;

pub enum NnModule {
    Linear {
        weight: Tensor,
        bias: Option<Tensor>,
    },
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Sequential {
        children: Vec<NnModule>,
    },
    Conv1d {
        weight: Tensor,
        bias: Option<Tensor>,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
    },
    Conv2d {
        weight: Tensor,
        bias: Option<Tensor>,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
    },
    ConvTranspose2d {
        weight: Tensor,
        bias: Option<Tensor>,
        stride: i64,
        padding: i64,
        output_padding: i64,
        groups: i64,
        dilation: i64,
    },
    Embedding {
        weight: Tensor,
    },
    LayerNorm {
        shape: Vec<i64>,
        weight: Tensor,
        bias: Tensor,
        eps: f64,
    },
    BatchNorm {
        weight: Tensor,
        bias: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        /// int64 scalar buffer, PyTorch state_dict parity (issue 0009
        /// exp 6) — incremented per train-mode forward.
        num_batches_tracked: Tensor,
        eps: f64,
        momentum: f64,
        training: bool,
    },
    GroupNorm {
        num_groups: i64,
        weight: Tensor,
        bias: Tensor,
        eps: f64,
    },
    Dropout {
        p: f64,
        training: bool,
    },
    LeakyRelu {
        slope: f64,
    },
    Softmax {
        dim: i64,
    },
    MaxPool2d {
        kernel: i64,
        stride: i64,
        padding: i64,
    },
    AvgPool2d {
        kernel: i64,
        stride: i64,
        padding: i64,
    },
    Flatten {
        start_dim: i64,
        end_dim: i64,
    },
}

impl NnModule {
    pub fn kind_name(&self) -> &'static str {
        match self {
            NnModule::Linear { .. } => "linear",
            NnModule::Relu => "relu",
            NnModule::Sigmoid => "sigmoid",
            NnModule::Tanh => "tanh",
            NnModule::Gelu => "gelu",
            NnModule::Sequential { .. } => "sequential",
            NnModule::Conv1d { .. } => "conv1d",
            NnModule::Conv2d { .. } => "conv2d",
            NnModule::ConvTranspose2d { .. } => "conv_transpose2d",
            NnModule::Embedding { .. } => "embedding",
            NnModule::LayerNorm { .. } => "layer_norm",
            NnModule::BatchNorm { .. } => "batch_norm",
            NnModule::GroupNorm { .. } => "group_norm",
            NnModule::Dropout { .. } => "dropout",
            NnModule::LeakyRelu { .. } => "leaky_relu",
            NnModule::Softmax { .. } => "softmax",
            NnModule::MaxPool2d { .. } => "max_pool2d",
            NnModule::AvgPool2d { .. } => "avg_pool2d",
            NnModule::Flatten { .. } => "flatten",
        }
    }

    /// Train/eval mode (issue 0009 exp 5): meaningful for Dropout and
    /// BatchNorm; recurses through Sequential like PyTorch's .train().
    pub fn set_training(&mut self, training: bool) {
        match self {
            NnModule::Dropout { training: t, .. } => *t = training,
            NnModule::BatchNorm { training: t, .. } => *t = training,
            NnModule::Sequential { children } => {
                for child in children {
                    child.set_training(training);
                }
            }
            _ => {}
        }
    }

    pub fn is_training(&self) -> bool {
        match self {
            NnModule::Dropout { training, .. } => *training,
            NnModule::BatchNorm { training, .. } => *training,
            NnModule::Sequential { children } => children.iter().any(|c| c.is_training()),
            _ => true,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        match self {
            NnModule::Linear { weight, bias } => input
                .f_linear(weight, bias.as_ref())
                .map_err(|e| format!("linear forward: {}", tch_error(e))),
            NnModule::Relu => input.f_relu().map_err(tch_error),
            NnModule::Sigmoid => input.f_sigmoid().map_err(tch_error),
            NnModule::Tanh => input.f_tanh().map_err(tch_error),
            NnModule::Gelu => input.f_gelu("none").map_err(tch_error),
            NnModule::Sequential { children } => {
                let mut current = input.shallow_clone();
                for child in children {
                    current = child.forward(&current)?;
                }
                Ok(current)
            }
            NnModule::Conv1d {
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
            } => input
                .f_conv1d(
                    weight,
                    bias.as_ref(),
                    [*stride],
                    [*padding],
                    [*dilation],
                    *groups,
                )
                .map_err(tch_error),
            NnModule::Conv2d {
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
            } => input
                .f_conv2d(
                    weight,
                    bias.as_ref(),
                    [*stride, *stride],
                    [*padding, *padding],
                    [*dilation, *dilation],
                    *groups,
                )
                .map_err(tch_error),
            NnModule::ConvTranspose2d {
                weight,
                bias,
                stride,
                padding,
                output_padding,
                groups,
                dilation,
            } => input
                // NOTE: groups BEFORE dilation here — the reverse of conv2d.
                .f_conv_transpose2d(
                    weight,
                    bias.as_ref(),
                    [*stride, *stride],
                    [*padding, *padding],
                    [*output_padding, *output_padding],
                    *groups,
                    [*dilation, *dilation],
                )
                .map_err(tch_error),
            NnModule::Embedding { weight } => {
                // f_embedding is weight-first; padding_idx -1 == None.
                Tensor::f_embedding(weight, input, -1, false, false).map_err(tch_error)
            }
            NnModule::LayerNorm {
                shape,
                weight,
                bias,
                eps,
            } => input
                .f_layer_norm(&shape[..], Some(weight), Some(bias), *eps, false)
                .map_err(tch_error),
            NnModule::BatchNorm {
                weight,
                bias,
                running_mean,
                running_var,
                num_batches_tracked,
                eps,
                momentum,
                training,
            } => {
                if *training {
                    let mut counter = num_batches_tracked.shallow_clone();
                    let _ = counter.f_add_scalar_(1).map_err(tch_error)?;
                }
                input
                    // Running stats mutate IN PLACE during train-mode forward
                    // (libtorch interior mutability) — &self survives.
                    .f_batch_norm(
                        Some(weight),
                        Some(bias),
                        Some(running_mean),
                        Some(running_var),
                        *training,
                        *momentum,
                        *eps,
                        false,
                    )
                    .map_err(tch_error)
            }
            NnModule::GroupNorm {
                num_groups,
                weight,
                bias,
                eps,
            } => input
                // cudnn_enabled=true matches F.group_norm's passthrough of
                // torch.backends.cudnn.enabled (1-ULP kernel-path difference
                // otherwise — found by the golden).
                .f_group_norm(*num_groups, Some(weight), Some(bias), *eps, true)
                .map_err(tch_error),
            NnModule::Dropout { p, training } => {
                if !training || *p == 0.0 {
                    return Ok(input.shallow_clone());
                }
                if *p == 1.0 {
                    // Special-cased: the 1/(1-p) scaling would be inf×0=NaN.
                    return input.f_zeros_like().map_err(tch_error);
                }
                // Own-mask convention: tch's manual_seed cannot reach the
                // MPS generator, so the mask draws on the seeded CPU
                // generator (deterministic) and transfers.
                let mask = Tensor::f_rand(&input.size(), (tch::Kind::Float, tch::Device::Cpu))
                    .and_then(|r| r.f_ge(*p))
                    .and_then(|m| m.f_to_kind(tch::Kind::Float))
                    .and_then(|m| m.f_mul_scalar(1.0 / (1.0 - *p)))
                    .and_then(|m| m.f_to_device(input.device()))
                    .map_err(tch_error)?;
                input.f_mul(&mask).map_err(tch_error)
            }
            NnModule::LeakyRelu { slope } => {
                // tch's f_leaky_relu takes NO slope (0.01 baked in) — a
                // non-default slope would be silently ignored. Computed
                // manually: max(0,x) + slope*min(0,x).
                let positive = input.f_clamp_min(0.0).map_err(tch_error)?;
                let negative = input
                    .f_clamp_max(0.0)
                    .and_then(|n| n.f_mul_scalar(*slope))
                    .map_err(tch_error)?;
                positive.f_add(&negative).map_err(tch_error)
            }
            NnModule::Softmax { dim } => input.f_softmax(*dim, tch::Kind::Float).map_err(tch_error),
            NnModule::MaxPool2d {
                kernel,
                stride,
                padding,
            } => input
                .f_max_pool2d(
                    [*kernel, *kernel],
                    [*stride, *stride],
                    [*padding, *padding],
                    [1, 1],
                    false,
                )
                .map_err(tch_error),
            NnModule::AvgPool2d {
                kernel,
                stride,
                padding,
            } => input
                .f_avg_pool2d(
                    [*kernel, *kernel],
                    [*stride, *stride],
                    [*padding, *padding],
                    false,
                    true,
                    None,
                )
                .map_err(tch_error),
            NnModule::Flatten { start_dim, end_dim } => {
                input.f_flatten(*start_dim, *end_dim).map_err(tch_error)
            }
        }
    }

    /// Depth-first, weight before bias (PyTorch's .parameters() order).
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        self.collect_parameters(&mut params);
        params
    }

    fn collect_parameters<'a>(&'a self, params: &mut Vec<&'a Tensor>) {
        match self {
            NnModule::Linear { weight, bias }
            | NnModule::Conv1d { weight, bias, .. }
            | NnModule::Conv2d { weight, bias, .. }
            | NnModule::ConvTranspose2d { weight, bias, .. } => {
                params.push(weight);
                if let Some(bias) = bias {
                    params.push(bias);
                }
            }
            NnModule::Embedding { weight } => params.push(weight),
            // Norm weight/bias are parameters; running stats are BUFFERS
            // (not optimized), as in PyTorch.
            NnModule::LayerNorm { weight, bias, .. }
            | NnModule::BatchNorm { weight, bias, .. }
            | NnModule::GroupNorm { weight, bias, .. } => {
                params.push(weight);
                params.push(bias);
            }
            NnModule::Sequential { children } => {
                for child in children {
                    child.collect_parameters(params);
                }
            }
            _ => {}
        }
    }

    pub fn param_bytes(&self) -> u64 {
        self.parameters()
            .iter()
            .map(|t| t.numel() as u64 * t.kind().elt_size_in_bytes() as u64)
            .sum()
    }

    /// PyTorch state_dict naming (issue 0009 exp 6): leaf entries are
    /// weight/bias (+ buffer names); Sequential children prefix their
    /// index recursively. Includes BUFFERS (running stats, the batch
    /// counter) — state_dict does, parameters() does not.
    pub fn named_state(&self) -> Vec<(String, &Tensor)> {
        let mut entries = Vec::new();
        self.collect_named_state("", &mut entries);
        entries
    }

    fn collect_named_state<'a>(&'a self, prefix: &str, entries: &mut Vec<(String, &'a Tensor)>) {
        let name = |leaf: &str| {
            if prefix.is_empty() {
                leaf.to_string()
            } else {
                format!("{prefix}.{leaf}")
            }
        };
        match self {
            NnModule::Linear { weight, bias }
            | NnModule::Conv1d { weight, bias, .. }
            | NnModule::Conv2d { weight, bias, .. }
            | NnModule::ConvTranspose2d { weight, bias, .. } => {
                entries.push((name("weight"), weight));
                if let Some(bias) = bias {
                    entries.push((name("bias"), bias));
                }
            }
            NnModule::Embedding { weight } => entries.push((name("weight"), weight)),
            NnModule::LayerNorm { weight, bias, .. } | NnModule::GroupNorm { weight, bias, .. } => {
                entries.push((name("weight"), weight));
                entries.push((name("bias"), bias));
            }
            NnModule::BatchNorm {
                weight,
                bias,
                running_mean,
                running_var,
                num_batches_tracked,
                ..
            } => {
                entries.push((name("weight"), weight));
                entries.push((name("bias"), bias));
                entries.push((name("running_mean"), running_mean));
                entries.push((name("running_var"), running_var));
                entries.push((name("num_batches_tracked"), num_batches_tracked));
            }
            NnModule::Sequential { children } => {
                for (index, child) in children.iter().enumerate() {
                    let child_prefix = if prefix.is_empty() {
                        index.to_string()
                    } else {
                        format!("{prefix}.{index}")
                    };
                    child.collect_named_state(&child_prefix, entries);
                }
            }
            _ => {}
        }
    }

    /// load_state_dict semantics: every file key must match a module key
    /// with an identical shape; missing/unexpected/mismatched errors
    /// name the key, all-or-nothing (validated BEFORE any copy); values
    /// copy in place under no_grad so optimizer views and autograd
    /// leaves survive.
    pub fn load_state(&self, file: &[(String, Tensor)]) -> Result<(), String> {
        let targets = self.named_state();
        for (key, _) in &targets {
            if !file.iter().any(|(k, _)| k == key) {
                return Err(format!("load: missing key in file: {key}"));
            }
        }
        for (key, value) in file {
            let Some((_, target)) = targets.iter().find(|(k, _)| k == key) else {
                return Err(format!("load: unexpected key in file: {key}"));
            };
            if target.size() != value.size() {
                return Err(format!(
                    "load: shape mismatch for {key}: module {:?}, file {:?}",
                    target.size(),
                    value.size()
                ));
            }
        }
        tch::no_grad(|| -> Result<(), String> {
            for (key, value) in file {
                let (_, target) = targets.iter().find(|(k, _)| k == key).unwrap();
                let mut alias = target.shallow_clone();
                let on_device = value.f_to_device(target.device()).map_err(tch_error)?;
                alias.f_copy_(&on_device).map_err(tch_error)?;
            }
            Ok(())
        })
    }

    /// One line per fact, for `torch nn info`.
    pub fn describe(&self) -> Vec<String> {
        let params = self.parameters();
        let elements: i64 = params.iter().map(|t| t.numel() as i64).sum();
        let mut lines = vec![
            format!("kind: {}", self.kind_name()),
            format!(
                "parameters: {} tensor(s), {} element(s)",
                params.len(),
                elements
            ),
        ];
        match self {
            NnModule::Linear { weight, bias } => {
                let shape = weight.size();
                lines.push(format!(
                    "features: {} in, {} out, bias: {}",
                    shape[1],
                    shape[0],
                    bias.is_some()
                ));
            }
            NnModule::Sequential { children } => {
                let kinds: Vec<&str> = children.iter().map(|c| c.kind_name()).collect();
                lines.push(format!("children: {}", kinds.join(" -> ")));
            }
            _ => {}
        }
        if matches!(
            self,
            NnModule::Dropout { .. } | NnModule::BatchNorm { .. } | NnModule::Sequential { .. }
        ) {
            lines.push(format!("training: {}", self.is_training()));
        }
        lines
    }
}

/// Optimizer hyperparameters and per-parameter state (issue 0009 exp 4).
/// The update math mirrors PyTorch's ACTUAL op sequence where it matters —
/// notably `lerp_` for Adam's first moment (the textbook mul_+add_ form
/// diverges by 1 ULP under coupled weight decay; established empirically
/// in the design review) — because the goldens pin bitwise equality with
/// torch.optim on MPS.
pub enum OptimKind {
    Sgd {
        momentum: f64,
        dampening: f64,
        nesterov: bool,
    },
    Adam {
        beta1: f64,
        beta2: f64,
        eps: f64,
        /// Coupled L2 (classic Adam) vs decoupled (AdamW).
        decoupled: bool,
    },
    RmsProp {
        alpha: f64,
        eps: f64,
        momentum: f64,
    },
}

pub struct Optimizer {
    pub kind: OptimKind,
    pub lr: f64,
    pub weight_decay: f64,
    /// Shallow clones of the module's parameters (shared TensorImpl —
    /// in-place steps propagate to the module).
    params: Vec<Tensor>,
    /// One state slot per param: SGD momentum buffer (second unused) /
    /// Adam (m, v) / RMSprop (square_avg, momentum buffer).
    state: Vec<Option<(Tensor, Option<Tensor>)>>,
    step_count: i64,
}

fn deep_copy(t: &Tensor) -> Result<Tensor, String> {
    let mut copy = t.f_zeros_like().map_err(tch_error)?;
    copy.f_copy_(t).map_err(tch_error)?;
    Ok(copy)
}

impl Optimizer {
    pub fn new(kind: OptimKind, lr: f64, weight_decay: f64, params: Vec<Tensor>) -> Self {
        let state = params.iter().map(|_| None).collect();
        Optimizer {
            kind,
            lr,
            weight_decay,
            params,
            state,
            step_count: 0,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match &self.kind {
            OptimKind::Sgd { .. } => "sgd",
            OptimKind::Adam {
                decoupled: false, ..
            } => "adam",
            OptimKind::Adam {
                decoupled: true, ..
            } => "adamw",
            OptimKind::RmsProp { .. } => "rmsprop",
        }
    }

    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    pub fn state_bytes(&self) -> u64 {
        let one = |t: &Tensor| t.numel() as u64 * t.kind().elt_size_in_bytes() as u64;
        self.state
            .iter()
            .flatten()
            .map(|(a, b)| one(a) + b.as_ref().map(one).unwrap_or(0))
            .sum()
    }

    /// One optimization step, in place. The caller wraps in tch::no_grad.
    /// Params whose grad is undefined are SKIPPED (PyTorch behavior).
    pub fn step(&mut self) -> Result<(), String> {
        self.step_count += 1;
        let t = self.step_count;
        let lr = self.lr;
        let wd = self.weight_decay;
        for index in 0..self.params.len() {
            let param = self.params[index].shallow_clone();
            let grad = param.f_grad().map_err(tch_error)?;
            if !grad.defined() {
                continue;
            }
            let grad = grad.f_detach().map_err(tch_error)?;
            match &self.kind {
                OptimKind::Sgd {
                    momentum,
                    dampening,
                    nesterov,
                } => {
                    let (momentum, dampening, nesterov) = (*momentum, *dampening, *nesterov);
                    let detached = param.f_detach().map_err(tch_error)?;
                    let grad = if wd != 0.0 {
                        grad.f_add(&detached.f_mul_scalar(wd).map_err(tch_error)?)
                            .map_err(tch_error)?
                    } else {
                        grad
                    };
                    let direction = if momentum != 0.0 {
                        if self.state[index].is_none() {
                            // FIRST step: buffer = clone of the
                            // (weight-decayed) grad — not zeros.
                            self.state[index] = Some((deep_copy(&grad)?, None));
                        } else if let Some((buf, _)) = &mut self.state[index] {
                            let _ = buf.f_mul_scalar_(momentum).map_err(tch_error)?;
                            let _ = buf
                                .f_add_(&grad.f_mul_scalar(1.0 - dampening).map_err(tch_error)?)
                                .map_err(tch_error)?;
                        }
                        let buf = self.state[index].as_ref().unwrap().0.shallow_clone();
                        if nesterov {
                            grad.f_add(&buf.f_mul_scalar(momentum).map_err(tch_error)?)
                                .map_err(tch_error)?
                        } else {
                            buf
                        }
                    } else {
                        grad
                    };
                    let mut param = param.shallow_clone();
                    let _ = param
                        .f_sub_(&direction.f_mul_scalar(lr).map_err(tch_error)?)
                        .map_err(tch_error)?;
                }
                OptimKind::Adam {
                    beta1,
                    beta2,
                    eps,
                    decoupled,
                } => {
                    let (beta1, beta2, eps, decoupled) = (*beta1, *beta2, *eps, *decoupled);
                    let mut param_alias = param.shallow_clone();
                    let grad = if wd != 0.0 && !decoupled {
                        // Classic Adam: coupled L2 into the grad.
                        let detached = param.f_detach().map_err(tch_error)?;
                        grad.f_add(&detached.f_mul_scalar(wd).map_err(tch_error)?)
                            .map_err(tch_error)?
                    } else {
                        grad
                    };
                    if wd != 0.0 && decoupled {
                        // AdamW: p *= (1 - lr*wd) before the moment update.
                        let _ = param_alias
                            .f_mul_scalar_(1.0 - lr * wd)
                            .map_err(tch_error)?;
                    }
                    if self.state[index].is_none() {
                        let m = grad.f_zeros_like().map_err(tch_error)?;
                        let v = grad.f_zeros_like().map_err(tch_error)?;
                        self.state[index] = Some((m, Some(v)));
                    }
                    let (m, v) = match &mut self.state[index] {
                        Some((m, Some(v))) => (m, v),
                        _ => unreachable!("adam state initialized above"),
                    };
                    // PyTorch's ACTUAL first-moment op: lerp_, NOT mul_+add_.
                    let _ = m.f_lerp_(&grad, 1.0 - beta1).map_err(tch_error)?;
                    // Second moment: v = v*beta2 + (g*g)*(1-beta2), in the
                    // same per-element rounding order as addcmul(value=).
                    let _ = v.f_mul_scalar_(beta2).map_err(tch_error)?;
                    let scaled_sq = grad
                        .f_mul(&grad)
                        .and_then(|sq| sq.f_mul_scalar(1.0 - beta2))
                        .map_err(tch_error)?;
                    let _ = v.f_add_(&scaled_sq).map_err(tch_error)?;
                    let bias_correction1 = 1.0 - beta1.powi(t as i32);
                    let bias_correction2 = 1.0 - beta2.powi(t as i32);
                    let denom = v
                        .f_sqrt()
                        .and_then(|s| s.f_div_scalar(bias_correction2.sqrt()))
                        .and_then(|s| s.f_add_scalar(eps))
                        .map_err(tch_error)?;
                    let step_size = lr / bias_correction1;
                    let update = m
                        .f_div(&denom)
                        .and_then(|u| u.f_mul_scalar(step_size))
                        .map_err(tch_error)?;
                    let _ = param_alias.f_sub_(&update).map_err(tch_error)?;
                }
                OptimKind::RmsProp {
                    alpha,
                    eps,
                    momentum,
                } => {
                    let (alpha, eps, momentum) = (*alpha, *eps, *momentum);
                    let detached = param.f_detach().map_err(tch_error)?;
                    let grad = if wd != 0.0 {
                        grad.f_add(&detached.f_mul_scalar(wd).map_err(tch_error)?)
                            .map_err(tch_error)?
                    } else {
                        grad
                    };
                    if self.state[index].is_none() {
                        let sq = grad.f_zeros_like().map_err(tch_error)?;
                        let buf = if momentum > 0.0 {
                            Some(grad.f_zeros_like().map_err(tch_error)?)
                        } else {
                            None
                        };
                        self.state[index] = Some((sq, buf));
                    }
                    let (sq, buf) = match &mut self.state[index] {
                        Some((sq, buf)) => (sq, buf),
                        None => unreachable!("rmsprop state initialized above"),
                    };
                    // sq = sq*alpha + (g*g)*(1-alpha)
                    let _ = sq.f_mul_scalar_(alpha).map_err(tch_error)?;
                    let scaled_sq = grad
                        .f_mul(&grad)
                        .and_then(|s| s.f_mul_scalar(1.0 - alpha))
                        .map_err(tch_error)?;
                    let _ = sq.f_add_(&scaled_sq).map_err(tch_error)?;
                    let avg = sq
                        .f_sqrt()
                        .and_then(|s| s.f_add_scalar(eps))
                        .map_err(tch_error)?;
                    let mut param = param.shallow_clone();
                    if let Some(buf) = buf {
                        let _ = buf.f_mul_scalar_(momentum).map_err(tch_error)?;
                        let _ = buf
                            .f_add_(&grad.f_div(&avg).map_err(tch_error)?)
                            .map_err(tch_error)?;
                        let _ = param
                            .f_sub_(&buf.f_mul_scalar(lr).map_err(tch_error)?)
                            .map_err(tch_error)?;
                    } else {
                        let update = grad
                            .f_div(&avg)
                            .and_then(|u| u.f_mul_scalar(lr))
                            .map_err(tch_error)?;
                        let _ = param.f_sub_(&update).map_err(tch_error)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn zero_grad(&mut self) -> Result<(), String> {
        zero_grads(&self.params)
    }
}

/// The per-tensor zero_grad recipe (issue 0008), looped.
pub fn zero_grads(params: &[Tensor]) -> Result<(), String> {
    for param in params {
        let mut grad = param.f_grad().map_err(tch_error)?;
        if grad.defined() {
            let _ = grad.f_detach_().map_err(tch_error)?;
            let _ = grad.f_zero_().map_err(tch_error)?;
        }
    }
    Ok(())
}
