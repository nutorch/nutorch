//! The tensor registry: the daemon-owned map from string handles to live
//! tensors. Ported from v1's TENSOR_REGISTRY concept (v1/cargo/src/lib.rs),
//! minus the global static — the daemon owns one instance for its lifetime.

use std::collections::HashMap;
use tch::Tensor;

#[derive(Default)]
pub struct Registry {
    tensors: HashMap<String, Tensor>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, tensor: Tensor) -> String {
        let handle = uuid::Uuid::new_v4().to_string();
        self.tensors.insert(handle.clone(), tensor);
        handle
    }

    pub fn get(&self, handle: &str) -> Option<&Tensor> {
        self.tensors.get(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_returns_distinct_uuid_handles() {
        let mut registry = Registry::new();
        let a = registry.insert(Tensor::from(1.0));
        let b = registry.insert(Tensor::from(2.0));
        assert_ne!(a, b);
        assert!(registry.get(&a).is_some());
        assert!(registry.get(&b).is_some());
        assert!(registry.get("not-a-handle").is_none());
    }
}
