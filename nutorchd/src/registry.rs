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

    pub fn remove(&mut self, handle: &str) -> Option<Tensor> {
        self.tensors.remove(handle)
    }

    /// Empty the registry; returns how many tensors were freed.
    pub fn clear(&mut self) -> usize {
        let count = self.tensors.len();
        self.tensors.clear();
        count
    }

    pub fn contains(&self, handle: &str) -> bool {
        self.tensors.contains_key(handle)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Approximate bytes held: Σ numel × element size. (Removed as dead code
    /// in issue 0002; legitimately needed by `status` since issue 0004.)
    pub fn approx_bytes(&self) -> u64 {
        self.tensors
            .values()
            .map(|t| t.numel() as u64 * t.kind().elt_size_in_bytes() as u64)
            .sum()
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
