use std assert
use std/testing *

@test
def "Test zero_grad single tensor via pipeline" [] {
  let input_data = $in
  let p = (torch full [1] 5 --requires_grad true)
  let loss = ($p | torch mean)
  $loss | torch backward
  # Gradient should exist after backward
  let grad_before = ($p | torch grad)
  assert ($grad_before != null)
  # Zero out gradient
  $p | torch zero_grad
  # Gradient should now be zero
  let grad_after = ($p | torch grad)
  assert ($grad_after != null)
  let grad_val = ($grad_after | torch value | get 0)
  assert ($grad_val == 0)
}

@test
def "Test zero_grad single tensor via argument" [] {
  let input_data = $in
  let p = (torch full [1] 5 --requires_grad true)
  let loss = ($p | torch mean)
  $loss | torch backward
  # Zero out gradient via argument form (must use list)
  torch zero_grad [$p]
  # Gradient should now be zero
  let grad_after = ($p | torch grad)
  let grad_val = ($grad_after | torch value | get 0)
  assert ($grad_val == 0)
}

@test
def "Test zero_grad list via pipeline" [] {
  let input_data = $in
  let p = (torch full [1] 5 --requires_grad true)
  let loss = ($p | torch mean)
  $loss | torch backward
  # Clear grad using list form
  [$p] | torch zero_grad
  let grad_after = ($p | torch grad)
  let grad_val = ($grad_after | torch value | get 0)
  assert ($grad_val == 0)
}

@test
def "Test zero_grad list via argument" [] {
  let input_data = $in
  let p = (torch full [1] 5 --requires_grad true)
  let loss = ($p | torch mean)
  $loss | torch backward
  # Clear grad using argument list form
  torch zero_grad [$p]
  let grad_after = ($p | torch grad)
  let grad_val = ($grad_after | torch value | get 0)
  assert ($grad_val == 0)
}

@test
def "Test zero_grad multiple tensors" [] {
  let input_data = $in
  let w1 = (torch full [1] 3 --requires_grad true)
  let w2 = (torch full [1] 4 --requires_grad true)
  # Create loss that depends on both
  let sum = (torch add $w1 $w2)
  let loss = ($sum | torch mean)
  $loss | torch backward
  # Both should have gradients
  assert (($w1 | torch grad) != null)
  assert (($w2 | torch grad) != null)
  # Zero out both gradients
  [$w1 $w2] | torch zero_grad
  # Both gradients should be zero
  let grad1_val = ($w1 | torch grad | torch value | get 0)
  let grad2_val = ($w2 | torch grad | torch value | get 0)
  assert ($grad1_val == 0)
  assert ($grad2_val == 0)
}

@test
def "Test zero_grad returns tensor IDs" [] {
  let input_data = $in
  let w1 = (torch full [1] 1 --requires_grad true)
  let w2 = (torch full [1] 2 --requires_grad true)
  # Should return list of same IDs
  let result = ([$w1 $w2] | torch zero_grad)
  assert ($result == [$w1 $w2])
}

@test
def "Test zero_grad prevents SGD step" [] {
  let input_data = $in
  let p = (torch full [1] 5 --requires_grad true)
  let loss = ($p | torch mean)
  $loss | torch backward
  [$p] | torch zero_grad
  # Run sgd_step with giant lr; value should NOT change (grad == 0)
  let before = ($p | torch value | get 0)
  [$p] | torch sgd_step --lr 10
  let after = ($p | torch value | get 0)
  assert ($before == $after)
}

@test
def "Test zero_grad on tensor without grad" [] {
  let input_data = $in
  let p = (torch full [1] 5 --requires_grad true)
  # No backward called, so no gradient yet
  # Should not error when calling zero_grad
  $p | torch zero_grad
  # Should still return the tensor ID
  let result = ($p | torch zero_grad)
  assert ($result == [$p])
}

@test
def "Error case - empty list" [] {
  let input_data = $in
  try {
    [] | torch zero_grad
    error make {msg: "Expected error from empty list"}
  } catch {
    # expected - list cannot be empty
  }
}

@test
def "Error case - invalid tensor ID" [] {
  let input_data = $in
  try {
    "invalid-uuid" | torch zero_grad
    error make {msg: "Expected error from invalid tensor ID"}
  } catch {
    # expected
  }
}

@test
def "Error case - both pipeline and argument" [] {
  let input_data = $in
  let t1 = (torch full [1] 1 --requires_grad true)
  let t2 = (torch full [1] 2 --requires_grad true)
  try {
    $t1 | torch zero_grad [$t2]
    error make {msg: "Expected error from dual input"}
  } catch {
    # expected - cannot provide both pipeline and argument
  }
}

@test
def "Error case - no input provided" [] {
  let input_data = $in
  try {
    torch zero_grad
    error make {msg: "Expected error from missing input"}
  } catch {
    # expected - must provide tensor IDs
  }
}
