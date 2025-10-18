use std assert
use std/testing *

@test
def "Test softmax 1D - pipeline form" [] {
  let input_data = $in
  let t = ([1 2 3] | torch tensor)
  let result = ($t | torch softmax --dim 0 | torch value)
  # Softmax should sum to 1.0
  let sum = ($result | math sum)
  assert ((($sum - 1.0) | math abs) < 0.0001)
}

@test
def "Test softmax 1D - argument form" [] {
  let input_data = $in
  let t = ([1 2 3] | torch tensor)
  let result = (torch softmax $t --dim 0 | torch value)
  # Softmax should sum to 1.0
  let sum = ($result | math sum)
  assert ((($sum - 1.0) | math abs) < 0.0001)
}

@test
def "Test softmax 2D dim 0" [] {
  let input_data = $in
  let t = ([[1 2] [3 4]] | torch tensor)
  let result = ($t | torch softmax --dim 0 | torch value)
  # Each column should sum to 1.0
  let col0_sum = ([$result.0.0 $result.1.0] | math sum)
  let col1_sum = ([$result.0.1 $result.1.1] | math sum)
  assert ((($col0_sum - 1.0) | math abs) < 0.0001)
  assert ((($col1_sum - 1.0) | math abs) < 0.0001)
}

@test
def "Test softmax 2D dim 1" [] {
  let input_data = $in
  let t = ([[1 2] [3 4]] | torch tensor)
  let result = ($t | torch softmax --dim 1 | torch value)
  # Each row should sum to 1.0
  let row0_sum = ($result.0 | math sum)
  let row1_sum = ($result.1 | math sum)
  assert ((($row0_sum - 1.0) | math abs) < 0.0001)
  assert ((($row1_sum - 1.0) | math abs) < 0.0001)
}

@test
def "Test softmax default dim - last dimension" [] {
  let input_data = $in
  let t = ([[1 2 3] [4 5 6]] | torch tensor)
  let result = ($t | torch softmax | torch value)
  # Default should be last dim (dim 1 for 2D), each row sums to 1
  let row0_sum = ($result.0 | math sum)
  let row1_sum = ($result.1 | math sum)
  assert ((($row0_sum - 1.0) | math abs) < 0.0001)
  assert ((($row1_sum - 1.0) | math abs) < 0.0001)
}

@test
def "Test softmax preserves shape" [] {
  let input_data = $in
  let t = ([[1 2] [3 4]] | torch tensor)
  let result = ($t | torch softmax --dim 1 | torch shape)
  # Shape should be preserved
  assert ($result == [2 2])
}

@test
def "Test softmax 3D tensor" [] {
  let input_data = $in
  let t = torch full [2 3 4] 1
  let result = ($t | torch softmax --dim 2 | torch shape)
  # Shape preserved
  assert ($result == [2 3 4])
}

@test
def "Test softmax with dtype" [] {
  let input_data = $in
  let t = ([1 2 3] | torch tensor)
  let result = ($t | torch softmax --dim 0 --dtype float64 | torch value)
  # Should still sum to 1.0
  let sum = ($result | math sum)
  assert ((($sum - 1.0) | math abs) < 0.0001)
}

@test
def "Test softmax uniform input" [] {
  let input_data = $in
  let t = ([1 1 1] | torch tensor)
  let result = ($t | torch softmax --dim 0 | torch value)
  # Uniform input should give uniform output (1/3 each)
  assert ((($result.0 - 0.3333) | math abs) < 0.001)
  assert ((($result.1 - 0.3333) | math abs) < 0.001)
  assert ((($result.2 - 0.3333) | math abs) < 0.001)
}

@test
def "Error case - invalid tensor ID" [] {
  let input_data = $in
  try {
    "invalid-uuid" | torch softmax
    error make {msg: "Expected error from invalid tensor ID"}
  } catch {
    # expected
  }
}

@test
def "Error case - invalid dimension" [] {
  let input_data = $in
  let t = ([[1 2] [3 4]] | torch tensor)
  try {
    $t | torch softmax --dim 5
    error make {msg: "Expected error from invalid dimension"}
  } catch {
    # expected - dim 5 out of bounds for 2D tensor
  }
}

@test
def "Error case - negative dimension" [] {
  let input_data = $in
  let t = ([[1 2] [3 4]] | torch tensor)
  try {
    $t | torch softmax --dim (-1)
    error make {msg: "Expected error from negative dimension"}
  } catch {
    # expected
  }
}

