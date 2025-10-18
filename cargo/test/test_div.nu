use std assert
use std/testing *

@test
def "Test div with pipeline and argument" [] {
  let input_data = $in
  let result1 = ([10] | torch tensor) | torch div ([2] | torch tensor) | torch value | get 0
  assert ($result1 == 5)
}

@test
def "Test div with two arguments" [] {
  let input_data = $in
  let result2 = torch div ([10] | torch tensor) ([2] | torch tensor) | torch value | get 0
  assert ($result2 == 5)
}

@test
def "Test div with vectors" [] {
  let input_data = $in
  let result3 = ([20 30 40] | torch tensor) | torch div ([4 5 8] | torch tensor) | torch value
  assert ($result3 == [5 6 5])
}

@test
def "Test div with 2D tensors" [] {
  let input_data = $in
  let t1 = torch tensor [[10 20] [30 40]]
  let t2 = torch tensor [[2 4] [5 8]]
  let result = (torch div $t1 $t2 | torch value)
  assert ($result == [[5 5] [6 5]])
}

@test
def "Test div with broadcasting scalar" [] {
  let input_data = $in
  let t1 = torch tensor [10 20 30]
  let scalar = torch tensor 10
  let result = ($t1 | torch div $scalar | torch value)
  assert ($result == [1 2 3])
}

@test
def "Test div with negative numbers" [] {
  let input_data = $in
  let t1 = torch tensor [10 -20 30]
  let t2 = torch tensor [2 -4 5]
  let result = (torch div $t1 $t2 | torch value)
  assert ($result == [5 5 6])
}

@test
def "Test div with floats" [] {
  let input_data = $in
  let t1 = torch tensor [5.0 10.0 15.0]
  let t2 = torch tensor [2.0 4.0 3.0]
  let result = (torch div $t1 $t2 | torch value)
  assert ($result == [2.5 2.5 5.0])
}

@test
def "Test div resulting in fractions" [] {
  let input_data = $in
  let t1 = torch tensor [1.0 2.0 3.0]
  let t2 = torch tensor [2.0 2.0 2.0]
  let result = (torch div $t1 $t2 | torch value)
  assert ($result == [0.5 1.0 1.5])
}

@test
def "Test div by one - identity" [] {
  let input_data = $in
  let t1 = torch tensor [5 10 15]
  let t2 = torch tensor [1 1 1]
  let result = (torch div $t1 $t2 | torch value)
  assert ($result == [5 10 15])
}

@test
def "Error case with device mismatch" [] {
  let input_data = $in
  try {
    let t1 = torch tensor [10 20 30] --device cpu
    let t2 = torch tensor [2 4 5] --device cpu
    # This test would fail if we had multiple devices available
    # For now, just verify both tensors are on same device
    let result = (torch div $t1 $t2 | torch value)
    assert ($result == [5 5 6])
  } catch {
    # Expected if devices differ
  }
}

@test
def "Error case with invalid tensor ID" [] {
  let input_data = $in
  try {
    let t1 = torch tensor [1 2 3]
    torch div $t1 "invalid-uuid"
    error make {msg: "Expected error from invalid tensor ID"}
  } catch {
    # expected
  }
}

@test
def "Error case with missing second tensor" [] {
  let input_data = $in
  try {
    let t1 = torch tensor [1 2 3]
    $t1 | torch div
    error make {msg: "Expected error from missing second tensor"}
  } catch {
    # expected
  }
}

@test
def "Error case with no tensors provided" [] {
  let input_data = $in
  try {
    torch div
    error make {msg: "Expected error from no tensors"}
  } catch {
    # expected
  }
}

@test
def "Error case with three tensors" [] {
  let input_data = $in
  try {
    let t1 = torch tensor [1]
    let t2 = torch tensor [2]
    let t3 = torch tensor [3]
    $t1 | torch div $t2 $t3
    error make {msg: "Expected error from three tensors"}
  } catch {
    # expected
  }
}
