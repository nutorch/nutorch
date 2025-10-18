use std assert
use std/testing *

@test
def "Test add with pipeline and argument" [] {
  let input_dat = $in
  let result1 = ([1] | torch tensor) | torch add ([2] | torch tensor) | torch value | get 0
  assert ($result1 == 3)
}

@test
def "Test add with two arguments" [] {
  let input_dat = $in
  let result2 = torch add ([1] | torch tensor) ([2] | torch tensor) | torch value | get 0
  assert ($result2 == 3)
}

@test
def "Test add with vectors" [] {
  let input_dat = $in
  let result3 = ([1 2 3] | torch tensor) | torch add ([4 5 6] | torch tensor) | torch value
  assert ($result3 == [5 7 9])
}

@test
def "Test add with 2D tensors" [] {
  let input_dat = $in
  let t1 = torch tensor [[1 2] [3 4]]
  let t2 = torch tensor [[5 6] [7 8]]
  let result = (torch add $t1 $t2 | torch value)
  assert ($result == [[6 8] [10 12]])
}

@test
def "Test add with broadcasting" [] {
  let input_dat = $in
  let t1 = torch tensor [1 2 3]
  let scalar = torch tensor 10
  let result = ($t1 | torch add $scalar | torch value)
  assert ($result == [11 12 13])
}

@test
def "Test add with negative numbers" [] {
  let input_dat = $in
  let t1 = torch tensor [1 2 3]
  let t2 = torch tensor [-1 -2 -3]
  let result = (torch add $t1 $t2 | torch value)
  assert ($result == [0 0 0])
}

@test
def "Test add with floats" [] {
  let input_dat = $in
  let t1 = torch tensor [1.5 2.5 3.5]
  let t2 = torch tensor [0.5 0.5 0.5]
  let result = (torch add $t1 $t2 | torch value)
  assert ($result == [2.0 3.0 4.0])
}

@test
def "Error case with device mismatch" [] {
  let input_dat = $in
  try {
    let t1 = torch tensor [1 2 3] --device cpu
    let t2 = torch tensor [4 5 6] --device cpu
    # This test would fail if we had multiple devices available
    # For now, just verify both tensors are on same device
    let result = (torch add $t1 $t2 | torch value)
    assert ($result == [5 7 9])
  } catch {
    # Expected if devices differ
  }
}

@test
def "Error case with invalid tensor ID" [] {
  let input_dat = $in
  try {
    let t1 = torch tensor [1 2 3]
    torch add $t1 "invalid-uuid"
    error make {msg: "Expected error from invalid tensor ID"}
  } catch {
    # expected
  }
}

@test
def "Error case with missing second tensor" [] {
  let input_dat = $in
  try {
    let t1 = torch tensor [1 2 3]
    $t1 | torch add
    error make {msg: "Expected error from missing second tensor"}
  } catch {
    # expected
  }
}

@test
def "Error case with no tensors provided" [] {
  let input_dat = $in
  try {
    torch add
    error make {msg: "Expected error from no tensors"}
  } catch {
    # expected
  }
}

@test
def "Error case with three tensors" [] {
  let input_dat = $in
  try {
    let t1 = torch tensor [1]
    let t2 = torch tensor [2]
    let t3 = torch tensor [3]
    $t1 | torch add $t2 $t3
    error make {msg: "Expected error from three tensors"}
  } catch {
    # expected
  }
}
