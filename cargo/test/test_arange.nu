use std assert
use std/testing *

@test
def "Arange test 1" [] {
  let r1 = (torch arange 5 | torch value)
  assert ($r1 == [0 1 2 3 4])
}

@test
def "Arange test 2" [] {
  let r2 = (torch arange 2 7 | torch value)
  assert ($r2 == [2 3 4 5 6])
}

@test
def "Arange test 3" [] {
  let r3 = (torch arange 1 5 0.5 --dtype float32 | torch value)
  let expected3 = [1 1.5 2 2.5 3 3.5 4 4.5]
  assert ($r3 == $expected3)
}

@test
def "Test arange with negative range" [] {
  let res = (torch arange -5 0 | torch value)
  assert ($res == [-5 -4 -3 -2 -1])
}

@test
def "Test arange with step of 2" [] {
  let res = (torch arange 0 10 2 | torch value)
  assert ($res == [0 2 4 6 8])
}

@test
def "Test arange with requires_grad flag" [] {
  let t = (torch arange 0 5 --requires_grad true)
  # Just verify it creates successfully - grad testing happens in backward tests
  assert ($t | describe | str contains "string")
}

@test
def "Test arange with device cpu" [] {
  let res = (torch arange 0 3 --device cpu | torch value)
  assert ($res == [0 1 2])
}

@test
def "Test arange with float64 dtype" [] {
  let res = (torch arange 0 3 --dtype float64 | torch value)
  assert ($res == [0 1 2])
}

@test
def "Test arange descending with negative step" [] {
  let res = (torch arange 5 0 -1 --dtype float32 | torch value)
  assert ($res == [5 4 3 2 1])
}

@test
def "Error case with zero step" [] {
  let input_data = $in
  try {
    torch arange 0 10 0
    error make {msg: "Expected error from zero step"}
  } catch {
    # expected
  }
}

@test
def "Error case with invalid device" [] {
  let input_data = $in
  try {
    torch arange 5 --device invalid_device
    error make {msg: "Expected error from invalid device"}
  } catch {
    # expected
  }
}

@test
def "Error case with invalid dtype" [] {
  let input_data = $in
  try {
    torch arange 5 --dtype invalid_dtype
    error make {msg: "Expected error from invalid dtype"}
  } catch {
    # expected
  }
}

@test
def "Test arange with single argument" [] {
  let res = (torch arange 3 | torch value)
  assert ($res == [0 1 2])
}

@test
def "Test arange empty range" [] {
  # When start >= end with positive step, should return empty or single value
  let res = (torch arange 5 5 | torch value)
  # PyTorch returns empty tensor for start == end
  assert (($res | length) == 0)
}
