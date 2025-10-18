use std assert
use std/testing *

@test
def "Reshape test 1" [] {
  let input_data = $in
  let v = ([1 2 3 4 5 6] | torch tensor)
  let s1 = ($v | torch reshape [2 3] | torch shape)
  assert ($s1 == [2 3])
}

@test
def "Reshape test 2" [] {
  let v = ([1 2 3 4 5 6] | torch tensor)
  let s2 = ($v | torch reshape [3 -1] | torch shape)
  assert ($s2 == [3 2])
}

@test
def "Reshape test 3" [] {
  let m = ([[1 2 3] [4 5 6]] | torch tensor)
  let s3 = ($m | torch reshape [6] | torch shape)
  assert ($s3 == [6])
}

@test
def "Test reshape 2D to 2D" [] {
  let input_data = $in
  let m = ([[1 2 3] [4 5 6]] | torch tensor)
  let result = ($m | torch reshape [3 2] | torch shape)
  # [2, 3] reshaped to [3, 2]
  assert ($result == [3 2])
}

@test
def "Test reshape to 3D" [] {
  let input_data = $in
  let v = ([1 2 3 4 5 6 7 8] | torch tensor)
  let result = ($v | torch reshape [2 2 2] | torch shape)
  # [8] reshaped to [2, 2, 2]
  assert ($result == [2 2 2])
}

@test
def "Test reshape with -1 at start" [] {
  let input_data = $in
  let v = ([1 2 3 4 5 6 7 8 9 10 11 12] | torch tensor)
  let result = ($v | torch reshape [-1 3] | torch shape)
  # [12] reshaped to [-1, 3] becomes [4, 3]
  assert ($result == [4 3])
}

@test
def "Test reshape preserves values" [] {
  let input_data = $in
  let v = ([1 2 3 4] | torch tensor)
  let result = ($v | torch reshape [2 2] | torch value)
  # Values should be preserved in row-major order
  assert ($result == [[1 2] [3 4]])
}

@test
def "Test reshape 3D to 2D" [] {
  let input_data = $in
  let t = torch full [2 3 4] 1
  let result = ($t | torch reshape [6 4] | torch shape)
  # [2, 3, 4] = 24 elements reshaped to [6, 4]
  assert ($result == [6 4])
}

@test
def "Test reshape to row vector" [] {
  let input_data = $in
  let t = ([[1 2 3] [4 5 6]] | torch tensor)
  let result = ($t | torch reshape [1 6] | torch shape)
  # [2, 3] reshaped to [1, 6]
  assert ($result == [1 6])
}

@test
def "Test reshape to column vector" [] {
  let input_data = $in
  let t = ([[1 2 3] [4 5 6]] | torch tensor)
  let result = ($t | torch reshape [6 1] | torch shape)
  # [2, 3] reshaped to [6, 1]
  assert ($result == [6 1])
}

@test
def "Test reshape chaining" [] {
  let input_data = $in
  let v = ([1 2 3 4 5 6 7 8] | torch tensor)
  let result = ($v | torch reshape [2 4] | torch reshape [4 2] | torch shape)
  # [8] -> [2, 4] -> [4, 2]
  assert ($result == [4 2])
}

@test
def "Test reshape to scalar" [] {
  let input_data = $in
  let v = ([42] | torch tensor)
  let result = ($v | torch reshape [] | torch shape)
  # [1] reshaped to [] (scalar)
  assert ($result == [])
}

@test
def "Test reshape from scalar" [] {
  let input_data = $in
  let s = (42 | torch tensor)
  let result = ($s | torch reshape [1] | torch shape)
  # [] (scalar) reshaped to [1]
  assert ($result == [1])
}
