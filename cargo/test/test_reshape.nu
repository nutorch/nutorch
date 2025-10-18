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
