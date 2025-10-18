use std assert
use std/testing *

@test
def "Test repeat 1D basic" [] {
  let input_data = $in
  let r1 = ([1 2] | torch tensor | torch repeat 3 | torch shape)
  # [2] repeated 3 times becomes [6]
  assert ($r1 == [6])
}

@test
def "Test repeat 1D with two dimensions" [] {
  let input_data = $in
  let t = ([1 2 3] | torch tensor)
  let result = ($t | torch repeat 2 4 | torch shape)
  # [3] with sizes [2, 4] auto-expands to [1, 3] then repeats to [2, 12]
  assert ($result == [2 12])
}

@test
def "Test repeat 2D basic" [] {
  let input_data = $in
  let t = ([[1 2] [3 4]] | torch tensor)
  let result = ($t | torch repeat 2 3 | torch shape)
  # [2, 2] repeated [2, 3] becomes [4, 6]
  assert ($result == [4 6])
}
