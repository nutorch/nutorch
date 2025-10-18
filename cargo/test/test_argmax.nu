use std assert
use std/testing *

@test
def "Test argmax with pipeline input flattened" [] {
  let input_data = $in
  let t = torch tensor [1 5 3 2 4]
  let result = ($t | torch argmax | torch value)
  # Argmax of [1,5,3,2,4] = 1 (index of max value 5)
  assert ($result == 1)
}

@test
def "Test argmax with argument input flattened" [] {
  let input_data = $in
  let t = torch tensor [1 5 3 2 4]
  let result = (torch argmax $t | torch value)
  # Argmax of [1,5,3,2,4] = 1 (index of max value 5)
  assert ($result == 1)
}

@test
def "Test argmax with 2D tensor flattened" [] {
  let input_data = $in
  let t = torch tensor [[1 2 3] [4 5 6]]
  let result = ($t | torch argmax | torch value)
  # Argmax of flattened tensor = 5 (index of max value 6)
  assert ($result == 5)
}

@test
def "Test argmax with 2D tensor along dim 0" [] {
  let input_data = $in
  let t = torch tensor [[1 5 3] [4 2 6]]
  let result = ($t | torch argmax --dim 0 | torch value)
  # Argmax along rows: [argmax(1,4), argmax(5,2), argmax(3,6)] = [1, 0, 1]
  assert ($result == [1 0 1])
}

@test
def "Test argmax with 2D tensor along dim 1" [] {
  let input_data = $in
  let t = torch tensor [[1 5 3] [4 2 6]]
  let result = ($t | torch argmax --dim 1 | torch value)
  # Argmax along columns: [argmax(1,5,3), argmax(4,2,6)] = [1, 2]
  assert ($result == [1 2])
}

@test
def "Test argmax with keepdim true" [] {
  let input_data = $in
  let t = torch tensor [[1 5 3] [4 2 6]]
  let result = ($t | torch argmax --dim 1 --keepdim true | torch value)
  # Argmax with keepdim should preserve dimensions as [[1], [2]]
  assert ($result == [[1] [2]])
}

@test
def "Test argmax with keepdim false" [] {
  let input_data = $in
  let t = torch tensor [[1 5 3] [4 2 6]]
  let result = ($t | torch argmax --dim 1 --keepdim false | torch value)
  # Argmax with keepdim false reduces dimensions to [1, 2]
  assert ($result == [1 2])
}

@test
def "Test argmax with negative numbers" [] {
  let input_data = $in
  let t = torch tensor [-5 -2 -8 -1 -10]
  let result = (torch argmax $t | torch value)
  # Argmax of negative numbers = 3 (index of -1)
  assert ($result == 3)
}

@test
def "Test argmax with mixed positive and negative" [] {
  let input_data = $in
  let t = torch tensor [-5 3 -2 8 -1]
  let result = ($t | torch argmax | torch value)
  # Argmax = 3 (index of 8)
  assert ($result == 3)
}

@test
def "Test argmax with all same values" [] {
  let input_data = $in
  let t = torch tensor [5 5 5 5]
  let result = (torch argmax $t | torch value)
  # When all values are same, returns index 0
  assert ($result == 0)
}

@test
def "Test argmax with floats" [] {
  let input_data = $in
  let t = torch tensor [1.5 3.75 2.25 3.5]
  let result = ($t | torch argmax | torch value)
  # Argmax = 1 (index of 3.75)
  assert ($result == 1)
}

@test
def "Test argmax with 3D tensor flattened" [] {
  let input_data = $in
  let t = torch tensor [[[1 2] [3 4]] [[5 6] [7 8]]]
  let result = ($t | torch argmax | torch value)
  # Argmax of flattened tensor = 7 (index of max value 8)
  assert ($result == 7)
}

@test
def "Test argmax with 3D tensor along dim 0" [] {
  let input_data = $in
  let t = torch tensor [[[1 2] [3 4]] [[5 6] [7 8]]]
  let result = ($t | torch argmax --dim 0 | torch value)
  # Argmax along first dimension = [[1, 1], [1, 1]]
  assert ($result == [[1 1] [1 1]])
}

@test
def "Test argmax first occurrence" [] {
  let input_data = $in
  let t = torch tensor [3 5 5 2]
  let result = ($t | torch argmax | torch value)
  # When there are multiple max values, returns first occurrence index
  assert ($result == 1)
}

@test
def "Error case with invalid tensor ID" [] {
  let input_data = $in
  try {
    torch argmax "invalid-uuid"
    error make {msg: "Expected error from invalid tensor ID"}
  } catch {
    # expected
  }
}

@test
def "Error case with no tensor provided" [] {
  let input_data = $in
  try {
    torch argmax
    error make {msg: "Expected error from no tensor"}
  } catch {
    # expected
  }
}

@test
def "Error case with both pipeline and argument" [] {
  let input_data = $in
  try {
    let t1 = torch tensor [1 2 3]
    let t2 = torch tensor [4 5 6]
    $t1 | torch argmax $t2
    error make {msg: "Expected error from conflicting input"}
  } catch {
    # expected
  }
}

@test
def "Error case with invalid dimension" [] {
  let input_data = $in
  try {
    let t = torch tensor [1 2 3]
    $t | torch argmax --dim 5
    error make {msg: "Expected error from invalid dimension"}
  } catch {
    # expected
  }
}

@test
def "Error case with negative dimension" [] {
  let input_data = $in
  try {
    let t = torch tensor [1 2 3]
    $t | torch argmax --dim -1
    error make {msg: "Expected error from negative dimension"}
  } catch {
    # expected
  }
}
