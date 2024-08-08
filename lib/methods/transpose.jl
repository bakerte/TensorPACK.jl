



"""
  transpose(c)

Performs adjoint of a `TensType`; assumes rank-2 tensor and flips indices (no conjugation)

See: [`TensType`](@ref) [`adjoint`](@ref)
"""
function transpose(M::TensType)
  pM = ndims(M) == 1 ? reshape(M,size(M)...,1) : M
  pM = permutedims(pM,[2,1])
  return pM
end


function transpose(A::Union{Matrix{dualnum}, Vector{dualnum}})
  num_vars = length(A[1,1].gradient)
  original_rows = size(A, 1)
  original_cols = size(A, 2)

  B = dualnum_zeros(original_cols, original_rows, num_vars)
  for i in 1:original_rows
    for j in original_cols
      B[j, i] = A[i, j]
    end
  end
  return B
end
