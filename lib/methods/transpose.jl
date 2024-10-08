#########################################################################
#
#           Tensor Linear Algebra Package (TENPACK)
#                          v1.0
#
#########################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.4+)
#

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

"""
  transpose(c)

Performs adjoint of a matrix of `dualnum` `A`; assumes rank-2 tensor and flips indices (no conjugation)

See: [`dualnum`](@ref) [`adjoint`](@ref)
"""
function transpose(A::Matrix{W}#=Union{Matrix{dualnum}, Vector{dualnum}}=#) where W <: dualnum
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
