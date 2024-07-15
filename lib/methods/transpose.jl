import Base.transpose
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