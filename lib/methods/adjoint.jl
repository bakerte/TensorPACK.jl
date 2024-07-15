import Base.adjoint
"""
  adjoint(c)

Performs adjoint of a `TensType`; assumes rank-2 tensor and flips indices (dagger operation)

See: [`TensType`](@ref) [`transpose`](@ref)
"""
function adjoint(M::TensType)
  if ndims(M) == 1
    rM = reshape(M,size(M,1),1)
  elseif ndims(M) > 2
    error("Issue defining input denstens to adjoint (must be rank-1 or 2)")
  else
    rM = M
  end
  pM = transpose(rM)
  conj!(pM)
  return pM
end


function adjoint(Qt::qarray)
  return conj!(permutedims(Qt,[2,1]))
end
