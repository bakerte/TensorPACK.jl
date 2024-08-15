###############################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                               v1.0
#
###############################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.0+)
#

"""
  adjoint(c)

Performs adjoint of a `TensType`; assumes rank-2 tensor and flips indices (dagger operation)

See: [`TensType`](@ref) [`transpose`](@ref)
"""
function adjoint(M::TensType)
  @assert(ndims(M)==2)
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

"""
    tQt = adjoint(Qt)

Takes the adjoint of a `qarray` of rank-2
"""
function adjoint(Qt::qarray)
  @assert(ndims(Qt)==2)
  return conj!(permutedims(Qt,[2,1]))
end
