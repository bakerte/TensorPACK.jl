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

#=
"""
  G = inverse_element(x,zero)

Computes the inverse of the input element `x` up to some numerical zero value `zero` with output value `G`.  Below this value, zero is returned to avoid NaNs.
"""
function inverse_element(x::Number,zeronum::Real)
  return abs(x) > zeronum ? 1/x : 0.
end
=#
"""
  G = invmat!(M[,zero=])

Creates inverse of a diagonal matrix in-place (dense matrices are copied) with output `G`; if value is below `zero`, the inverse is set to zero

See also: [`invmat!`](@ref)
"""
function invmat!(M::tens{W};zeronum::Float64=1E-16) where W <: Number
  rM = reshape(M.T,M.size)
  return tens{W}(inv(rM))
end

function invmat!(M::AbstractArray{W,2};zeronum::Float64=1E-16) where  W <: Number
  return inv(M)
end

function invmat!(M::Diagonal{W};zeronum::Float64=1E-12) where W <: Number
  for w = 1:length(M)
    if abs(M[w]) > zeronum
      M[w] = 1/M[w]
    else
      M[w] = 0.
    end
  end
  return M
end
export invmat!

"""
  G = invmat(M[,zero=])

Creates inverse of a diagonal matrix with output `G`; if value is below `zero`, the inverse is set to zero

See also: [`invmat!`](@ref)
"""
function invmat(M::TensType;zeronum::Float64=1E-16) 
  return invmat!(copy(M))#,zeronum=zeronum)
end
export invmat

function invmat(M::AbstractMatrix;zeronum::Float64=1E-16) 
  return invmat!(copy(M))#,zeronum=zeronum)
end

"""
    invmat!(Qt[,zero=])

Creates inverse of a diagonal matrix in place (dense matrices are copied anyway);
if value is below `zero`, the inverse is set to zero

See also: [`invmat`](@ref)
"""
function invmat!(A::qarray)
  @inbounds for q = 1:length(A.T)
    invmat!(A.T[q])
  end
  return A
end
export invmat!

