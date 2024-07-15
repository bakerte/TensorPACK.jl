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
    A = diagonal{W}(P)

Converts `diagonal` type `P` (element type `Z`) into a diagonal of element type `W`
"""
function diagonal{W}(P::diagonal{Z}) where {W <: Number, Z <: Number}
  if W != Z
    newtens = diagonal{W}(convert(Array{W,1},P.T))
  else
    newtens = P
  end
  return newtens
end

"""
    A = diagonal(alpha,beta)

Converts arrays of numbers `alpha` (diagonal column) and `beta` (off-diagonal elements) into a tridiagonal matrix
"""
function diagonal(alpha::Array{W,1},beta::Array{W,1}) where W <: Real
  return Base.SymTridiagonal(alpha,beta)
end

"""
    M = diagonal(alpha,beta)

Converts arrays of arrays (either `Array{W,2}` or `tens{W}`) numbers `alpha` (diagonal column) and `beta` (off-diagonal elements) into a block-tridiagonal matrix
"""
function diagonal(alpha::Array{W,1},beta::Array{W,1}) where W <: Union{Array{S,2},tens{S}} where S <: Number
  g = size(alpha[1],1)
  p = 0
  while p < length(alpha) && isassigned(alpha,p+1)
    p += 1
  end

  m = p*g
  
  M = zeros(S,m,m)
  M[1:g,1:g] = alpha[1]
  for n = 2:p
    interval = (1+g*(n-1)):g*n
    intervaln = (1+g*(n-2)):g*(n-1)

    M[interval,interval] = alpha[n]
    M[intervaln,interval] = beta[n-1]'
    M[interval,intervaln] = beta[n-1]
  end
  return M
end
