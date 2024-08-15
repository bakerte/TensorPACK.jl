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
    C = ^(A,B)

Computes `A^B` for a diagonal matrix `A` by a power `B`
"""
function ^(A::Diagonal{W},P::B) where {W <: Number, B <: Number}
  C = A.T .^ P
  return Diagonal(C)
end

"""
    C = ^(A,B)

Computes `A^B` for a `denstens` matrix `A` by a power `B`
"""
function ^(A::tens{W},P::B) where {W <: Number, B <: Number}
  if B <: Integer
    C = prod(w->A,1:P)
  else
    R = Array(A)
    C = tens(R^P)
  end
  return C
end

"""
  ^(x,y)

Exponent of two `dualnum`s
"""
function ^(x::dualnum, y::Number)
  r = x.val ^ y
  g = y*x.val^(y-1) * x.grad
  return dualnum(r, g)
end

"""
    C = ^(A,B)

Computes `A^B` for a `denstens` matrix `A` by a power `B`
"""
function ^(A::nametens,P::B) where B <: Number
  @assert(ndims(A)==2)
  return nametens(A.N^P,A.names)
end
