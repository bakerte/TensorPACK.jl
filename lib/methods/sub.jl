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
    G = sub!(A,B,x)

Subtracts `A - x*B` for dense or quantum tensors with output `G`

See also: [`-`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function sub!(A::W,B::W,mult::Number) where W <: TensType
  cnum = typeof(mult) == eltype(A) ? mult : convert(eltype(A),mult)
  return add!(A,B,-cnum)
end

"""
    G = sub!(A,B)

Subtracts `A - B` for dense or quantum tensors with output `G`

See also: [`-`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function sub!(A::W,B::W) where W <: TensType
  return add!(A,B,eltype(A)(-1))
end
export sub!

"""
    sub!(A,B)

Subtracts tensor `A` from `B` (changes `A`)

See also: [`-`](@ref)
"""
function sub!(A::TNobj,B::TNobj)
  reorder = matchnames(A,B.names)
  if !issorted(reorder)
    C = permutedims(B,reorder)
  else
    C = B
  end
  A.N = sub!(A.N,C.N)
  return A
end
