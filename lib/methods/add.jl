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
    G = add!(A,B,x)

Adds `A + x*B` for dense or quantum tensors with output `G`

See also: [`+`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function add!(A::W, B::W, num::Number) where W <: TensType
  cnum = typeof(num) == eltype(A) ? num : convert(eltype(A),num)
  return tensorcombination!((eltype(A)(1),cnum),A,B)
end

"""
    G = add!(A,B)

Adds `A + B` for dense or quantum tensors with output `G`

See also: [`+`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function add!(A::W, B::W) where W <: TensType
  return add!(A,B,eltype(B)(1))
end
export add!

function add!(A::TNobj,B::TNobj)
  reorder = matchnames(A,B.names)
  if !issorted(reorder)
    C = permutedims(B,reorder)
  else
    C = B
  end
  A.N = add!(A.N,C.N)
  return A
end

function +(x::dtens, y::dtens)
  r = x[0] + y[0]
  g = x[1] + y[1]
  return dtens(r, g)
end