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

"""
   C = add!(A,B)

adds two `TNobj`s together
"""
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