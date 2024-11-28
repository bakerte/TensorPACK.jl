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
    G = mult!(A,x)

Multiplies `x*A` (commutative) for dense or quantum tensors with output `G`

See also: [`*`](@ref) [`add!`](@ref) [`sub!`](@ref) [`div!`](@ref)
"""
function mult!(M::W, num::Number) where W <: TensType
  cnum = typeof(num) == eltype(M) ? num : convert(eltype(M),num)
  return tensorcombination!(M,alpha=(cnum,))
end

"""
    G = mult!(x,A)

Multiplies `x*A` (commutative) for dense or quantum tensors with output `G`

See also: [`*`](@ref) [`add!`](@ref) [`sub!`](@ref) [`div!`](@ref)
"""
function mult!(num::Number,M::W) where W <: TensType
  return mult!(M,num)
end

"""
    mult!(A,num)

Gives the multiplication of named tensor `A` by number `num`

See also: [`*`](@ref)
"""
function mult!(A::TNobj,num::Number)
  A.N = mult!(A.N,num)
  return A
end
