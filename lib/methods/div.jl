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
    B = div!(A,num)

Gives the division of named tensor `A` by number `num`

See also: [`/`](@ref)
"""
function div!(A::TNobj,num::Number)
  A.N = div!(A.N,num)
  return A
end

"""
    B = div!(A,x)

Division by a scalar `A/x` (default x = 1) for dense or quantum tensors with output `B`

See also: [`/`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref)
"""
function div!(M::TensType, num::Number)
  cnum = typeof(num) == eltype(M) ? num : convert(eltype(M),num)
  return tensorcombination!(M,alpha=(cnum,),fct=/)
end
export div!