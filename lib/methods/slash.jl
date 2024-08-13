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

############
# Division #
############

"""
  /(x,y)

Divide two `dualnum`s
"""
function /(x::dualnum, y::dualnum)
  r = x.val / y.val
  g = (y.val*x.grad - x.val*y.grad) / y.val^2
  return dualnum(r, g)
end

"""
  /(x,y)

Divide a `dualnum` by a number `y`
"""
function /(x::dualnum, y::Number)
  r = x.val / y
  g = x.grad / y
  return dualnum(r, g)
end

function /(A::Matrix{dualnum},B::dualnum)
  C = Array{dualnum,2}(undef,size(A,1),size(A,2))
  for y = 1:size(A,2)
    for x = 1:size(A,1)
      C[x,y] = A[x,y]/B
    end
  end
  return C
end

"""
    /(A,num)

Gives the division of named tensor `A` by number `num`

See also: [`div!`](@ref)
"""
function /(A::TNobj,num::Number)
  return div!(copy(A),num)
end

"""
  G = /(A,num)

Divides a tensor `A` by a number `num` with output `G`

See also: [`div!`](@ref)
"""
function /(M::TensType, num::Number)
  P = copy(M)
  @inbounds @simd for w = 1:length(P)
    P[w] /= num
  end
  return P
#  return tensorcombination(M,alpha=(num,),fct=/)
end
