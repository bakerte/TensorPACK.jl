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
  ccontract(A,B[,alpha=])

Similar to contract but 'A' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontract(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  out = dot(mA,mB,Lfct=adjoint,Rfct=identity)
  return  alpha*out
end

"""
    ccontract(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A` with one of the inputs conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontract(A::TensType;alpha::Number=eltype(A)(1))
  out = dot(A,A,Lfct=adjoint,Rfct=identity)
  return  alpha*out
end

"""
    ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`ccontract`](@ref) to `order`
"""
function ccontract(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = ccontract(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontract(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = contract(A,iA,B,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontract(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = contract(A,B,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    C = ccontract(A,iA,B,iB[,Z,alpha=1,beta=1])

contraction of two `dtens` tensors `A` (conjugated) and `B` over indices `iA` and `iB`; can perform the contraction `alpha`*`A`*`B` + `beta`*`Z` with other inputs
"""
function ccontract(A::dtens,iA::intvecType,B::dtens,iB::intvecType,Z::dtens...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))

  if length(Z) > 0
    C = ccontract(A[0],iA,B[0],iB,Z[0],alpha=alpha,beta=beta)
    dC = ccontract(A[1],iA,B[1],iB,Z[1],alpha=alpha,beta=beta)
    out = dtens(C,ccontract(A[1],iA,B[0],iB,dC))
  else
    C = ccontract(A[0],iA,B[0],iB,alpha=alpha)
    dC = ccontract(A[1],iA,B[1],iB,alpha=alpha)
    out = dtens(C,ccontract(A[1],iA,B[0],iB,dC))
  end
  return out
end
