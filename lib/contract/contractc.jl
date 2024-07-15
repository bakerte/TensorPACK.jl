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
  contractc(A,B[,alpha=])

Similar to contract but 'B' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contractc(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  out = dot(mA,mB,Lfct=identity,Rfct=adjoint)
  return  alpha*out
end

"""
    contractc(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A` with one of the inputs conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contractc(A::TensType;alpha::Number=eltype(A)(1))
  out = dot(A,A,Lfct=identity,Rfct=adjoint)
  return  alpha*out
end

"""
    contractc(A,iA,B,iB[,Z,alpha=,beta=])

Similar to contract but 'B' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contractc(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  if typeof(A) <: diagonal || typeof(B) <: diagonal
    return diagcontract!(false,true,A,convIn(iA),B,convIn(iB),Z...,alpha=alpha,beta=beta,inplace=false)
  else
    mA,mB = checkType(A,B)
    return maincontractor(false,true,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  end
end

"""
  contractc(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contractc(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  iB = ntuple(w->w,ndims(B))
  return contractc(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
end

"""
  contractc(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iA` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contractc(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  iA = ntuple(w->w,ndims(A))
  return contractc(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
end

"""
    contractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contractc`](@ref) to `order`
"""
function contractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = contractc(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = contract(A,iA,B,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contractc(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = contract(A,B,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end



"""
    C = contractc(A,iA,B,iB[,Z,alpha=1,beta=1])

contraction of two `dtens` tensors `A` and `B` (conjugated) over indices `iA` and `iB`; can perform the contraction `alpha`*`A`*`B` + `beta`*`Z` with other inputs
"""
function contractc(A::dtens,iA::intvecType,B::dtens,iB::intvecType,Z::dtens...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))

  if length(Z) > 0
    C = contractc(A[0],iA,B[0],iB,Z[0],alpha=alpha,beta=beta)
    dC = contractc(A[1],iA,B[1],iB,Z[1],alpha=alpha,beta=beta)
    out = dtens(C,contractc(A[1],iA,B[0],iB,dC))
  else
    C = contractc(A[0],iA,B[0],iB,alpha=alpha)
    dC = contractc(A[1],iA,B[1],iB,alpha=alpha)
    out = dtens(C,contractc(A[1],iA,B[0],iB,dC))
  end
  return out
end
