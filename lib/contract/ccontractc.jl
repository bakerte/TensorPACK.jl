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
    ccontractc(A,B[,alpha=])

Similar to contract but 'A' and 'B' are conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['contract'](@ref)
"""
function ccontractc(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  out = dot(mA,mB,Lfct=adjoint,Rfct=adjoint)
  return  alpha*out
end

"""
    ccontractc(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A` with both inputs conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['contract'](@ref)
"""
function ccontractc(A::TensType;alpha::Number=eltype(A)(1))
  out = dot(A,A,Lfct=adjoint,Rfct=adjoint)
  return  alpha*out
end

"""
    ccontract(A,iA,B,iB[,Z,alpha=,beta=])

Similar to contract but 'A' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontract(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  if typeof(A) <: diagonal || typeof(B) <: diagonal
    return diagcontract!(true,false,A,convIn(iA),B,convIn(iB),Z...,alpha=alpha,beta=beta,inplace=false)
  else
    mA,mB = checkType(A,B)
    return maincontractor(true,false,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  end
end

"""
  ccontract(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontract(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  iB = ntuple(w->w,ndims(B))
  return ccontract(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
end

"""
  ccontract(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontract(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  iA = ntuple(w->w,ndims(A))
  return ccontract(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
end


"""
    ccontractc(A,iA,B,iB[,Z,alpha=,beta=])

Similar to contract but 'A' and 'B' are conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontractc(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  if typeof(A) <: diagonal || typeof(B) <: diagonal
    return diagcontract!(true,true,A,convIn(iA),B,convIn(iB),Z...,alpha=alpha,beta=beta,inplace=false)
  else
    mA,mB = checkType(A,B)
    return maincontractor(true,true,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  end
end
export contract,ccontract,contractc,ccontractc

"""
  ccontractc(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontractc(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  iB = ntuple(w->w,ndims(B))
  return ccontractc(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
end

"""
  ccontractc(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontractc(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  iA = ntuple(w->w,ndims(A))
  return ccontractc(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
end


"""
    ccontractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`ccontractc`](@ref) to `order`
"""
function ccontractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = ccontractc(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = ccontractc(A,iA,B,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontractc(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = ccontractc(A,B,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    C = ccontractc(A,iA,B,iB[,Z,alpha=1,beta=1])

contraction of two `dtens` tensors `A` (conjugated) and `B` (conjugated) over indices `iA` and `iB`; can perform the contraction `alpha`*`A`*`B` + `beta`*`Z` with other inputs
"""
function ccontractc(A::dtens,iA::intvecType,B::dtens,iB::intvecType,Z::dtens...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))

  if length(Z) > 0
    C = ccontractc(A[0],iA,B[0],iB,Z[0],alpha=alpha,beta=beta)
    dC = ccontractc(A[1],iA,B[1],iB,Z[1],alpha=alpha,beta=beta)
    out = dtens(C,ccontractc(A[1],iA,B[0],iB,dC))
  else
    C = ccontractc(A[0],iA,B[0],iB,alpha=alpha)
    dC = ccontractc(A[1],iA,B[1],iB,alpha=alpha)
    out = dtens(C,ccontractc(A[1],iA,B[0],iB,dC))
  end
  return out
end
