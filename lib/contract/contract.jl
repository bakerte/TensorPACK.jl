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
  C = contract(A,B[,alpha=])

Contracts to (alpha * A * B and returns a scalar output...if only `A` is specified, then the norm is evaluated

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contract(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  out = dot(mA,mB,Lfct=identity,Rfct=identity)
  return  alpha*out
end

"""
    contract(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A`

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contract(A::TensType;alpha::Number=eltype(A)(1))
  out = dot(A,A,Lfct=identity,Rfct=identity)
  return alpha*out
end

"""
    contract(A,iA,B,iB[,Z,alpha=,beta=])
Contracts to (alpha * A * B + beta * Z) on input indices `iA` and `iB`; accepts different formats, ex: 1,[1,2],[1 2]); accepts any rank and also Qtensors

# Example:

```julia
julia> A = [1 0;0 -1];B = [-1 0;0 1];Z=[1 0;0 0];
julia> contract(A,2,B,1,Z,alpha=2.,beta=5)
2×2 Array{Float64,2}:
3.0   0.0
0.0  -2.0
```

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contract(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  if typeof(A) <: diagonal || typeof(B) <: diagonal
    out = diagcontract!(false,false,A,convIn(iA),B,convIn(iB),Z...,alpha=alpha,beta=beta,inplace=false)
    return out
  else
    mA,mB = checkType(A,B)
    return maincontractor(false,false,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  end
end

"""
  contract(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contract(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  iB = ntuple(w->w,ndims(B))
  return contract(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
end

"""
  contract(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contract(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  iA = ntuple(w->w,ndims(A))
  return contract(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
end

"""
    contract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contract(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = contract(A,iA,B,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contract(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = contract(A,iA,B,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contract(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  newT = contract(A,B,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end


"""
    C = contract(A,iA,B,iB[,Z,alpha=1,beta=1])

contraction of two `dtens` tensors `A` and `B` over indices `iA` and `iB`; can perform the contraction `alpha`*`A`*`B` + `beta`*`Z` with other inputs
"""
function contract(A::dtens,iA::intvecType,B::dtens,iB::intvecType,Z::dtens...)#;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))

  if length(Z) > 0
    C = contract(A[0],iA,B[0],iB,Z[1][0])#,alpha=alpha,beta=beta)
    dC = contract(A[1],iA,B[1],iB,Z[1][1])#,alpha=alpha,beta=beta)
    out = dtens(C,contract(A[1],iA,B[0],iB,dC))
  else
    C = contract(A[0],iA,B[0],iB)#,alpha=alpha)
    dC = contract(A[1],iA,B[1],iB)#,alpha=alpha)
    out = dtens(C,contract(A[1],iA,B[0],iB,dC))
  end
  return out
end
