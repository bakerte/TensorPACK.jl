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
  exp!(A,prefactor)

exponentiate (exp(`prefactor`*`A`)) a matrix or `denstens` (also matrix)
"""
function exp!(A::Array{W,2},prefactor::Number) where W <: Number
  if !isapprox(prefactor,1)
    if W == typeof(prefactor)
      for x = 1:size(A,1)
        @inbounds @simd for y = 1:size(A,2)
          A[x,y] *= prefactor
        end
      end
    else
      A = A*prefactor
    end
  end
  expA = exp(A)
  return expA
end

function exp!(A::tens{W},prefactor::Number) where W <: Number
  X = reshape(A.T,A.size)
  expX = exp!(X,prefactor)
  newtype = typeof(W(1)*prefactor)
  if newtype == W
    A.T = reshape(expX,prod(size(A)))
  else
    A = tens(exp(expX,prefactor))
  end
  return A
end

"""
  exp!(A)

exponentiate (exp(`A`)) a matrix or `denstens` (also matrix)
"""
function exp!(A::Array{W,2}) where W <: Number
  return exp!(A,W(1))
end

function exp!(A::tens{W}) where W <: Number
  return exp!(A,W(1))
end
export exp!

import Base.exp
function exp(A::Array{W,2},prefactor::Number) where W <: Number
#  if W <: Real && typeof(prefactor) <: Real
    newexpA = copy(A)
#  else
#    newexpA = 
#  end
  return exp!(newexpA,prefactor)
end

"""
  G = exp(A)

Exponentiate a matrix `A` from the `denstens` type with output `G`
"""
function exp(A::tens{W},prefactor::Number) where W <: Number
  X = reshape(A.T,A.size)
  newtype = typeof(W(1)*prefactor)
  return tens(newtype,exp(X,prefactor))
end

function exp(A::tens{W}) where W <: Number
  return exp(A,W(1))
end

"""
  G = exp(alpha,beta)

Exponentiate a tridiagonal matrix from the two lists `alpha` (diagonal elements) and `beta` (off-diagonal elements) type with output `G`
"""
function exp(alpha::Array{W,1},beta::Array{Y,1},prefactor::Number) where {W <: Number, Y <: Number}
  d = length(alpha)
  if Y <: Complex || W <: Complex
    G = zeros(typeof(Y(1)*W(1)),length(alpha),length(alpha))
    @inbounds @simd for i = 1:d
      G[i,i] = alpha[i]
    end
    @inbounds @simd for i = 1:d-1
      G[i,i+1] = beta[i]
      G[i+1,i] = conj(beta[i])
    end
  else  
    G = LinearAlgebra.SymTridiagonal(alpha,beta)
  end
  return exp(G,prefactor)
end

#best implementation
"""
  G = exp(A)

Exponentiate a symmetric, tridiagonal matrix `A` with output `G`
"""
function exp(G::LinearAlgebra.SymTridiagonal{W, Vector{W}},prefactor::Number) where W <: Number
  D,U = LinearAlgebra.eigen(G)
  if typeof(prefactor) == W
    @inbounds for i = 1:length(D)
      D[i] = exp(prefactor * D[i])
    end
    out = U*Diagonal(D)*U'
  else
    D = [exp(D[i]*prefactor) for i = 1:length(D)]
    newtype = typeof(typeof(prefactor)(1)*eltype(G)(1))
    convU = convert(Array{newtype,2},U)
    out = convU * Diagonal(D) * convU'
  end
  return out
end

function exp(alpha::Array{W,1},beta::Array{Y,1}) where {W <: Number, Y <: Number}
  return exp(alpha,beta,W(1))
end

function exp(G::LinearAlgebra.SymTridiagonal{W, Vector{W}}) where W <: Number
  return exp(G,W(1))
end


#decomposition of unitaries, so can do block by block
function exp!(C::Qtens{W,Q},prefactor::Number) where {W <: Number, Q <: Qnum}
  if W == typeof(prefactor)
    A = changeblock(C,C.currblock[1],C.currblock[2])
    @inbounds for q = 1:length(A.T)
      A.T[q] = exp(A.T[q],prefactor)
    end
    B = A
  else
    B = exp(C,prefactor)
  end
  return B
end

function exp!(C::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return exp!(C,W(1))
end

import Base.exp
function exp(C::Qtens{W,Q},prefactor::Number) where {W <: Number, Q <: Qnum}
  A = changeblock(C,C.currblock)

  newtype = typeof(W(1)*prefactor)
  newT = Array{W,1}(undef,length(A.T))
  @inbounds @simd for q = 1:length(A.T)
    newT[q] = exp(A.T[q],prefactor)
  end

  newsize = Array{intType,1}(undef,length(A.size))
  @inbounds @simd for w = 1:length(newsize)
    newsize[w] = A.size[w] #could be copy(A.size[w]) but this is never changing
  end
  newcurrblock = A.currblock
  newind = A.ind
  newQblocksum = A.Qblocksum
  newMat = A.QnumMat
  newSum = A.QnumSum
  newflux = copy(A.flux)

  return Qtens{newtype,Q}(newsize,newT,newind,newcurrblock,newQblocksum,newMat,newSum,newflux)
end

function exp(C::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return exp(C,W(1))
end

