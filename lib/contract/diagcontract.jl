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
    C = diagcontract!(conjA,conjB,A,iA,B,iB[,Z,alpha=1,beta=1,inplace=true])

Generalization of a*x+y but for tensors (`C` = `alpha` * `A` * `B` + `beta` * `Z`) over indices `iA` and `iB` where `A` is a `diagonal` type and `B` is a `TensType`

See also: [`maincontractor`](@ref)
"""
function diagcontract!(conjA::Bool,conjB::Bool,A::Diagonal{W},iA::intvecType,B::Union{denstens,Array},iB::intvecType,Z::densTensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1),inplace::Bool=true) where W <: Number #, R <: denstens

  if prod(w->size(A,iA[w]),1:length(iA)) != prod(w->size(B,iB[w]),1:length(iB))
    error("not matching sizes for input tensors to contraction [A size ",size(A)," on indices ",iA," and B size ",size(B)," on ",iB,"]")
  end

  Lperm = true
  w = 0
  while Lperm && w < length(iB)
    w += 1
    Lperm = iB[w] == w
  end

  Rperm = true
  w = length(iB)
  while Rperm && w > 0
    Rperm = iB[w] == w
    w -= 1
  end

  if conjA
    A = eltype(A) <: Real ? A : conj(A)
  end

  if Lperm || Rperm
    B = eltype(B) <: Real ? B : conj(B)
  end

  if Lperm
    C = inplace ? dmul!(A,B) : dmul(A,B)
  elseif Rperm
    C = inplace ? dmul!(B,A) : dmul(B,A)
  else
    notvecB = findnotcons(ndims(B),iB)
    mB = prepareT(B,iB,notvecB,conjB)
    C = inplace ? dmul!(A,mB) : dmul(A,mB)
  end
  if length(Z) == 0
    if isapprox(alpha,1)
      out = C
    else
      out = dmul!(alpha,C)
    end
  else
    out = tensorcombination!((alpha,beta),C,Z[1])
  end
  return out
end

"""
    C = diagcontract!(conjA,conjB,A,iA,B,iB[,Z,alpha=1,beta=1,inplace=true])

Generalization of a*x+y but for tensors (`C` = `alpha` * `A` * `B` + `beta` * `Z`) over indices `iA` and `iB` where `A` is a `TensType` type and `B` is a `diagonal`

See also: [`maincontractor`](@ref)
"""
function diagcontract!(conjA::Bool,conjB::Bool,A::Union{denstens,Array},iA::intvecType,B::Diagonal{W},iB::intvecType,Z::densTensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1),inplace::Bool=true) where W <: Number

  if prod(w->size(A,iA[w]),1:length(iA)) != prod(w->size(B,iB[w]),1:length(iB))
    error("not matching sizes for input tensors to contraction [A size ",size(A)," on indices ",iA," and B size ",size(B)," on ",iB,"]")
  end

  Lperm = true
  w = 0
  while Lperm && w < length(iA)
    w += 1
    Lperm = iA[w] == w
  end

  Rperm = true
  w = length(iA)
  while Rperm && w > 0
    Rperm = iA[w] == w
    w -= 1
  end

  if conjB
    B = eltype(B) <: Real ? B : conj(B)
  end

  if Lperm || Rperm
    A = eltype(A) <: Real ? A : conj(A)
  end

  if Lperm
    C = inplace ? dmul!(B,A) : dmul(B,A)
  elseif Rperm
    C = inplace ? dmul!(A,B) : dmul(A,B)
  else
    notvecA = findnotcons(ndims(A),iA)
    mA = prepareT(A,notvecA,iA,conjA)
    C = inplace ? dmul!(mA,B) : dmul(mA,B)
  end
  if length(Z) == 0
    if isapprox(alpha,1)
      out = C
    else
      out = dmul!(alpha,C)
    end
  else
    out = tensorcombination!((alpha,beta),C,Z[1])
  end
  return out
end

"""
    C = diagcontract!(conjA,conjB,A,iA,B,iB[,Z,alpha=1,beta=1,inplace=true])

Generalization of a*x+y but for tensors (`C` = `alpha` * `A` * `B` + `beta` * `Z`) over indices `iA` and `iB` where `A` is a `diagonal` type and `B` is a `diagonal`

See also: [`maincontractor`](@ref)
"""
function diagcontract!(conjA::Bool,conjB::Bool,A::Diagonal{W},iA::intvecType,B::Diagonal{R},iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1),inplace::Bool=true) where {W <: Number, R <: Number}

  if length(A) != length(B)
    error("Contracting diagonal types together must have equal sizes")
  end

  Lfct = conjA ? adjoint : identity
  Rfct = conjB ? adjoint : identity

  newtype = typeof(W(1)*R(1))
  out = Array{newtype,1}(undef,length(A))
  for w = 1:length(A)
    out[w] = Lfct(A[w])*Rfct(B[w])
  end

  answer = Diagonal(out)
  if length(Z) == 1
    answer += Z[1]
  end
  return answer
end

#=
function diagcontract(A::diagonal{W},iA::intvecType,B::densTensType,iB::intvecType,Z::densTensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1),inplace::Bool=false) where W <: Number
  return diagcontract!(A,iA,B,iB,Z...,alpha=alpha,beta=beta,inplace=inplace)
end
=#
