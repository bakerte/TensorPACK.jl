#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

"""
  Module: contractions

Contract two tensors together

See also: [`decompositions`](@ref)
"""
#=
module contractions
using ..tensor
import LinearAlgebra
=#



#       +------------------------+
#>------|    Matrix multiply     |---------<
#       +------------------------+
"""
  C = libmult([alpha,]C,D[,beta,Z])

Chooses the best matrix multiply function for tensor contraction (dense tensor or sub-block) withoutput matrix `C`

+ Outputs `C` * `D` if `alpha`=1
+ Outputs `alpha` * `C` * `D` if `Z` is not input
+ Outputs `alpha` * `C` * `D` + `beta` * `Z` if `Z` is input
"""
function libmult! end

"""
    libmult(tA, tB, alpha, A, B)
Return `alpha*A*B` or the other three variants according to [`tA`](@ref stdlib-blas-trans) and `tB`.
"""
libmult(tA, tB, alpha, A, B)

"""
    libmult(tA, tB, alpha, A, B, beta, C)
Return `alpha*A*B+beta*C` or the other three variants according to [`tA`](@ref stdlib-blas-trans) and `tB`.
"""
libmult(tA, tB, alpha, A, B,beta,C)

"""
    libmult(tA, tB, A, B)
Return `A*B` or the other three variants according to [`tA`](@ref stdlib-blas-trans) and `tB`.
"""
libmult(tA, tB, A, B)

"""
  permq(A,iA)

Answers the question of whether to permute the tensor `A` with contracted indices `iA`
"""
function permq(A::densTensType,iA::Union{Array{intType,1},NTuple{K,intType}}) where K
  nopermL = true
  w = 0
  while nopermL && w < length(iA)
    w += 1
    nopermL = iA[w] == w
  end
  nopermR = !nopermL
  if nopermR
    w = length(iA)
    end_dim = ndims(A)
    while nopermR && w > 0
      nopermR = iA[w] == end_dim
      end_dim -= 1
      w -= 1
    end
  end
  return nopermL,nopermR
end

"""
  willperm(conjA,WA,AnopermL,AnopermR)

Determines LAPACK flag based on output of `permq` and whether to conjugate `conjA` and what type `WA`
"""
function willperm(conjA::Bool,WA::DataType,AnopermL::Bool,AnopermR::Bool)
  if AnopermL
    transA = conjA && WA <: Complex ? 'C' : 'T'
    Aperm = true
  else
    transA = 'N'
    Aperm = AnopermR
  end
  return Aperm,transA
end

"""
  prepareT(A,Lvec,Rvec,conjvar)

Converts input tensor `A` to its matrix equivalent with left indices contained in `Lvec` and right indices contained in `Rvec` and whether to conjugate (`conjvar`)
"""
function prepareT(A::densTensType,Lvec::Union{Array{intType,1},NTuple{K,intType}},Rvec::Union{Array{intType,1},NTuple{P,intType}},conjvar::Bool) where {K,P}
  newdimsA = Array{intType,1}(undef,ndims(A))
  counter = 0
  @inbounds @simd for w = 1:length(Lvec)
    counter += 1
    newdimsA[counter] = Lvec[w]
  end
  @inbounds @simd for w = 1:length(Rvec)
    counter += 1
    newdimsA[counter] = Rvec[w]
  end
  pA = permutedims(A,newdimsA)
  if conjvar && eltype(A) <: Complex
    conj!(pA)
  end
  return pA
end

"""
  getsizes(A,iA,AAsizes,counter)

Finds sizes of the matrix equivalents for tensor `A`, contracted indices `iA`, sizes of the new tensor `AAsizes`, and a `counter` which increments between the two tensors being contracted over
"""
function getsizes(A::TensType,iA::intvecType#=,AAsizes::Array{intType,1},counter::intType=#)
  Lsize = innersizeL = 1
  @inbounds for w = 1:ndims(A)
    if !(w in iA)
#      counter += 1
#      AAsizes[counter] = size(A,w)
      Lsize *= size(A,w)
    else
      innersizeL *= size(A,w)
    end
  end
  return Lsize,innersizeL#,AAsizes
end
#=
function getsizes(A::TensType,iA::intvecType,AAsizes::NTuple{G,intType}) where G
  Lsize = innersizeL = 1
  @inbounds for w = 1:ndims(A)
    if !(w in iA)
#      counter += 1
      AAsizes = (AAsizes...,size(A,w)) #AAsizes[counter] = size(A,w)
      Lsize *= size(A,w)
    else
      innersizeL *= size(A,w)
    end
  end
  return Lsize,innersizeL,AAsizes
end
=#

function makesize(w,A,iA,B,iB)
  counter = 0
  a = 0
  if w <= ndims(A)-length(iA)
    while counter < w
      a += 1
      notinset = true
      p = 0
      while notinset && p < length(iA)
        p += 1
        notinset = a != iA[p]
      end
      if notinset
        counter += 1
      end
    end
    return size(A,a)
  else
    while counter < w-ndims(A)+length(iA)
      a += 1
      notinset = true
      p = 0
      while notinset && p < length(iB)
        p += 1
        notinset = a != iB[p]
      end
      if notinset
        counter += 1
      end
    end
    return size(B,a)
  end
end

"""
  maincontractor!(conjA,conjB,A,iA,B,iB,Z...;alpha=1,beta=1)

Contraction function (`alpha`*`A`*`B`+`beta`*`Z`) and can conjugate A (`conjA`) or B (`conjB`). Does not construct return tensor, just outputs a vector.
"""
function maincontractor!(conjA::Bool,conjB::Bool,A::densTensType,iA::intvecType,B::densTensType,iB::intvecType,Z::TensType...;alpha::Number=1,beta::Number=1)

#  testval = prod(w->size(A,iA[w]),1:length(iA))==prod(w->size(B,iB[w]),1:length(iB))
#  @test(testval,"not matching sizes for input tensors to contraction")
#a,b = prod(w->size(A,iA[w]),1:length(iA)),prod(w->size(B,iB[w]),1:length(iB))
#test(a,b,"not matching sizes for input tensors to contraction")

  AnopermL,AnopermR = permq(A,iA)
  BnopermL,BnopermR = permq(B,iB)

  Aperm,transA = willperm(conjA,eltype(A),AnopermL,AnopermR)
  Bperm,transB = willperm(conjB,eltype(B),BnopermR,BnopermL)

  Aremain = ndims(A)-length(iA)
  Bremain = ndims(B)-length(iB)

#  AAsizes = Array{intType,1}(undef,Aremain+Bremain)

  Lsize,innersizeL = getsizes(A,iA)
  Rsize,innersizeR = getsizes(B,iB)

  AAsizes = Aremain == 0 && Bremain == 0 ? intType[0] : intType[makesize(w,A,iA,B,iB) for w = 1:Aremain+Bremain]

  if Aperm && Bperm
    mulA = conjA && eltype(A) <: Complex && !AnopermL ? conj(A) : A
    mulB = conjB && eltype(B) <: Complex && !BnopermR ? conj(B) : B
  elseif Aperm
    mulA = conjA && eltype(A) <: Complex && !AnopermL ? conj(A) : A
    notvecB = findnotcons(ndims(B),iB)
    mulB = prepareT(B,iB,notvecB,conjB)
  elseif Bperm
    notvecA = findnotcons(ndims(A),iA)
    mulA = prepareT(A,notvecA,iA,conjA)
    mulB = conjB && eltype(B) <: Complex && !BnopermR ? conj(B) : B
  else
    notvecA = findnotcons(ndims(A),iA)
    mulA = prepareT(A,notvecA,iA,conjA)
    notvecB = findnotcons(ndims(B),iB)
    mulB = prepareT(B,iB,notvecB,conjB)
  end

  if length(Z) > 0
    outType = typeof(eltype(A)(1)*eltype(B)(1)*eltype(Z[1])(1)*typeof(alpha)(1)*typeof(beta)(1))
    type_alpha = convert(outType,alpha)
    type_beta = convert(outType,beta)
    out = libmult(transA,transB,type_alpha,mulA,mulB,type_beta,Z[1],Lsize,innersizeL,innersizeR,Rsize)
  elseif isapprox(alpha,1)
    out = libmult(transA,transB,mulA,mulB,Lsize,innersizeL,innersizeR,Rsize)
  else
    outType = typeof(eltype(A)(1)*eltype(B)(1)*typeof(alpha)(1))
    if outType != typeof(alpha)
      type_alpha = convert(outType,alpha)
    else
      type_alpha = alpha
    end
    out = libmult(transA,transB,type_alpha,mulA,mulB,Lsize,innersizeL,innersizeR,Rsize)
  end

  return AAsizes,out
end

"""
  maincontractor(conjA,conjB,A,iA,B,iB,Z...;alpha=1,beta=1)

Contraction function (`alpha`*`A`*`B`+`beta`*`Z`) and can conjugate A (`conjA`) or B (`conjB`)
"""
function maincontractor(conjA::Bool,conjB::Bool,A::densTensType,iA::intvecType,B::densTensType,iB::intvecType,Z::TensType...;alpha::Number=1,beta::Number=1)
  AAsizes,out = maincontractor!(conjA,conjB,A,iA,B,iB,Z...,alpha=alpha,beta=beta)
  if typeof(A) <: denstens || typeof(B) <: denstens
    outTens = tens(AAsizes,out)
  else
    outTens = reshape!(out,AAsizes)
  end
  return outTens
end

#       +----------------------------+
#>------| Contraction function calls |---------<
#       +----------------------------+


"""
  dot(A,B;Lfct=adjoint,Rfct=identity)

takes `identity` or `adjoint` (or equivalently `conj`) for the `Lfct` and `Rfct`

See also: [`identity`](@ref) [`adjoint`](@ref) [`conj`](@ref)
"""
function dot(inA::densTensType,inB::densTensType;Lfct::Function=adjoint,Rfct::Function=identity,transA::Bool=true)
  A = typeof(inA) <: denstens ? inA.T : inA
  B = typeof(inB) <: denstens ? inB.T : inB

  newtypeAB = typeof(eltype(A)(1) * eltype(B)(1))
  val = newtypeAB(0)
  dim1 = length(A) #size(inA,transA ? 1 : 2)
  @inbounds @simd for j = 1:dim1
    val += Lfct(A[j]) * Rfct(B[j])
  end
  return val
end

"""
  

The function will admit any dimension or input element type in terms of the arrays `A`, `H`, and `B`. However, the function will only work properly when the total elements of `A` times those in `B` equal the elements in `H`

If more operators `H` should be contracted between `A` and `B`, then it is advised here to contract them first before using this function

"""
function dot(inA::densTensType,inH::densTensType,inB::densTensType;Lfct::Function=adjoint,Rfct::Function=identity,transA::Bool=true)

  A = typeof(inA) <: denstens ? inA.T : inA
  H = typeof(inH) <: denstens ? inH.T : inH
  B = typeof(inB) <: denstens ? inB.T : inB

  newtypeAHB = typeof(eltype(A)(1) * eltype(H)(1) * eltype(B)(1))
  val = newtypeAHB(0)

  dim1 = length(A) #size(inA,transA ? 1 : 2)
  dim2 = length(B) #size(inB,1)
  newtypeAH = typeof(eltype(A)(1) * eltype(H)(1))
  @inbounds for j = 1:dim2
    ival = newtypeAH(0)
    savedim = dim1*(j-1)
    @inbounds @simd for i = 1:dim1
      ival += Lfct(A[i]) * H[i + savedim]
    end
    val += ival * Rfct(B[j])
  end
  return val
end
export dot


function *(X::densTensType,Y::densTensType)
  if !(typeof(X) <: Diagonal) && !(typeof(Y) <: Diagonal)
    m = ndims(X) == 1 ? 1 : prod(w->size(X,w),1:ndims(X)-1)
    k = size(X,ndims(X))
    n = ndims(Y) < 2 ? 1 : prod(w->size(Y,w),2:ndims(Y))
    out = libmult('N','N',X,Y,m,k,k,n)

    A,B = X,Y
    iA = (ndims(X),)
    iB = (1,)
    Aremain = ndims(A)-1
    Bremain = ndims(B)-1
    AAsizes = [makesize(w,A,iA,B,iB) for w = 1:Aremain+Bremain]#ntuple(w->makesize(w,A,iA,B,iB),Aremain+Bremain)
    if typeof(X) <: denstens || typeof(Y) <: denstens
      finaltens = tens(AAsizes,out)
    else
      finaltens = reshape!(out,AAsizes...)
    end
  else
    finaltens = dmul(X,Y)
  end
  return finaltens
end

function *(X::LinearAlgebra.Diagonal{R, Vector{R}},Y::tens{W}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds @simd for x = 1:size(X,1)
      Z[x + tempind] = X[x,x]*Y.T[x + tempind]
    end
  end
  return tens{outType}(Y.size,Z)
end

function *(Y::tens{W},X::Diagonal{R}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for x = 1:longdim
    tempind = x - longdim
    @inbounds @simd for y = 1:size(X,1)
      zval = tempind + longdim*y
      Z[zval] = Y.T[zval]*X[y,y]
    end
  end
  return tens{outType}(Y.size,Z)
end



#=

function *(X::LinearAlgebra.UpperTriangular{R, Matrix{R}},Y::tens{W}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds for x = 1:size(X,1)
      val = outType(0)
      @inbounds @simd for k = x:size(X,2)
        val += X[x,k]*Y.T[k + tempind]
      end
      Z[x + tempind] = val
    end
  end
  return tens{outType}(Y.size,Z)
end

function *(Y::tens{W},X::LinearAlgebra.UpperTriangular{R, Matrix{R}}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for x = 1:size(X,1)
    tempind = x - size(X,1)
    @inbounds for y = 1:longdim
      val = outType(0)
      @inbounds @simd for k = 1:x
        val += Y.T[tempind + size(X,1)*k]*X[k,y]
      end
      zval = tempind + size(X,1)*y
      Z[zval] = val
    end
  end
  return tens{outType}(Y.size,Z)
end



function *(X::LinearAlgebra.LowerTriangular{R, Matrix{R}},Y::tens{W}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds for x = 1:size(X,1)
      val = outType(0)
      @inbounds @simd for k = 1:x
        val += X[x,k]*Y.T[k + tempind]
      end
      Z[x + tempind] = val
    end
  end
  return tens{outType}(Y.size,Z)
end

function *(Y::tens{W},X::LinearAlgebra.LowerTriangular{R, Matrix{R}}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for x = 1:size(X,1)
    tempind = x - size(X,1)
    @inbounds for y = 1:longdim
      val = outType(0)
      @inbounds @simd for k = x:size(X,2)
        val += Y.T[tempind + size(X,1)*k]*X[k,y]
      end
      zval = tempind + size(X,1)*y
      Z[zval] = val
    end
  end
  return tens{outType}(Y.size,Z)
end
=#



#import LinearAlgebra.rmul!
function dmul!(Y::tens{R},X::Diagonal{W}) where {R <: Number, W <: Number}
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:size(X,1)
    zval = longdim*(y-1)
    val = X[y,y]
    @inbounds @simd for x = 1:longdim
      Y.T[x + zval] *= val
    end
  end
  return Y
end

function dmul!(Y::Array{R,N},X::Diagonal{W}) where {R <: Number, N, W <: Number}
  longdim = cld(length(Y),size(X,1))
  for y = 1:size(X,1)
    zval = longdim*(y-1)
    val = X[y]
    @inbounds @simd for x = 1:longdim
      Y[x + zval] *= val
    end
  end
  return Y
end
#=
function rmul!(Y::AbstractArray{W,N},X::LinearAlgebra.Diagonal{W, Vector{W}}) where {R <: Number, W <: Number, N}
  return LinearAlgebra.rmul!(Y,X)
end
=#
function dmul!(X::R,Y::tens{W}) where {R <: Number, W <: Number}
  return tensorcombination!((X,),Y)
end
function dmul!(Y::tens{W},X::R) where {R <: Number, W <: Number}
  return dmul!(X,Y)
end
#export rmul!

#import LinearAlgebra.lmul!
function dmul!(X::Diagonal{R},Y::tens{W}) where {R <: Number, W <: Number}
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds @simd for x = 1:size(X,1)
      Y.T[x + tempind] *= X[x]
    end
  end
  return Y
end

function dmul!(X::Diagonal{R},Y::Array{W,N}) where {R <: Number, W <: Number, N}
  longdim = cld(length(Y),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds @simd for x = 1:size(X,1)
      Y[x + tempind] *= X[x]
    end
  end
  return Y
end

function dmul(X::Diagonal{R},Y::densTensType) where {R <: Number}
  return dmul!(X,copy(Y))
end

function dmul(X::densTensType,Y::Diagonal{R}) where {R <: Number}
  return dmul!(copy(X),Y)
end
#=
function lmul!(X::LinearAlgebra.Diagonal{R, Vector{R}},Y::AbstractArray{W,N}) where {R <: Number, W <: Number, N}
  return LinearAlgebra.lmul!(Y,X)
end
=#
#=
function dmul!(Y::TensType,X::R) where {R <: Number}
  return tensorcombination!((X,),Y)
end
function dmul!(X::R,Y::tens{W}) where {R <: Number, W <: Number}
  return dmul!(Y,X)
end
=#
export dmul!


function contract(A::Diagonal{W},B::densTensType;alpha::Number=eltype(A)(1)) where W <: Number
  return trace(isapprox(alpha,1) ? A*B : alpha*A*B)
end

function contract!(A::Diagonal{W},B::densTensType;alpha::Number=eltype(A)(1)) where W <: Number
  out = dot(A,B,Lfct=identity,Rfct=identity)
  return  alpha*out
end
#=
function contract(A::Union{TensType,LinearAlgebra.Diagonal{W, Vector{W}}},iA::intvecType,B::Union{TensType,LinearAlgebra.Diagonal{W, Vector{W}}},iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1)) where W <: Number
  return diagcontract(A,convIn(iA),B,convIn(iB),Z...,alpha=alpha,beta=beta)
end
=#

function contract(A::Diagonal{W},iA::intvecType,B::densTensType,iB::intvecType,Z::densTensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1)) where W <: Number
  return diagcontract!(A,convIn(iA),B,convIn(iB),Z...,alpha=alpha,beta=beta,inplace=false)
end

function contract(A::densTensType,iA::intvecType,B::Diagonal{W},iB::intvecType,Z::densTensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1)) where W <: Number
  return diagcontract!(A,convIn(iA),B,convIn(iB),Z...,alpha=alpha,beta=beta,inplace=false)
end

function diagcontract!(A::Diagonal{W},iA::intvecType,B::densTensType,iB::intvecType,Z::densTensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1),inplace::Bool=true) where W <: Number

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

  if Lperm
    C = inplace ? dmul!(A,B) : dmul(A,B)
  elseif Rperm
    C = inplace ? dmul!(B,A) : dmul(B,A)
  else
    notvecB = findnotcons(ndims(B),iB)
    mB = prepareT(B,iB,notvecB,false)
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

function diagcontract!(A::densTensType,iA::intvecType,B::Diagonal{W},iB::intvecType,Z::densTensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1),inplace::Bool=true) where W <: Number

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

  if Lperm
    C = inplace ? dmul!(B,A) : dmul(B,A)
  elseif Rperm
    C = inplace ? dmul!(A,B) : dmul(A,B)
  else
    notvecA = findnotcons(ndims(A),iA)
    mA = prepareT(A,notvecA,iA,false)
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
export diagcontract!

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
  ccontractc(A,B[,alpha=])

Similar to contract but 'A' and 'B' are conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontractc(A::TensType,B::TensType;alpha::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  out = dot(mA,mB,Lfct=adjoint,Rfct=adjoint)
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
    ccontract(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A` with one of the inputs conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontract(A::TensType;alpha::Number=eltype(A)(1))
  out = dot(A,A,Lfct=adjoint,Rfct=identity)
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
    ccontractc(A[,alpha=])

Self-contraction of input tensor `A` to a scalar value of `A`.`A` with both inputs conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['contract'](@ref)
"""
function ccontractc(A::TensType;alpha::Number=eltype(A)(1))
  out = dot(A,A,Lfct=adjoint,Rfct=adjoint)
  return  alpha*out
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
  mA,mB = checkType(A,B)
  return maincontractor(false,false,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
  contract(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contract(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iB = [w for w = 1:ndims(B)]#ntuple(w->w,ndims(B))
  return maincontractor(false,false,mA,convIn(iA),mB,iB,Z...,alpha=alpha,beta=beta)
end

"""
  contract(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contract(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iA = [w for w = 1:ndims(A)] #ntuple(w->w,ndims(A))
  return maincontractor(false,false,mA,iA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
    ccontract(A,iA,B,iB[,Z,alpha=,beta=])

Similar to contract but 'A' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontract(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return maincontractor(true,false,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
  ccontract(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontract(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iB = [w for w = 1:ndims(B)] #ntuple(w->w,ndims(B))
  return maincontractor(true,false,mA,convIn(iA),mB,iB,Z...,alpha=alpha,beta=beta)
end

"""
  ccontract(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontract(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iA = [w for w = 1:ndims(A)] #ntuple(w->w,ndims(A))
  return maincontractor(true,false,mA,iA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
    contractc(A,iA,B,iB[,Z,alpha=,beta=])

Similar to contract but 'B' is conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function contractc(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return maincontractor(false,true,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
  contractc(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contractc(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iB = [w for w = 1:ndims(B)] #ntuple(w->w,ndims(B))
  return maincontractor(false,true,mA,convIn(iA),mB,iB,Z...,alpha=alpha,beta=beta)
end

"""
  contractc(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iA` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function contractc(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iA = [w for w = 1:ndims(A)] #ntuple(w->w,ndims(A))
  return maincontractor(false,true,mA,iA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
    ccontractc(A,iA,B,iB[,Z,alpha=,beta=])

Similar to contract but 'A' and 'B' are conjugated

See also: ['ccontract'](@ref) ['contractc'](@ref) ['ccontractc'](@ref)
"""
function ccontractc(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  return maincontractor(true,true,mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end
export contract,ccontract,contractc,ccontractc

"""
  ccontractc(A,iA,B[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontractc(A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iB = [w for w = 1:ndims(B)] #ntuple(w->w,ndims(B))
  return maincontractor(true,true,mA,convIn(iA),mB,iB,Z...,alpha=alpha,beta=beta)
end

"""
  ccontractc(A,B,iB[,Z,alpha=,beta=])

same as `contract(A,iA,B,iB)` but `iB` is over all indices.

See also: [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref)
"""
function ccontractc(A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  iA = [w for w = 1:ndims(A)] #ntuple(w->w,ndims(A))
  return maincontractor(true,true,mA,iA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
end

"""
    contract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contract(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contract(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,iA,mB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contract(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,mB,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`ccontract`](@ref) to `order`
"""
function ccontract(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = ccontract(mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontract(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,convIn(iA),mB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontract(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontract(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contractc`](@ref) to `order`
"""
function contractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contractc(mA,convIn(iA),mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,iA,mB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    contractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function contractc(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = contract(mA,mB,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`ccontractc`](@ref) to `order`
"""
function ccontractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = ccontractc(mA,iA,mB,iB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontractc(order::intvecType,A::TensType,iA::intvecType,B::TensType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = ccontractc(mA,convIn(iA),mB,Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    ccontractc(order,A,iA,B,iB[,Z,alpha=,beta=])

Permutes result of [`contract`](@ref) to `order`
"""
function ccontractc(order::intvecType,A::TensType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)
  newT = ccontractc(mA,mB,convIn(iB),Z...,alpha=alpha,beta=beta)
  return permutedims!(newT,convIn(order))
end

"""
    trace!(A,iA)

Computes trace of `A` over indices `iA` by first making an identity tensor among those indices; `iA` is a vector of numbers (ex: [1,3]) or a vector of 2-element vectors ([[1,2],[3,4],[5,6]])

# Example:

```julia
julia> A = [1 0;0 -1];
julia> trace(A,[1,2])
0-dimensional Array{Float64,0}:
0.0

julia> A = ones(10,20,10,20);
julia> trace(A,[[1,3],[2,4]])
0-dimensional Array{Float64,0}:
200.0
```
"""
function trace!(A::TensType,iA::Array{NTuple{2,P},1}) where P <: Integer
  Id = makeId(A,iA)
  conA = (iA[1]...,)
  for w = 2:length(iA)
    conA = (conA...,iA[w]...)
  end

  conId = ntuple(w->w,2*length(iA))

#  permutedims!(A,conA)
  return contract(A,conA,Id,conId)
end

function trace!(A::TensType,iA::Array{Array{P,1},1}) where P <: Integer
  return trace!(A,[convIn(iA[w]) for w = 1:length(iA)])
end

function trace!(A::TensType,iA::Array{P,1}...) where P <: Integer
  return trace!(A,[convIn(iA[w]) for w = 1:length(iA)])
end

"""
    trace!(A)

Find trace of a matrix `A`
"""
function trace!(A::TensType)
  return sum(w->searchindex(A,w,w),1:size(A,1))
end
export trace!

"""
    trace(A,iA)

Computes trace of `A` (copying `A`) over indices `iA` by first making an identity tensor among those indices; `iA` is a vector of numbers (ex: [1,3]) or a vector of 2-element vectors ([[1,2],[3,4],[5,6]])

# Example:

```julia
julia> A = [1 0;0 -1];
julia> trace(A,[1,2])
0-dimensional Array{Float64,0}:
0.0

julia> A = ones(10,20,10,20);
julia> trace(A,[[1,3],[2,4]])
0-dimensional Array{Float64,0}:
200.0
```
"""
function trace(A::TensType,iA::R...) where R <: Union{Array{P,1},Array{Array{P,1},1},Array{NTuple{2,P},1}} where P <: Integer
  return trace!(copy(A),iA...)
end
export trace
