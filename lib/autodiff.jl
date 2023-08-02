
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker and Aaron Dayton (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

#Based on a tutorial by Prof. Alan Edelman at https://www.youtube.com/watch?v=vAp6nUMrKYg
#=
import Base: +, -, /, *, ^, log, conj, conj!, abs, abs2, convert, promote_rule, zeros, transpose, sin, cos
import LinearAlgebra: norm
# +(x::dualnum,y::dualnum) = dualnum(x.num .+ y.num)
# -(x::dualnum,y::dualnum) = dualnum(x.num .- y.num)
# -(x::dualnum) = dualnum(-1) * x
# /(x::dualnum,y::dualnum) = dualnum((x.num[1]/y.num[1],(y.num[1]*x.num[2]-x.num[1]*y.num[2])/y.num[1]^2))
# *(x::dualnum,y::dualnum) = dualnum((x.num[1]*y.num[1],(y.num[1]*x.num[2]+x.num[1]*y.num[2])))
# ^(x::dualnum,y::Number) = dualnum((x.num[1]^y,(y*x.num[1]^(y-1)*x.num[2])))
# sqrt(x::dualnum) = x^0.5
# conj(x::dualnum) = dualnum((conj(x.num[1]),conj(x.num[2])))
# conj!(x::dualnum) = conj(x)
# abs2(x::dualnum) = x*conj(x)
# abs(x::dualnum) = sqrt(abs2(x))

# TODO - Implement these as well!
# convert(::Type{dualnum},x::Union{Real,Complex}) = dualnum((x,zero(x))) #how to do complex numbers?
# promote_rule(::Type{dualnum}, ::Type{<:Number}) = dualnum
=#

"""
  dualnum{W}

stores a value (`val`) and a gradient (`grad`) for each number. 
"""
struct dualnum{W <: Number} <: Number
  val::W
  grad::W
end

"""
  dualnum(val[,grad=1.])

Initializes the `dualnum` type with value `val` and a gradient `grad` (default value 1)
"""
function dualnum(val::W;grad::W=W(1)) where W <: Number
  return dualnum{W}(val, grad)
end

"""
  dualnum(val,grad)

Initializes the `dualnum` type with value `val` and a gradient `grad`
"""
function dualnum(val::W, grad::W) where W <: Number
  return dualnum(val, grad=grad)
end
export dualnum

import Base.zero
"""
  zero(x)

Creates the zero value for a given `dualnum`, `x`
"""
function zero(x::dualnum{W}) where W <: Number
  return dualnum{W}(W(0),W(1))
end

"""
  dualnum(W)

Creates a `dualnum` of type `W`
"""
function dualnum(W::DataType)
  return dualnum{W}(W(0),W(1))
end
export dualnum

############
# Addition #
############

import Base.+
"""
  +(x,y)

Addition of two `dualnum`s
"""
function +(x::dualnum, y::dualnum)
  r = x.val + y.val
  g = x.grad .+ y.grad
  return dualnum(r, g)
end

"""
  +(x,y)

Addition of a `dualnum` and a regular number `y` (commutes)
"""
function +(x::dualnum, y::Number)
  return dualnum(x.val + y, x.grad)
end
+(x::Number, y::dualnum) = y+x

###############
# Subtraction #
###############

import Base.-
"""
  -(x,y)

Subtraction of two `dualnum`s
"""
function -(x::dualnum, y::dualnum)
  r = x.val - y.val
  g = x.grad .- y.grad
  return dualnum(r, g)
end

"""
  -(x,y)

Subtraction of a `dualnum` and a regular number
"""
function -(x::dualnum, y::Number)
  return dualnum(x.val - y, x.grad)
end
-(x::Number, y::dualnum) = y-x

##################
# Multiplication #
##################

import Base.*
"""
  *(A,B)

Multiply two `dualnum`s
"""
function *(x::dualnum, y::dualnum)
  r = x.val * y.val
  g = y.val * x.grad + x.val * y.grad
  return dualnum(r, g)
end

"""
  *(A,B)

Multiply a `dualnum` with a number
"""
function *(x::dualnum, y::Number)
  r = x.val * y
  g = x.grad * y
  return dualnum(r, g)
end
*(y::Number, x::dualnum) = x*y

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

########
# Trig #
########

"""
  sin(x)

Sine-function of a `dualnum`
"""
function sin(x::dualnum)
  r = sin(x.val)
  g = x.grad * cos(x.val)
  return dualnum(r,g)
end

"""
  cos(x)

Cosine-function of a `dualnum`
"""
function cos(x::dualnum)
  r = cos(x.val)
  g = -x.grad * sin(x.val)
  return dualnum(r,g)
end

#########
# Other #
#########

"""
  ^(x,y)

Exponent of two `dualnum`s
"""
function ^(x::dualnum, y::Number)
  r = x.val ^ y
  g = y*x.val^(y-1) * x.grad
  return dualnum(r, g)
end

"""
  log(x)

Logarithm of a `dualnum` (base: e)
"""
function log(x::dualnum)
  r = log(x.val)
  g = 1/x.val * x.grad
  return dualnum(r, g)
end

"""
  log(b,x)

Logarithm of a `dualnum` (base: `b`)
"""
function log(b::Number, x::dualnum)
  r = log(x.val)
  g = 1/(x.val*log(b)) * x.grad
  return dualnum(r, g)
end

"""
  conj(x)

Conjugation of a `dualnum`
"""
function conj(x::dualnum)
  r = conj(x.val)
  g = conj(x.grad)
  return dualnum(r, g)
end

"""
  sqrt(x)

Square-root of a `dualnum`
"""
sqrt(x::dualnum) = x^0.5

"""
  conj!(x)

Conjugation (in-place) of a `dualnum`
"""
function conj!(x::dualnum{W}) where W <: Number
  if W <: Complex
    out = conj(x)
  else
    out = x
  end
  return out
end

import Base.abs2
"""
  abs2(x)

Absolute value squared of a `dualnum`
"""
function abs2(x::dualnum)
  return x*conj(x)
end

"""
  abs(x)

Absolute value of a `dualnum`
"""
abs(x::dualnum) = sqrt(abs2(x))

"""
  makeId(m)

Make identity of `dualnum`s of size `m`
"""
function makeId(m::Integer)
  return Diagonal(dualnum{W}[dualnum(W) for w = 1:m])
end

function joindiag(nvars::Integer, args...)
  matr = args[1]
  for i in 1:length(args)-1
    mrows = size(matr, 1)
    mcols = size(matr, 2)
    zrows = size(args[i+1], 1)
    zcols = size(args[i+1], 2)
    right_matr = dualnum_zeros(mrows, zcols, nvars)
    lower_left_matr = dualnum_zeros(zrows, mcols, nvars)

    upper_matr = hcat(matr, right_matr)
    lower_matr = hcat(lower_left_matr, args[i+1])
    matr = vcat(upper_matr, lower_matr)
  end
  return matr
end

"""
  norm(A)

L2 norm of a vector `A` containing `dualnum`s
"""
function norm(A::AbstractArray{W}) where W <: dualnum
  total = 0
  for x in A
    total += abs2(x.val)
  end
  return Base.sqrt(total)
end

##########
##########
##########

mutable struct dtens#={W,R}=# <: denstens #where {W <: TensType, R <: TensType}
  d::Tuple#{W,R} #zero is one!
end

function dtens(A::W,B::R) where {W <: TensType, R <: TensType}
  return dtens((A,B))
end

function dtens(A::W) where W <: TensType
  return dtens((A,ones(A)))
end
export dtens


function getindex!(A::dtens,w::intType)
  return A.d[w+1]
end

import Base.setindex!
function setindex!(A::dtens,B::dtens,a::genColType...)
  for w = 1:length(A.d)
    setindex!(B[w-1],A[w-1],a...)
  end
  nothing
end

function setindex!(A::dtens,B::TensType,a::Integer)
#  A.d[a+1] = B
if a == 0
  A.d = (B,A[1])
elseif a == 1
  A.d = (A[0],B)
end
#  setindex!(A.d,B,a+1)
#  setindex!(A[a+1],B,)
  nothing
end

import Base.size
function size(A::dtens,i)
  return size(A[0],i)
end

import Base.*
function *(A::dtens,B::dtens)
  C = contract(A[0],ndims(A[0]),B[0],1)
  dC = contract(A[0],ndims(A[0]),B[1],1)
  dC = contract(A[1],ndims(A[1]),B[0],1,dC)
  return dtens(C,dC)
end

#import .TensorPACK.contract,.TensorPACK.contractc,.TensorPACK.ccontract,.TensorPACK.ccontractc
#import .TensorPACK.svd
#import .TensorPACK.eigen
#import .TensorPACK.trace

function ndims(A::dtens)
  return ndims(A[0])
end

function contract(A::dtens,iA::intvecType,B::dtens,Z::dtens...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))

  if length(Z) > 0
    C = contract(A[0],iA,B[0],iB,Z[0],alpha=alpha,beta=beta)
    dC = contract(A[1],iA,B[1],iB,Z[1],alpha=alpha,beta=beta)
    out = dtens(C,contract(A[1],iA,B[0],iB,dC))
  else
    C = contract(A[0],iA,B[0],iB)
    dC = contract(A[1],iA,B[1],iB)
    out = dtens(C,contract(A[1],iA,B[0],iB,dC))
  end
  return out
end

function ccontract(A::dtens,iA::intvecType,B::dtens,Z::dtens...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))

  if length(Z) > 0
    C = ccontract(A[0],iA,B[0],iB,Z[0],alpha=alpha,beta=beta)
    dC = ccontract(A[1],iA,B[1],iB,Z[1],alpha=alpha,beta=beta)
    out = dtens(C,ccontract(A[1],iA,B[0],iB,dC))
  else
    C = ccontract(A[0],iA,B[0],iB)
    dC = ccontract(A[1],iA,B[1],iB)
    out = dtens(C,ccontract(A[1],iA,B[0],iB,dC))
  end
  return out
end

function contractc(A::dtens,iA::intvecType,B::dtens,Z::dtens...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))

  if length(Z) > 0
    C = contractc(A[0],iA,B[0],iB,Z[0],alpha=alpha,beta=beta)
    dC = contractc(A[1],iA,B[1],iB,Z[1],alpha=alpha,beta=beta)
    out = dtens(C,contractc(A[1],iA,B[0],iB,dC))
  else
    C = contractc(A[0],iA,B[0],iB)
    dC = contractc(A[1],iA,B[1],iB)
    out = dtens(C,contractc(A[1],iA,B[0],iB,dC))
  end
  return out
end

function ccontractc(A::dtens,iA::intvecType,B::dtens,Z::dtens...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))

  if length(Z) > 0
    C = ccontractc(A[0],iA,B[0],iB,Z[0],alpha=alpha,beta=beta)
    dC = ccontractc(A[1],iA,B[1],iB,Z[1],alpha=alpha,beta=beta)
    out = dtens(C,ccontractc(A[1],iA,B[0],iB,dC))
  else
    C = ccontractc(A[0],iA,B[0],iB)
    dC = ccontractc(A[1],iA,B[1],iB)
    out = dtens(C,ccontractc(A[1],iA,B[0],iB,dC))
  end
  return out
end

function dot(A::dtens,B::dtens)
  return dot(A[0],B[0]),dot(A[0],B[1])+dot(A[1],B[0])
end

#=
function trace!(A::dtens,iA::Array{NTuple{2,P},1}) where P <: Integer

end
=#

function hadamardprod(F::tens{W},S::tens{R}) where {W <: Number, R <: Number}
  return tens(S.size,[F[w]*S[w] for w = 1:length(F)])
end

function hadamardprod(F::AbstractArray{W},S::tens{R}) where {W <: Number, R <: Number}
  return tens(S.size,[F[w]*S[w] for w = 1:length(F)])
end

function hadamardprod(F::tens{W},S::AbstractArray{R}) where {W <: Number, R <: Number}
  return tens([size(S,w) for w = 1:ndims(S)],[F[w]*S[w] for w = 1:length(F)])
end

function hadamardprod(F::Array{W,N},S::Array{R,N}) where {N, W <: Number, R <: Number}
  return F .* S
end


function hadamardprod(F::Qtens{W,Q},S::Qtens{R,Q}) where {W <: Number, R <: Number, Q <: Qnum}
  return error("HELP!")
end


function makeF(s::Diagonal)
  F = Float64[i == j ? 0 : 1/(s[j]^2-s[i]^2) for i = 1:length(s), j = 1:length(s)]
  return F
end

function svd(A::dtens;power::Number=2,cutoff::Float64 = 0.,
  m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=true,
  effZero::Number=defzero,keepdeg::Bool=false,decomposer::Function=libsvd,
  a::Integer = size(A,1),b::Integer=size(A,2),inplace::Bool=false)

  U,D,V,truncerr,sumD = svd(A[0],power=power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,effZero=effZero,keepdeg=keepdeg,decomposer=decomposer,a=a,b=b,inplace=inplace)

  Uconvec = (1,) #ntuple(w->w,ndims(U)-1)
  Vconvec = (2,) #ntuple(w->w+1,ndims(V)-1)
  Aconvec = (2,) #ntuple(w->w+ndims(A)-length(Vconvec),length(Vconvec))

F = makeF(D)
invD = invmat(D)


rA = reshape(A[1],a,b)



  #dU
  Cterm = contract(rA,Aconvec,V,Vconvec)
  Lterm = ccontract(U,Uconvec,Cterm,Uconvec)

  Rterm = conj(Lterm)
  Rterm = permutedims!(Rterm,[2,1])

  G = hadamardprod(Lterm*D + D*Rterm,F)

  dU = contract(U,ndims(U),G,1)

  dAVS = contractc(rA,Aconvec,invD*V,Vconvec)
  UdAVS = Lterm * invD #contract(U,Uconvec,dAVS,Uconvec)
  UUdAVS = ccontract(U,ndims(U),UdAVS,1,dAVS,alpha=-1)
  dU += UUdAVS

  #dV
  G = hadamardprod(D*Lterm + Rterm*D,F)

  dV = contract(G,ndims(G),V,1,alpha=-1)

  SUdA = ccontract(U*invD,Uconvec,rA,Uconvec)
#  otherVconvec = ntuple(w->w,length(Vconvec))
  SUdAV = contractc(SUdA,Vconvec,V,Vconvec)
  SUdAVV = contract(SUdAV,ndims(SUdAV),V,1,SUdA,alpha=-1)
  dV += SUdAVV#contract(SUdAV,ndims(SUdAV),V,1) #SUdA

dD = Diagonal([Lterm[w,w] for w = 1:size(Lterm,1)])

  return dtens(U,dU),dtens(D,dD),dtens(V,dV),truncerr,sumD
  #can reassemble dA = dU*dD*dVt...in principle
end
#=
function eigen(A::dtens;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,a::Integer=size(A,1),b::Integer=size(A,2),minm::Integer=2,nozeros::Bool=false,power::Number=1,effZero::Real=defzero,keepdeg::Bool=false,transpose::Bool=false,decomposer::Function=libeigen,rev::Bool=true)
  D,U,truncerr,sumD = eigen(A[0],cutoff=cutoff,m=m,mag=mag,a=a,b=b,minm=minm,nozeros=nozeros,power=power,effZero=effZero,keepdeg=keepdeg,transpose=transpose,decomposer=decomposer,rev=rev)


  if transpose

    Uconvec = ntuple(w->w+1,ndims(U)-1)  
    Aconvec = ntuple(w->w+ndims(A)-length(Vconvec),length(Vconvec))

    dD = contractc(A[1],Aconvec,U,Uconvec)
    dD = contract(U,Uconvec,dD,Uconvec)

    invD = inv(D)
    dU = invD*ccontract(U,Uconvec,A[1],Uconvec)

  else
    Uconvec = ntuple(w->w,ndims(U)-1)  
    Aconvec = ntuple(w->w+ndims(A)-length(Vconvec),length(Vconvec))

    dD = contract(A[1],Aconvec,U,Uconvec)
    dD = ccontract(U,Uconvec,dD,Uconvec)

    invD = inv(D)
    dU = contract(A[1],Aconvec,U,Uconvec)*invD
  end
  return dtens(D,dD),dtens(U,dU),truncerr,sumD
end
=#

function trace!(A::dtens)
  return sum(w->searchindex(A[0],w,w),1:size(A[0],1)),sum(w->searchindex(A[1],w,w),1:size(A[1],1))
end


function makeArray(A::dtens)
  return makeArray(A[0]),makeArray(A[1])
end

function reshape!(A::dtens,a::Integer...)
  A[0] = reshape!(A[0],a...)
  A[1] = reshape!(A[1],a...)
  return A
end

function reshape!(A::dtens,order)
  A[0] = reshape!(A[0],order)
  A[1] = reshape!(A[1],order)
  return A
end

function reshape(A::dtens,order)
  A[0] = reshape!(A[0],order)
  A[1] = reshape!(copy(A[1]),order)
  return A
end
