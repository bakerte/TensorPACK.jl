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


import Base.^
function ^(A::Diagonal{W},P::B) where {W <: Number, B <: Number}
  C = A.T .^ P
  return Diagonal(C)
end

"""
  loadM!(out,in)

Simple copy operation from `in` matrix to `out` matrix. Assumes same element type and useful for ensuring compiler efficiency.
"""
function loadM!(output::Array{W,N},input::Array{W,N}) where {N, W <: Number}
  @inbounds @simd for x = 1:length(input)
    output[x] = input[x]
  end
  nothing
end



"""
  G = mult!(A,x)

Multiplies `x*A` (commutative) for dense or quantum tensors with output `G`

See also: [`*`](@ref) [`add!`](@ref) [`sub!`](@ref) [`div!`](@ref)
"""
function mult!(M::W, num::Number) where W <: TensType
  cnum = typeof(num) == eltype(M) ? num : convert(eltype(M),num)
  return tensorcombination!(M,alpha=(cnum,))
end

function mult!(num::Number,M::W) where W <: TensType
  return mult!(M,num)
end
export mult!

"""
  G = add!(A,B,x)

Adds `A + x*B` for dense or quantum tensors with output `G`

See also: [`+`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function add!(A::W, B::W, num::Number) where W <: TensType
  cnum = typeof(num) == eltype(A) ? num : convert(eltype(A),num)
  return tensorcombination!((eltype(A)(1),cnum),A,B)
end

"""
  G = add!(A,B)

Adds `A + B` for dense or quantum tensors with output `G`

See also: [`+`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function add!(A::W, B::W) where W <: TensType
  return add!(A,B,eltype(B)(1))
end
export add!

"""
  G = sub!(A,B,x)

Subtracts `A - x*B` for dense or quantum tensors with output `G`

See also: [`-`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function sub!(A::W,B::W,mult::Number) where W <: TensType
  cnum = typeof(mult) == eltype(A) ? mult : convert(eltype(A),mult)
  return add!(A,B,-cnum)
end

"""
  G = sub!(A,B)

Subtracts `A - B` for dense or quantum tensors with output `G`

See also: [`-`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref) [`div!`](@ref)
"""
function sub!(A::W,B::W) where W <: TensType
  return add!(A,B,eltype(A)(-1))
end
export sub!

"""
  G = div!(A,x)

Division by a scalar `A/x` (default x = 1) for dense or quantum tensors with output `G`

See also: [`/`](@ref) [`add!`](@ref) [`sub!`](@ref) [`mult!`](@ref)
"""
function div!(M::TensType, num::Number)
  cnum = typeof(num) == eltype(M) ? num : convert(eltype(M),num)
  return tensorcombination!(M,alpha=(cnum,),fct=/)
end
export div!

function norm!(M::TensType)
  return div!(M,norm(M))
end
export norm!









import LinearAlgebra.+
"""
  G = +(A,B)

Adds two tensors `A` and `B` together with output `G`

See also: [`add!`](@ref)
"""
function +(A::TensType, B::TensType)
  mA,mB = checkType(A,B)
  return tensorcombination(mA,mB)
end

import LinearAlgebra.-
"""
  G = -(A,B)

Subtracts two tensors `A` and `B` (`A`-`B`) with output `G`

See also: [`sub!`](@ref)
"""
function -(A::TensType, B::TensType)
  mA,mB = checkType(A,B)
  return tensorcombination((eltype(A)(1),eltype(B)(-1)),mA,mB)
end

import Base.*
"""
  G = *(A,num)

Mutiplies a tensor `A` by a number `num` with output `G`

See also: [`mult!`](@ref)
"""
function *(num::Number, M::TensType)
  if !(typeof(num) <: eltype(M))
    num = convert(eltype(M),num)
  end
  return tensorcombination(M,alpha=(num,))
end

"""
  G = *(num,A)

Mutiplies a tensor `A` by a number `num` with output `G`. Ensures commutativity of the operation

See also: [`mult!`](@ref)
"""
function *(M::TensType, num::Number)
  return num * M
end

import LinearAlgebra./
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




"""
    *(a,b)

concatenates string `a` with integer `b` after converting integer to a string
"""
function *(a::String,b::Integer)
  return a*string(b)
end


"""
    div!(A,num)

Gives the division of named tensor `A` by number `num`

See also: [`/`](@ref)
"""
function div!(A::TNobj,num::Number)
  A.N = div!(A.N,num)
  return A
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
    mult!(A,num)

Gives the multiplication of named tensor `A` by number `num`

See also: [`*`](@ref)
"""
function mult!(A::TNobj,num::Number)
  A.N = mult!(A.N,num)
  return A
end

"""
    *(A,num)

Gives the multiplication of named tensor `A` by number `num` (commutative)

See also: [`mult!`](@ref)
"""
function *(A::TNobj,num::Number)
  return mult!(copy(A),num)
end

function *(num::Number,A::TNobj)
  return A*num
end




"""
    matchnames(AA,order,q)

Matches `order` (a length 2 vector of vectors of strings for indices) to the indices in `AA` for the left (right) with `q`=1 (2)
"""
function matchnames(AA::TNobj,order::Array{B,1}) where B <: String
  vect = Array{intType,1}(undef,length(order))
  for a = 1:length(order)
    condition = true
    w = 0
    while condition && w < length(AA.names)
      w += 1
      if order[a] == AA.names[w]
        vect[a] = w
        condition = false
      end
    end
  end
  return vect
end


function add!(A::TNobj,B::TNobj)
  reorder = matchnames(A,B.names)
  if !issorted(reorder)
    C = permutedims(B,reorder)
  else
    C = B
  end
  A.N = add!(A.N,C.N)
  return A
end

"""
    +(A,B)

Adds tensors `A` and `B`

See also: [`add!`](@ref)
"""
function +(A::TNobj,B::TNobj)
  return add!(copy(A),B)
end

"""
    sub!(A,B)

Subtracts tensor `A` from `B` (changes `A`)

See also: [`-`](@ref)
"""
function sub!(A::TNobj,B::TNobj)
  reorder = matchnames(A,B.names)
  if !issorted(reorder)
    C = permutedims(B,reorder)
  else
    C = B
  end
  A.N = sub!(A.N,C.N)
  return A
end

"""
    -(A,B)

Subtracts tensor `A` from `B`

See also: [`sub!`](@ref)
"""
function -(A::TNobj,B::TNobj)
  return sub!(copy(A),B)
end

function *(val::Number,A::dtens)
  B = copy(A)
  B[0] *= val
  return B
end






#Based on a tutorial by Prof. Alan Edelman at https://www.youtube.com/watch?v=vAp6nUMrKYg

#import Base: +, -, /, *, ^, log, conj, conj!, abs, abs2, convert, promote_rule, zeros, transpose, sin, cos
#import LinearAlgebra: norm
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


function +(x::dtens, y::dtens)
  r = x[0] + y[0]
  g = x[1] + y[1]
  return dtens(r, g)
end

function -(x::dtens, y::dtens)
  r = x[0] - y[0]
  g = x[1] - y[1]
  return dtens(r, g)
end


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
  g = x.grad + y.grad
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
  g = x.grad - y.grad
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
#=
"""
  makeId(m)

Make identity of `dualnum`s of size `m`
"""
function makeId(m::Integer)
  return Diagonal(dualnum{W}[dualnum(W) for w = 1:m])
end
=#
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

function -(M::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return tensorcombination(M,alpha=(W(-1),))
end
