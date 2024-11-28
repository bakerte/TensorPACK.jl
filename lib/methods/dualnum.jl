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

"""
  dualnum(val[,grad=1.])

Initializes the `dualnum` type with value `val` and a gradient `grad` (default value 1)
"""
function dualnum(val::W; grad::W=W(1)) where W <: Number
  return dualnum{W}(val, grad)
end


function dualnum{W}(val::Number; grad::Number=W(1)) where W <: Number
  return dualnum{W}(val, grad)
end
#=
"""
  dualnum(val,grad)

Initializes the `dualnum` type with value `val` and a gradient `grad`
"""
function dualnum(val::W, grad::W) where W <: Number
  return dualnum{W}(val, grad)
end
=#
#=
function dualnum{W}(X::Number) where W <: Number
  return dualnum{W}(X)
end
=#
"""
  dualnum(W)

Creates a `dualnum` of type `W`
"""
function dualnum(W::DataType)
  return dualnum{W}(W(0),W(1))
end

"""
  dualnum([,type=Float64])

Creates a zero `dualnum` of type Float64
"""
function dualnum(;type::DataType=Float64)
  return dualnum{type}(type(0),type(1))
end


"""
    dualnum_zeros(inds)

creates a `dualnum` of zeros
"""
function dualnum_zeros(inds::Integer)
  return tens([dualnum() for i = 1:inds])
end

"""
    dualnum_zeros(rows,cols)

creates a matrix of `dualnum`s (`rows` x `cols`) of zeros
"""
function dualnum_zeros(rows::Integer, cols::Integer)
  return [dualnum() for i = 1:rows, j = 1:cols]
end

"""
    dualnum_identity(rows,cols)

creates a matrix of `dualnum`s (`rows` x `cols`) of ones in an identity matrix
"""
function dualnum_identity(len::Integer)
  return diagonal(dualnum_ones(len))
end


#=
"""
    init_dualnum(val[,nvars])

"""
function init_dualnum(real; nvars=1)
  return dualnum(real, zeros(num_variables))
end
=#
#=
function init_dualnum(real, num_variables, variable_index)
  if variable_index > num_variables
    error("The given variable index must be less than or equal to the number of variables for a dualnum")
  elseif variable_index <= 0
    error("Any derivative index less than 1 is invalid for a dualnum")
  end
  g = zeros(num_variables)
  g[variable_index] = 1
  return dualnum(real, g)
end
=#
import Base.real
function real(X::dualnum)
  return dualnum{Float64}(real(X.val),real(X.grad))
end

import Base.isless
function isless(X::dualnum,Y::Number)
  return X.val < Y
end

function isless(X::Number,Y::dualnum)
  return Y.val < X
end