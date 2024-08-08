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
#=
"""
  dualnum(val,grad)

Initializes the `dualnum` type with value `val` and a gradient `grad`
"""
function dualnum(val::W, grad::W) where W <: Number
  return dualnum{W}(val, grad)
end
=#
"""
  dualnum(W)

Creates a `dualnum` of type `W`
"""
function dualnum(W::DataType)
  return dualnum{W}(W(0),W(1))
end



function dualnum_zeros(indeces::Integer, nvars::Integer)
  z = dualnum[]
  for i in 1:indeces
    append!(z, [init_dualnum(0, nvars)])
  end
  return z
end

function dualnum_zeros(rows::Integer, cols::Integer, nvars::Integer)
  zarray = dualnum_zeros(rows, nvars)
  zmatrix = zarray
  for c in 2:cols
    zmatrix = hcat(zmatrix, zarray)
  end
  return zmatrix
end

function dualnum_identity(len::Integer, nvars::Integer)
  matr = dualnum_zeros(len, len, nvars)
  for i in 1:len
    matr[i,i] = init_dualnum(1, nvars)
  end
  return matr
end




function init_dualnum(real, num_variables)
  return dualnum(real, zeros(num_variables))
end

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

