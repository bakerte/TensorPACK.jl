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

########
# Trig #
########

import Base.sin
"""
    sin(x)

Sine-function of a `dualnum`
"""
function sin(x::dualnum)
  r = sin(x.val)
  g = x.grad * cos(x.val)
  return dualnum(r,g)
end

import Base.cos
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
