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
    -(A,B)

Subtracts two `TNobj`s `A` and `B`

See also: [`sub!`](@ref)
"""
function -(A::TNobj,B::TNobj)
  return sub!(copy(A),B)
end

"""
    -(A,B)

Subtracts two `dtens` `A` and `B`

See also: [`sub!`](@ref)
"""
function -(x::dtens, y::dtens)
  r = x[0] - y[0]
  g = sqrt((x[1])^2 + (y[1])^2)
  return dtens(r, g)
end

"""
    C = -(A,B)

Subtracts a matrix `A` to a `diagonal` `B`
"""
function -(A::Matrix{W},B::diagonal) where W <: Number
  C = copy(A)
  for w = 1:size(C,1)
    C[w,w] -= B[w]
  end
  return C
end

"""
    C = +(A,B)

Subtracts a `diagonal` `A` to a matrix `B`
"""
function -(B::diagonal,A::Matrix{W}) where W <: Number
  C = copy(A)
  for w = 1:size(C,1)
    C[w,w] = -C[w,w] + B[w]
  end
  return C
end


"""
    C = -(A,B)

Subtracts a `diagonal` `A` to a `diagonal` `B`
"""
function -(B::diagonal,A::diagonal)
  #=
  C = copy(A)
  for w = 1:size(C,1)
    C[w,w] += B[w]
  end
  =#
  return Diagonal(B.T - A.T)
end


###############
# Subtraction #
###############

"""
    -(x,y)

Subtraction of two `dualnum`s
"""
function -(x::dualnum, y::dualnum)
  r = x.val - y.val
  g = sqrt(x.grad^2 + y.grad^2)
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

"""
    -(M)

Negation of a `qarray`
"""
function -(M::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return tensorcombination(M,alpha=(W(-1),))
end

"""
    -(x)

Negation of a `dualnum`
"""
function -(x::dualnum)
  r = -x.val
  g = -x.grad
  return dualnum(r, g)
end

"""
    G = -(A,B)

Subtracts two tensors `A` and `B` (`A`-`B`) with output `G`

See also: [`sub!`](@ref)
"""
function -(A::TensType, B::TensType)
  mA,mB = checkType(A,B)
  return tensorcombination((eltype(A)(1),eltype(B)(-1)),mA,mB)
end

"""
    C = -(A,B)

Subtracts a matrix `A` full of `dualnum`s from a matrix full of numbers `B` with output `G`

See also: [`sub!`](@ref)
"""
function -(A::Matrix{dualnum}, B::Matrix{Number})
  derivC = Array{dualnum,2}(undef,shape[1],shape[2])
  for i in 1:shape[1]
    for j in 1:shape[2]
      derivC[i, j] = A[i, j] - B[i, j]
    end
  end
  return derivC
end

"""
    C = -(A,B)

Subtracts a matrix `A` full of numbers from a matrix full of `dualnum`s `B` with output `G`

See also: [`sub!`](@ref)
"""
function -(A::Matrix{Number}, B::Matrix{dualnum})
  shape = size(A)

  derivC = Array{dualnum,2}(undef,shape[1],shape[2])
  for i in 1:shape[1]
    for j in 1:shape[2]
      derivC[i, j] = A[i, j] - B[i, j]
    end
  end
  return derivC
end

"""
    C = -(A,B)

Subtracts a vector `A` full of `dualnum`s from a vector full of numbers `B` with output `G`

See also: [`sub!`](@ref)
"""
function -(A::Vector{dualnum}, B::Vector{<:Number})
  shape = size(A)

  derivC = Array{dualnum,1}(undef,shape[1])
  for i in 1:shape[1]
    derivC[i] = A[i] - B[i]
  end
  return derivC
end

"""
    C = -(A,B)

Subtracts a vector `A` full of numbers from a vector full of `dualnum`s `B` with output `G`

See also: [`sub!`](@ref)
"""
function -(A::Vector{Number}, B::Vector{dualnum})
  shape = size(A)

  derivC = Array{dualnum,1}(undef,shape[1])
  for i in 1:shape[1]
    derivC[i] = A[i] - B[i]
  end
  return derivC
end

