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
    B = *(c,A)

Multiplies value `c` onto a `dtens`
"""
function *(val::Number,A::dtens)
  B = copy(A)
  B[0] *= val
  return B
end


##################
# Multiplication #
##################

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

Multiply a `dualnum` `A` with a number `B`
"""
function *(x::dualnum, y::Number)
  r = x.val * y
  g = x.grad * y
  return dualnum(r, g)
end

"""
  *(A,B)

Multiply a `dualnum` `B` with a number `A`
"""
*(y::Number, x::dualnum) = x*y

"""
    C = *(A,B)

Multiplies a matrix of `dualnum` `A` onto a `dualnum` `B`
"""
function *(A::Matrix{dualnum},B::dualnum)
  C = Array{dualnum,2}(undef,size(A,1),size(A,2))
  for y = 1:size(A,2)
    for x = 1:size(A,1)
      C[x,y] = A[x,y]*B
    end
  end
  return C
end

"""
    C = *(A,B)

Multiplies a `dualnum` `A` onto a matrix of `dualnum` `B`
"""
function *(B::dualnum,A::Matrix{dualnum})
  return A*B
end

"""
    C = *(A,B)

Multiplies a matrix of regular numbers `A` onto a `dualnum` `B`
"""
function *(A::Matrix{<:Number},B::dualnum)
  C = Array{dualnum,2}(undef,size(A,1),size(A,2))
  for y = 1:size(A,2)
    for x = 1:size(A,1)
      C[x,y] = A[x,y]*B
    end
  end
  return C
end

"""
    C = *(A,B)

Multiplies a `dualnum` `A` onto a matrix of regular numbers `B`
"""
function *(B::dualnum,A::Matrix{<:Number})
  return A*B
end

"""
    C = *(A,B)

Multiplies a matrix of `dualnum` `A` onto a vector of `dualnum` or numbers `B`
"""
function *(A::Matrix{dualnum}, B::Vector{W}) where W <: Union{Number, dualnum}
  # For some reason this will only work with Float64 in the vector and not Number

  num_variables = length(A[1,1].grad)
  derivC = Array{dualnum,2}(undef,size(A,1),size(B,2))
  for x = 1:size(A,1)
    derivC[x] = dualnum(0, num_variables)
    for z = 1:size(A,2)
      derivC[x] += A[x,z]*B[z]
    end
  end
  return derivC
end

#=
"""
    C = *(A,B)

Multiplies a matrix of `dualnum` `A` onto a matrix of `dualnum` `B`
"""
function *(A::Matrix{dualnum},B::Matrix{dualnum})
  num_variables = length(A[1,1].grad)
  derivC = Array{dualnum,2}(undef,size(A,1),size(B,2))
  for y = 1:size(B,2)
    for x = 1:size(A,1)
      derivC[x,y] = dualnum(0, num_variables)
      for z = 1:size(A,2)
        derivC[x,y] += A[x,z]*B[z,y]
      end
    end
  end
  return derivC
end
=#

"""
    *(A,num)

Gives the multiplication of named tensor `A` by number `num` (commutative)

See also: [`mult!`](@ref)
"""
function *(A::TNobj,num::Number)
  return mult!(copy(A),num)
end

"""
    *(num,A)

Gives the multiplication of named tensor `A` by number `num` (commutative)

See also: [`mult!`](@ref)
"""
function *(num::Number,A::TNobj)
  return A*num
end

"""
    *(a,b)

concatenates string `a` with integer `b` after converting integer to a string
"""
function *(a::String,b::Integer)
  return a*string(b)
end

"""
  G = *(A,num)

Mutiplies a `TensType` `A` by a number `num` with output `G`

See also: [`mult!`](@ref)
"""
function *(num::T, M::TensType) where T <: Number
  if !(T <: eltype(M))
    if T <: Complex && !(eltype(M) <: Complex)
      aM = convertTens(ComplexF64,M)
      xnum = ComplexF64(num)
    else
      aM = M
      xnum = convert(eltype(M),num)
    end
  else
    aM = M
    xnum = num
  end
  return tensorcombination(aM,alpha=(xnum,))
end

"""
  G = *(num,A)

Mutiplies a `TensType` (or `Diagonal`) `A` by a number `num` with output `G`. Ensures commutativity of the operation

See also: [`mult!`](@ref) [`allTensType`](@ref)
"""
function *(M::Union{TensType,Diagonal}, num::Number)
  return num * M
end

"""
  G = *(num,A)

Mutiplies a `Diagonal` tensor `A` by a number `num` with output `G`

See also: [`mult!`](@ref)
"""
function *(num::T, M::Diagonal) where T <: Number
  return Diagonal(num * M.T)
end
