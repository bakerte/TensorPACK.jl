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
    submatrix

Abstract type used for StrassOPen to compute multi-level Strassen algorithms
"""
abstract type submatrix end

#stores position in large matrix
"""
    submat

Stores information for Strassen matrix multiplication

  For example, the two level Strassen would have a matrix like [i for i = 1:16] reshaped into a 4x4 matrix. This matrix has indices `i` in the field of the `submat`

# Fields:
+ `i`: vector or tuple storing the position in the matrix
+ `sign`: a vector or a tuple of integer values (+/-1) signifying whether there is a negative or a plus sign in front of the submatrix
"""
struct submat <: submatrix# where W <: Union{Integer,submatrix}
  i::Union{Vector,Tuple} #Union{Vector{NTuple{P,W}},NTuple{P,W},W} where {P, R <: Tuple}
  sign::Union{Vector,Tuple} #Union{Vector{NTuple{G,Bool}},NTuple{G,Bool},Bool} where {G, S <: Tuple}
end

"""
    A = submat(i)

Generates a submat at position `i` and default sign of 1
"""
function submat(i::Integer)
  return submat([i],[1])
end

"""
    A = negate(x)

negates input `submat` by changing the sign of the integer (sign field in struct)

See also: [`submat`](@ref)
"""
function negate(x)
  return -1*x
end

"""
    -(A)

negates a `submat` type and creates a new one
"""
function -(A::submatrix)
    newtup = negate(A.sign)
  return submat(A.i,newtup)
end

"""
    -(A,B)

subtracts two `submat`s
"""
function -(A::submatrix,B::submatrix)
  return A + (-B)
end

"""
    +(A,B)

adds two `submat`s
"""
function +(A::submatrix,B::submatrix)

  if eltype(A.i) <: Integer && typeof(A.i) <: Vector && eltype(B.i) <: Integer && typeof(B.i) <: Vector
    newi = vcat(A.i,B.i)
    newsign = vcat(A.sign,B.sign)
  else
    newi = [A.i,B.i]
    newsign = [A.sign,B.sign]
  end
  return submat(newi,newsign)
end


"""
    *(A,B)

multiplies (concatenates) two `submat`s
"""
function *(A::submatrix,B::submatrix)
  return submat((A.i,B.i),(A.sign,B.sign))
end

"""
    *(A,B)

multiplies (concatenates) two matrices of `submat`s
"""
function *(A::Matrix{W},B::Matrix{W}) where W <: submatrix
  return submat((A[1].i,B[1].i),[A[1].sign,B[1].sign])
end

"""
    A = length(submat)

length of a given submat (will accumulate terms with +,-,*,/ operations)
"""
function length(submat::submatrix)
  return length(submat.i)
end

"""
    A = eltype(submat)

element type of a given `submat`
"""
function eltype(submat::submatrix)
  return eltype(submat.i)
end

"""
    A,B = makematrix(n)

Creates two matrices `A` and `B` suitable for Strassen determination at a level `n`
"""
function makematrix(nLevel::Integer)
  M = [i for i = 1:4^nLevel]

  m = 2^nLevel
  M = reshape(M,m,m)'

  A = submat.(M)
  B = submat.(M)

  return A,B
end

"""
   ... = strassen_recursive(A,B)

Recursive function call to generate strings of submatrices that will evaluate an n-level Strassen algorithm (n specified in `strassOPen` function or loop)
"""
function strassen_recursive(A::Matrix{W},B::Matrix{W}) where W <: submat

  half = fld(size(A,1),2)
  A1 = A[1:half,1:half] #top left
  A2 = A[half+1:end,1:half] #bottom left
  A3 = A[1:half,half+1:end] #top right
  A4 = A[half+1:end,half+1:end] #bottom right

  B1 = B[1:half,1:half]
  B2 = B[half+1:end,1:half]
  B3 = B[1:half,half+1:end]
  B4 = B[half+1:end,half+1:end]

  if half == 1

    A_i = [(A1+A4)[1].i, (A2+A4)[1].i, A1[1].i, A4[1].i, (A1+A3)[1].i, (A2-A1)[1].i, (A3-A4)[1].i]
    A_sign = [(A1+A4)[1].sign, (A2+A4)[1].sign, A1[1].sign, A4[1].sign, (A1+A3)[1].sign, (A2-A1)[1].sign, (A3-A4)[1].sign]
    B_i = [(B1+B4)[1].i, B1[1].i, (B3-B4)[1].i, (B2-B1)[1].i, B4[1].i, (B1+B3)[1].i, (B2+B4)[1].i]
    B_sign = [(B1+B4)[1].sign, B1[1].sign, (B3-B4)[1].sign, (B2-B1)[1].sign, B4[1].sign, (B1+B3)[1].sign, (B2+B4)[1].sign]

    return [A_i, A_sign], [B_i, B_sign]

  else

    X1,Y1 = strassen_recursive(A1+A4,B1+B4)
    X2,Y2 = strassen_recursive(A2+A4,B1)
    X3,Y3 = strassen_recursive(A1,B3-B4)
    X4,Y4 = strassen_recursive(A4,B2-B1)
    X5,Y5 = strassen_recursive(A1+A3,B4)
    X6,Y6 = strassen_recursive(A2-A1,B1+B3)
    X7,Y7 = strassen_recursive(A3-A4,B2+B4)

    A_i = vcat(X1[1],X2[1],X3[1],X4[1],X5[1],X6[1],X7[1])
    A_sign = vcat(X1[2],X2[2],X3[2],X4[2],X5[2],X6[2],X7[2])
    B_i = vcat(Y1[1],Y2[1],Y3[1],Y4[1],Y5[1],Y6[1],Y7[1])
    B_sign = vcat(Y1[2],Y2[2],Y3[2],Y4[2],Y5[2],Y6[2],Y7[2])

    return [A_i, A_sign], [B_i, B_sign]

  end
end

"""
    strassOPen(n)

Generates rules for any level-`n` Strassen algorithm
"""
function strassOPen(n::Integer)
  A,B = makematrix(n)
  return strassen_recursive(A,B)
end
export strassOPen

"""
    strassen_max (=2)

This indicates the default level of the Strassen algorithm in the library
"""
const strassen_max = 3 #level of the strassen algorithm

"""
    strassen_rules
    
These are the defaults used for the library's Strassen algorithm generated from `strassOPen` for the Strassen matrix multiplication algorithm
"""
const strassen_rules = [strassOPen(w) for w = 1:strassen_max]
export strassen_rules