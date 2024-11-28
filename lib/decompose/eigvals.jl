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
    D = eigvals(A)

Eigenvalues of input `A` (allowing julia's arrays and `denstens` types) output to a vector `D`

See also: [`eigvals!`](@ref)
"""
function eigvals(A::Union{Array{W,2},tens{W}}) where W <: Number
  D,U = libeigen(A,job='N')
  return D
end

"""
    D = eigvals(alpha,beta)

Eigenvalues of a symmetric tridiagonal matrix with diagonal elements `alpha` and off-diagonal elements `beta` using first `n` elements of those vectors

See also: [`eigvals!`](@ref)
"""
function eigvals(alpha::Array{W,1},beta::Array{W,1},n::intType) where W <: Number
  D,U = libeigen(alpha,beta,n,job='N')
  return D
end

"""
    D = eigvals(alpha,beta)

Eigenvalues of a symmetric tridiagonal matrix with diagonal elements `alpha` and off-diagonal elements `beta`

See also: [`eigvals!`](@ref)
"""
function eigvals(alpha::Array{W,1},beta::Array{W,1}) where W <: Number
  D,U = libeigen(alpha,beta,job='N')
  return D
end

"""
    D = eigvals!(A)

Eigenvalues of input `A` (allowing julia's arrays and `denstens` types) output to a vector `D`

See also: [`eigvals`](@ref)
"""
function eigvals!(A::Union{Array{W,2},tens{W}}) where W <: Number
  D,U = libeigen!(A,job='N')
  return D
end

"""
    D = eigvals!(alpha,beta,n)

Eigenvalues of a symmetric tridiagonal matrix with diagonal elements `alpha` and off-diagonal elements `beta` using first `n` elements of those vectors

See also: [`eigvals`](@ref)
"""
function eigvals!(alpha::Array{W,1},beta::Array{W,1},n::intType) where W <: Number
  D,U = libeigen(alpha,beta,n,job='N')
  return D
end

"""
    D = eigvals!(alpha,beta)

Eigenvalues of a symmetric tridiagonal matrix with diagonal elements `alpha` and off-diagonal elements `beta`

See also: [`eigvals`](@ref)
"""
function eigvals!(alpha::Array{W,1},beta::Array{W,1}) where W <: Number
  D,U = libeigen(alpha,beta,job='N')
  return D
end
