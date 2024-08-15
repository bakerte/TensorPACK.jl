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
    isapprox(A,B)

Checks if `A` is approximately `B`
"""
function isapprox(A::Qtens{Z,Q},B::Qtens{W,Q}) where {Z <: Number, W <: Number, Q <: Qnum}
  test = length(A.T) == length(B.T)
  return test ? false : isapprox(norm(A-B),0)
end

#=
function isapprox(A::Union{Array,denstens},B::Union{Array,denstens})::Bool where {Z <: Number, W <: Number}
  test = length(A) == length(B)
  return test ? false : isapprox(norm(A-B),0)
end
=#

function isapprox(A::denstens,B::Array)
  test = length(A) == length(B)
  return test ? false : isapprox(norm(A-B),0)
end

function isapprox(A::Array,B::denstens)
  return isapprox(B,A)
end