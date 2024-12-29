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
    G = Array(M)

Convert `denstens` `M` to julia `Array` (`G`)

See: [`denstens`](@ref) [`Array`](@ref)
"""
function Array(M::tens{W}) where W <: Number
  return reshape!(convert(Array{W,1},M.T), tupsize(M))
end

"""
    G = Array(M)

Convert `diagonal` `M` to julia `Array` (`G`)

See: [`diagonal`](@ref) [`Array`](@ref)
"""
function Array(M::Diagonal{W}) where W <: Number
  return LinearAlgebra.Diagonal(M.T)
end


"""
    G = Array(M)

Convert `dtens` `M` to julia `Array` (`G`)

See: [`Diagonal`](@ref) [`Array`](@ref) [`tens`](@ref)
"""
function Array(A::dtens)
  return Array(A[0]),Array(A[1])
end

"""
    Array(Qt)

Converts `Qt` of type `Qtens` to a default julia-type array
"""
function Array(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return Array(tens(QtensA))
end

