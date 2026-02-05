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
    G = Matrix(M)

Convert `denstens` `M` to julia `Matrix` (`G`)

See: [`denstens`](@ref) [`Matrix`](@ref)
"""
function Matrix(M::tens{W}) where W <: Number
  if ndims(M) != 2
    error("input tensor is not of rank 2 for Matrix (use Array for general function instead)")
  end
  return Array(M)
end

"""
    G = Matrix(M)

Convert `diagonal` `M` to julia `Matrix` (`G`)

See: [`diagonal`](@ref) [`Matrix`](@ref)
"""
function Matrix(M::Diagonal{W}) where W <: Number
  if ndims(M) != 2
    error("input tensor is not of rank 2 for Matrix (use Array for general function instead)")
  end
  return Array(M)
end


"""
    G = Matrix(M)

Convert `dtens` `M` to julia `Matrix` (`G`)

See: [`Diagonal`](@ref) [`Matrix`](@ref) [`tens`](@ref)
"""
function Matrix(A::dtens)
  if ndims(M) != 2
    error("input tensor is not of rank 2 for Matrix (use Array for general function instead)")
  end
  return Array(A)
end

"""
    Matrix(Qt)

Converts `Qt` of type `Qtens` to a default julia-type matrix
"""
function Matrix(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  if ndims(M) != 2
    error("input tensor is not of rank 2 for Matrix (use Array for general function instead)")
  end
  return Array(QtensA)
end

