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
    G = Vector(M)

Convert `denstens` `M` to julia `Vector` (`G`)

See: [`denstens`](@ref) [`Vector`](@ref)
"""
function Vector(M::tens{W}) where W <: Number
  if ndims(M) != 1
    error("input tensor is not of rank 1 for Vector (use Array for general function instead)")
  end
  return Array(M)
end

"""
    G = Vector(M)

Convert `diagonal` `M` to julia `Vector` (`G`)

See: [`diagonal`](@ref) [`Vector`](@ref)
"""
function Vector(M::Diagonal{W}) where W <: Number
  if ndims(M) != 1
    error("input tensor is not of rank 1 for Vector (use Array for general function instead)")
  end
  return Array(M)
end


"""
    G = Vector(M)

Convert `dtens` `M` to julia `Vector` (`G`)

See: [`Diagonal`](@ref) [`Vector`](@ref) [`tens`](@ref)
"""
function Vector(A::dtens)
  if ndims(M) != 1
    error("input tensor is not of rank 1 for Vector (use Array for general function instead)")
  end
  return Array(A)
end

"""
    Vector(Qt)

Converts `Qt` of type `Qtens` to a default julia-type vector
"""
function Vector(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  if ndims(M) != 1
    error("input tensor is not of rank 1 for Vector (use Array for general function instead)")
  end
  return Array(QtensA)
end

