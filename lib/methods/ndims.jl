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
    B = ndims(A)

Number of dimensions (rank) `B` of a `denstens` (identical usage to `Array` `ndims` call)

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function ndims(A::denstens)
  return length(A.size)
end

"""
    B = ndims(A)

Returns the number of indices of named tensor `A`

See also: [`TNobj`](@ref)
"""
function ndims(A::TNobj)
  return length(A.names)
end

"""
    B = ndims(A)

Returns the number of indices of named tensor `A`
"""
function ndims(A::dtens)
  return ndims(A[0])
end

"""
    B = ndims(A)

number of dimensions of a Qtensor (identical usage to dense `size` call)
"""
function ndims(A::qarray)
  return length(A.size)
end

"""
    B = basedims(A)

Finds the number of dimensions of the `qarray` `A` no matter how it was reshaped
"""
function basedims(Qtensor::qarray)
  return length(Qtensor.QnumMat)
end

"""
    B = basedims(A)

Finds the number of dimensions of the `denstens` `A`
"""
function basedims(A::denstens)
  return ndims(A)
end

"""
    B = ndims(A)

Number of dimensions (rank) `B` of a `diagonal` (identical usage to `Array` `ndims` call); returns a value of 2 for `diagonal`

See also: [`diagonal`](@ref) [`Array`](@ref)
"""
function ndims(A::diagonal)
  return 2
end
