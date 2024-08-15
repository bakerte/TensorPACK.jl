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
    G = det(A)

Finds the determinant of a `tens`
"""
function det(A::tens)
  return LinearAlgebra.det(Array(A))
end

"""
    G = Array(A)

Finds the determinant of an `Array`
"""
function det(A::Array)
  return LinearAlgebra.det(A)
end

"""
    G = logdet(A)

Finds the log-determinant of a `tens`
"""
function logdet(A::tens)
  return LinearAlgebra.logdet(Array(A))
end

"""
    G = logdet(A)

Finds the log-determinant of an `Array`
"""
function logdet(A::Array)
  return LinearAlgebra.logdet(A)
end