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
    C = dtens(A,B)

Creates a tensor `C` containing both values (`A`) and derivatives (`B`) useful for automatic differenation
"""
function dtens(A::Union{TensType,diagonal},B::Union{TensType,diagonal})
  return dtens((A,B))
end

"""
    C = dtens(A)

Creates a tensor `C` containing both values (`A`) and creates a second tensor of derivatives useful for automatic differenation
"""
function dtens(A::Union{TensType,diagonal})
  return dtens((A,ones(A)))
end
