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
    A = undefMat(outtype,x,y)

Generates a matrix of size `(x,y)` with element type `outtype`.
"""
 function undefMat(outtype::DataType,x::Integer,y::Integer)
  return Array{outtype,2}(undef,x,y)
end
export undefMat