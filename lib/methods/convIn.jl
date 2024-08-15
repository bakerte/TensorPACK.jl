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
    G = convIn(iA)

Convert `iA` of type Int64 (ex: 1), Array{Int64,1} ([1,2,3]), or Array{Int64,2}* ([1 2 3]) to Array{Int64,1} (`G`)

*- two-dimensional arrays must be size "m x 1"
"""
function convIn(iA::Union{Array{P,1},Array{P,2}}) where P <: Integer
  return ntuple(i->iA[i],length(iA))
end

"""
    B = convIn(A)

Converts an integer `A` into a tuple
"""
function convIn(iA::Integer)
  return (iA,)
end

"""
    B = convIn(A)

Converts input tuple to a tuple (identity operation)
"""
function convIn(iA::NTuple{N,intType}) where N
  return iA
end