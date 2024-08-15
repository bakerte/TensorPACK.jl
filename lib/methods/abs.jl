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
    y = abs2(x)

Absolute value squared of a `dualnum`
"""
function abs2(x::dualnum)
  return x*conj(x)
end

"""
    y = abs(x)

Absolute value of a `dualnum`
"""
abs(x::dualnum) = sqrt(abs2(x))

"""
    y = abs(x,a)

Equivalent to `abs(x)`
"""
function abs(x::Number,a::Number)
  return abs(x)
end