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
    C = +(A,B)

Adds two tensors `A` and `B` together with output `C`

See also: [`add!`](@ref)
"""
function +(A::TensType, B::TensType)
  mA,mB = checkType(A,B)
  return tensorcombination(mA,mB)
end

"""
    +(A,B)

Adds tensors `A` and `B`

See also: [`add!`](@ref)
"""
function +(A::TNobj,B::TNobj)
  return add!(copy(A),B)
end

############
# Addition #
############

"""
  +(x,y)

Addition of two `dualnum`s
"""
function +(x::dualnum, y::dualnum)
  r = x.val + y.val
  g = x.grad + y.grad
  return dualnum(r, g)
end

"""
  +(x,y)

Addition of a `dualnum` and a regular number `y` (commutes)
"""
function +(x::dualnum, y::Number)
  return dualnum(x.val + y, x.grad)
end
+(x::Number, y::dualnum) = y+x
