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
  log(x)

Logarithm of a `dualnum` (base: e)
"""
function log(x::dualnum)
  r = log(x.val)
  g = 1/x.val * x.grad
  return dualnum(r, g)
end

"""
  log(b,x)

Logarithm of a `dualnum` (base: `b`)
"""
function log(b::Number, x::dualnum)
  r = log(x.val)
  g = 1/(x.val*log(b)) * x.grad
  return dualnum(r, g)
end
