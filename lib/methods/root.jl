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
    a = root(fct,x0[,niter=1000,tol=1E-9,m=1,dstart=10_000])

Newton's root finding method for an input function `fct` (evaluated on a position input only) and starting point `x0`; note that the function must allow for automatic differentiation allowed in the library to be used; outputs `a` the numerical estimate for the root

#Inputs:
+ `fct`: Function to evaluate over
+ `x0`: Starting point

#Optional inputs:
+ `niter`: maximum number of iterations
+ `tol`: tolerance to converge to
+ `m`: update is `x0 -= m * df/f` which helps convergence
+ `dstart`: starting derivative
"""
function root(fct::Function,x0::Number;niter::Integer=1000,tol::Number=1E-9,m::Number=1,dstart::Number=10_000)
  w = 0
  xdiff = TENPACK.dualnum(dstart)
  while w < niter && xdiff[0] > tol
    x = TENPACK.dualnum(x0)
    f = fct(x)
    x0 -= m * f[0]/f[1]
    xdiff = abs(x-x0)
  end
  return x0
end
