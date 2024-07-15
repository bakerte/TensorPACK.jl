"""
    root(fct,x0)

Newton's root finding method for an input function `fct` (evaluated on a position input only) and starting point `x0`; note that the function must allow for automatic differentiation allowed in the library to be used
"""
function root(fct::Function,x0::Number;niter::Integer=100,tol::Number=1E-9,m::Number=1)
  w = 0
  xdiff = TENPACK.dualnum(10000)
  while w < niter && xdiff[0] > tol
    x = TENPACK.dualnum(x0)
    f = fct(x)
    x0 -= m * f[0]/f[1]
    xdiff = abs(x-x0)
  end
  return x0
end