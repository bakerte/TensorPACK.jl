

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
