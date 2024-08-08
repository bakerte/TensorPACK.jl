

"""
  abs2(x)

Absolute value squared of a `dualnum`
"""
function abs2(x::dualnum)
  return x*conj(x)
end

"""
  abs(x)

Absolute value of a `dualnum`
"""
abs(x::dualnum) = sqrt(abs2(x))