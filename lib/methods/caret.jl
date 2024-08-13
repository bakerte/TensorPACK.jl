

function ^(A::Diagonal{W},P::B) where {W <: Number, B <: Number}
  C = A.T .^ P
  return Diagonal(C)
end

function ^(A::tens{W},P::B) where {W <: Number, B <: Number}
  if B <: Integer
    C = prod(w->A,1:P)
  else
    R = Array(A)
    C = tens(R^P)
  end
  return C
end

"""
  ^(x,y)

Exponent of two `dualnum`s
"""
function ^(x::dualnum, y::Number)
  r = x.val ^ y
  g = y*x.val^(y-1) * x.grad
  return dualnum(r, g)
end