"""
  dualnum(val[,grad=1.])

Initializes the `dualnum` type with value `val` and a gradient `grad` (default value 1)
"""
function dualnum(val::W; grad::W=W(1)) where W <: Number
  return dualnum{W}(val, grad)
end
#=
"""
  dualnum(val,grad)

Initializes the `dualnum` type with value `val` and a gradient `grad`
"""
function dualnum(val::W, grad::W) where W <: Number
  return dualnum{W}(val, grad)
end
=#
"""
  dualnum(W)

Creates a `dualnum` of type `W`
"""
function dualnum(W::DataType)
  return dualnum{W}(W(0),W(1))
end
export dualnum