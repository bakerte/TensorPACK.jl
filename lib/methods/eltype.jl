import Base.eltype
"""
  G = eltype(A)

Returns the element type `G` contained in the `T` field of `denstens` `A`
"""
function eltype(A::tens{W}) where W <: Number
  return W
end

function eltype(A::diagonal{W}) where W <: Number
  return W
end

function eltype(op::bigvec)
  return op.type
end

"""
  G = elnumtype(A)

Returns the element type `G` contained in the `T` field of `denstens` `A`.  Same behavior as `eltype` for `denstens`.

See also: [`eltype`](@ref) [`denstens`](@ref)
"""
function elnumtype(A::tens{W}) where W <: Number
  return eltype(A)
end
export elnumtype

function eltype(X::TNobj)
  return eltype(X.N)
end

import Base.eltype
"""
    eltype(A)

element type of a Qtensor (i.e., `T` field)

See also: [`Qtens`](@ref)
"""
function eltype(A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return W
end

#  import .tensor.elnumtype
function elnumtype(A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return eltype(A)
end
