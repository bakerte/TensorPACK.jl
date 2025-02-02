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
  G = eltype(A)

Returns the element type `G` contained in the `T` field of `denstens` `A`
"""
function eltype(A::tens{W}) where W <: Number
  return W
end

"""
  G = eltype(A)

Returns the element type `G` contained in the `T` field of `diagonal` `A`
"""
function eltype(A::Diagonal{W}) where W <: Number
  return W
end

"""
  G = eltype(A)

Returns the element type `G` contained in the `T` field of a large vector type (`bigvec`) `A`
"""
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

"""
  G = eltype(A)

Returns the element type `G` contained in the `T` field of `TNobj` `A`
"""
function eltype(X::TNobj)
  return eltype(X.N)
end

"""
    eltype(A)

element type of a Qtensor (i.e., `T` field)

See also: [`Qtens`](@ref)
"""
function eltype(A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return W
end

"""
    elnumtype(A)

Alias for `eltype` on a `qarray`

See also: [`eltype`](@ref) [`qarray`](@ref)
"""
function elnumtype(A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return eltype(A)
end
