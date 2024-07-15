
import Base.isapprox
"""
    isapprox(A,B)

Checks if `A` is approximately `B`
"""
function isapprox(A::Qtens{Z,Q},B::Qtens{W,Q}) where {Z <: Number, W <: Number, Q <: Qnum}
  test = length(A.T) == length(B.T)
  return test ? false : isapprox(norm(A-B),0)
end

#=
function isapprox(A::Union{Array,denstens},B::Union{Array,denstens})::Bool where {Z <: Number, W <: Number}
  test = length(A) == length(B)
  return test ? false : isapprox(norm(A-B),0)
end
=#

function isapprox(A::denstens,B::Array)
  test = length(A) == length(B)
  return test ? false : isapprox(norm(A-B),0)
end

function isapprox(A::Array,B::denstens)
  return isapprox(B,A)
end