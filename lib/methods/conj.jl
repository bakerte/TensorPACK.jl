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
  G = conj!(A)

Conjugates a `denstens` `A` in-place (creates no copy) but also has output `G`

See also: [`conj!`](@ref) [`denstens`](@ref)`
"""
function conj!(currtens::tens{W}) where W <: Number
  if !(W <: Real)
    LinearAlgebra.conj!(currtens.T)
  end
  return currtens
end

"""
  G = conj(A)

Conjugates a `denstens` by creating a copy `G`

See also: [`conj`](@ref) [`denstens`](@ref)`
"""
function conj(M::tens{W}) where W <: Number
  if W <: Real
    out = M
  else
    newT = LinearAlgebra.conj(M.T)
    out = tens{W}(M.size,newT)
  end
  return out
end

"""
  G = conj!(A)

Conjugates a `diagonal` `A` in-place (creates no copy) but also has output `G`

See also: [`conj!`](@ref) [`diagonal`](@ref)`
"""
function conj!(M::diagonal{W}) where W <: Number
  if !(W <: Real)
    LinearAlgebra.conj!(M.T)
  end
  return M
end

"""
  G = conj(A)

Conjugates a `diagonal` by creating a copy `G`

See also: [`conj`](@ref) [`diagonal`](@ref)`
"""
function conj(M::diagonal{W}) where W <: Number
  if !(W <: Real)
    newT = LinearAlgebra.conj(M.T)
    out = diagonal{W}(newT)
  else
    out = M
  end
  return M
end





"""
    conj!(A)

Conjugates named tensor `A` in-place

See also: [`conj`](@ref)
"""
function conj!(A::nametens)
  conj!(A.N)
  return A
end




#=
function conj!(A::nametens)
  conj!(A.N)
  A.conj = !A.conj
  return A
end
=#

#=
function conj!(A::directedtens)
  @inbounds @simd for w = 1:ndims(A)
    A.arrows[w] = !A.arrows[w]
  end
  A.conj = !A.conj
  nothing
end
=#
"""
    conj(A)

Conjugates named tensor `A`

See also: [`conj!`](@ref)
"""
function conj(A::TNobj)
  B = copy(A)
  conj!(B)
  return B
end

"""
    conj(A)

Conjugates directedtens `A`

See also: [`directedtens`](@ref)
"""
function conj(A::directedtens)
  return directedtens(conj(A.N),copy(A.names),-1*A.arrows)
end

"""
    conj!(A)

Conjugates directedtens `A`

See also: [`directedtens`](@ref)
"""
function conj!(A::directedtens)
  @inbounds @simd for w = 1:length(A.arrows)
    A.arrows[w] *= -1
  end
  conj!(A.N)
  return A
end

"""
    conj(A)

Conjugates named MPS `A`

See also: [`conj!`](@ref)
"""
function conj(A::TNnetwork)
  return conj!(copy(A))
end

function conj!(A::TNnetwork)
  B = dual!(A)
  @inbounds @simd for w = 1:length(B)
    conj!(B[w])
  end
  return B
end

"""
    conj(A)

conjugates a Qtensor by creating a copy

See also: [`conj!`](@ref)
"""
function conj(currQtens::qarray)
  Qtens = copy(currQtens)
  conj!(Qtens)
  return Qtens
end

"""
    conj!(A)

conjugates a Qtensor in place

See also: [`conj`](@ref)
"""
function conj!(currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds for q = 1:length(currQtens.T)
    currQtens.T[q] = conj!(currQtens.T[q])
  end

  newQnumSum = Array{Array{Q,1},1}(undef,length(currQtens.QnumSum))
  @inbounds for w = 1:length(currQtens.QnumSum)
    newQnumSum[w] = inv.(currQtens.QnumSum[w])
  end
  currQtens.QnumSum = newQnumSum

  newQblocksum = Array{NTuple{2,Q},1}(undef,length(currQtens.Qblocksum))
  @inbounds for q = 1:length(newQblocksum)
    newQblocksum[q] = (inv(currQtens.Qblocksum[q][1]),inv(currQtens.Qblocksum[q][2]))
  end
  currQtens.Qblocksum = newQblocksum

  currQtens.flux = inv(currQtens.flux)
  return currQtens
end


"""
  conj!(x)

Conjugation (in-place) of a `dualnum`
"""
function conj!(x::dualnum{W}) where W <: Number
  if W <: Complex
    out = conj(x)
  else
    out = x
  end
  return out
end


"""
  conj(x)

Conjugation of a `dualnum`
"""
function conj(x::dualnum)
  r = conj(x.val)
  g = conj(x.grad)
  return dualnum(r, g)
end


conj!(x::dualnum) = conj(x)

