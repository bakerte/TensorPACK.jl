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
    B = norm(A)

Froebenius norm of a `denstens` `A` as output `G`

See: [`denstens`](@ref)`
"""
function norm(A::Union{denstens,diagonal})
  return real(norm(A.T))
end

"""
    B = norm(A)

Gives the norm of named tensor `A`
"""
function norm(A::TNobj)
  return norm(A.N)
end

function norm(A::dtens)
  return norm(A[0])
end

"""
    B = norm(A)

L2 norm of a vector `A` containing `dualnum`s
"""
function norm(A::AbstractArray{W}) where W <: dualnum
  total = 0
  for x in A
    total += abs2(x.val)
  end
  return Base.sqrt(total)
end

"""
    B = norm(A)

Froebenius norm of a Qtensor
"""
function norm(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return real(metricdistance(QtensA,power=2,fct=abs))
end

"""
    B = norm!(A)

Normalizes any input `TensType` `A`
"""
function norm!(M::TensType)
  return div!(M,norm(M))
end


"""
    B = norm(A)

Returns the norm of a vector of `dualnum` elements
"""
function norm(A::Vector{dualnum})
  total = 0
  for x in A
    total += x.real^2
  end
  return Base.sqrt(total)
end
