
"""
  G = norm(A)

Froebenius norm of a `denstens` `A` as output `G`

See: [`denstens`](@ref)`
"""
function norm(A::denstens)
  return real(norm(A.T))
end
export norm

"""
    norm(A)

Gives the norm of named tensor `A`
"""
function norm(A::TNobj)
  return norm(A.N)
end

function norm(A::dtens)
  return norm(A[0])
end

"""
  norm(A)

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
    norm(A)

Froebenius norm of a Qtensor
"""
function norm(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return real(metricdistance(QtensA,power=2,fct=abs))
end
export norm

function norm!(M::TensType)
  return div!(M,norm(M))
end
export norm!



function norm(A::Vector{dualnum})
  total = 0
  for x in A
    total += x.real^2
  end
  return Base.sqrt(total)
end
