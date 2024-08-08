

"""
  G = sum(A)

Sum elements of a `denstens` `A` (ouptut `G`)

See: [`denstens`](@ref)
"""
function sum(A::denstens)
  return sum(A.T)
end

function sum(R::TNobj)
  return sum(R.N)
end



"""
  metricdistance(A[,power=])

computes the Forebenius norm of all elements in the tensor...equal to L^power norm
"""
function metricdistance(D::Qtens{W,Q};power::Number=1,fct::Function=identity) where {W <: Number, Q <: Qnum}
  powersums = W(0)
  for q = 1:length(D.T)
    for y = 1:size(D.T[q],2)
      @inbounds @simd for x = 1:size(D.T[q],1)
        powersums += fct(D.T[q][x,y])^power
      end
    end
  end
  return powersums^(1/power)
end

"""
    sum(A)

sum elements of a Qtensor
"""
function sum(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return metricdistance(QtensA,power=1)
end

