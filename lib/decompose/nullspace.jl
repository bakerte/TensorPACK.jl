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
    B = nullspace(A[,left=false,atol=0.0,rtol=...])

Finds the nullspace vector `B` of an input matrix `A` of `TensType`

#Optional inputs:
+ `left`: finds left (if true from U) or right (if false from V) from `svd` inside of function
+ `atol`: tolerance parameter in algorithm
+ `rtol`: tolerance parameter in algorithm

See also: [`svd`](@ref)
"""
function nullspace(A::TensType; left::Bool=false,atol::Real = 0.0, rtol::Real = (min(size(A, 1), size(A, 2))*eps(real(float(one(eltype(A))))))*iszero(atol))

  U,D,V = svd(A)
  Dvals = [searchindex(D,i,i) for i = 1:size(D,1)]
  tol = max(atol, Dvals[1]*rtol)
  indstart = sum(s -> s .> tol, Dvals) + 1

  minval = minimum(abs.(Dvals))
  ipos = findfirst(w->isapprox(abs(Dvals[w]),minval),1:length(Dvals))

  if length(ipos) > 1
    g = rand(length(ipos))
    minpos = ipos[g]
  else
    minpos = ipos[1]
  end
  outTens = left ? U[:,minpos:minpos] : V[minpos:minpos,:]
  return outTens
end
export nullspace