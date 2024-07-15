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
    D = svdvals(A)

Returns singular values of input tensor `A` for any `densTensType`; output is a vector of singular values
"""
@inline function svdvals(A::densTensType) #Union{AbstractVecOrMat,denstens}
  return LinearAlgebra.svdvals(A)
end
export svdvals
#=
"""
  svdvals!(A)

Returns singular values of input tensor `A`, modifying the tensor
"""
function svdvals(A::denstens)
  U,B,V = libsvd(A,job='N') 
  return B
end
export svdvals!
=#


#       +------------------------+
#>------| Quantum number version |---------<
#       +------------------------+

"""
    D = svdvals(A)

Returns singular values of input tensor `A` for any `Qtens`; output is a vector of singular values
"""
function svdvals(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Rsize = QtensA.size
  Linds = Rsize[1]
  Rinds = Rsize[2]

  A = changeblock(QtensA,Linds,Rinds)

  nQN = length(A.T)
  newD = [svdvals(A.T[q]) for q = 1:nQN]
  return vcat(newD...)
end
