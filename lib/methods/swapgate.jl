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
    swapgate(A,iA,B,iB)

generates a swap gate (order of indices: in index for `A`, in index for `B`, out index for `A`, out index for `B`) for `A` and `B`'s indices `iA` and `iB`
"""
function (swapgate(A::TensType,iA::W,B::TensType,iB::R) where {W <: Union{intvecType,Array{Array{P,1},1}},R <: Union{intvecType,Array{Array{P,1},1}}}) where P <: Integer
  LId = eye(A,iA)
  RId = eye(B,iB)
  if typeof(LId) <: qarray
    push!(LId.size,[ndims(LId)+1])
  else
    LId = reshape(LId,size(LId)...,1)
  end
  if typeof(RId) <: qarray
    push!(RId.size,[ndims(RId)+1])
  else
    RId = reshape(RId,size(RId)...,1)
  end
  fullId = contract(LId,4,RId,4)
  return permute(fullId,[1,3,2,4])
end