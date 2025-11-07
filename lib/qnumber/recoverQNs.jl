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
    B = recoverQNs(q,QnumMat,QnumSum)

Returns `Qnum` `B` associated with the `q`th value of the `QnumMat` (vectors of integers) according to the `QnumSum`

See also: [`Qnum`](@ref)
"""
function recoverQNs(q::Integer,QnumMat::Array{Array{intType,1},1},QnumSum::Array{Array{Q,1},1}) where Q <: Qnum
  return [getQnum(q,i,QnumMat,QnumSum) for i = 1:length(QnumMat[q])]
end

"""
    B = recoverQNs(q,Qt)

Returns `Qnum` `B` associated with the `q`th value of the .QnumMat (vectors of integers) according to the .QnumSum of a tensor `Qt`

See also: [`Qnum`](@ref)
"""
function recoverQNs(q::Integer,Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return recoverQNs(q,Qt.QnumMat,Qt.QnumSum)
end
export recoverQNs