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
    getQnum(a,b,QnumMat,QnumSum)

Finds `Qnum` from `QnumSum` (vector of vectors of `Qnum`) based on index from `QnumMat` (vector of vectors of integers); `a` indexes the index represented on the `qarray`; `b` represents the value of the index
"""
function getQnum(a::Integer,b::Integer,QnumMat::Array{Array{P,1},1},QnumSum::Array{Array{Q,1},1}) where {Q <: Qnum, P <: intType}
  Qnumber = QnumMat[a][b]
  return QnumSum[a][Qnumber]
end

"""
    getQnum(a,b,Qt)

Finds `Qnum` from `qarray` `Qt`; `a` indexes the index represented on the `qarray`; `b` represents the value of the index
"""
function getQnum(a::Integer,b::Integer,Qt::Qtens{W,Q}) where {Q <: Qnum, W <: Number}
  return getQnum(a,b,Qt.QnumMat,Qt.QnumSum)
end