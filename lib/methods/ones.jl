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

import Base.ones
"""
    G = zeros(A)

Generates a tensor of ones `G` from inputting another tensor (rank-2) or from quantum number labels
"""
function ones(A::AbstractArray{W,N}) where {W <: Number, N}
  return ones(W,size(A))
end

function ones(A::tens{W}) where W <: Number
  return tens{W}(ones(W,size(A)))
end

function ones(R::DataType,A::Union{AbstractArray{W,N},Array{W,N}}) where {W <: Number, N}
  return ones(R,size(A))
end

function ones(R::DataType,A::tens{W}) where W <: Number
  return tens{R}(ones(R,size(A)))
end


import Base.ones
function ones(Qlabels::Array{Array{Q,1},1}, arrows::Array{Bool,1};datatype::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
  return Qtens(newQlabels,datatype=datatype,flux=flux,blockfct=ones)
end

function ones(Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels),datatype::DataType=Float64) where Q <: Qnum
  return Qtens(Qlabels,datatype=datatype,flux=flux,currblock=currblock,blockfct=ones)
end

function ones(datatype::DataType,Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels)) where Q <: Qnum
  return Qtens(Qlabels,datatype=datatype,flux=flux,currblock=currblock,blockfct=ones)
end

function ones(currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,datatype=W,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=ones)
end

function ones(datatype::DataType,currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,datatype=datatype,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=ones)
end
