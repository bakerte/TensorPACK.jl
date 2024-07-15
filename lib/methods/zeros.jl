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

import Base.zeros
"""
    G = zeros(A)

Generates a tensor of zeros `G` from inputting another tensor (rank-2) or from quantum number labels
"""
function zeros(A::AbstractArray{W,N}) where {W <: Number, N}
  return zeros(W,size(A))
end

function zeros(A::tens{W}) where W <: Number
  return tens{W}(zeros(W,size(A)))
end

function zeros(R::DataType,A::Union{AbstractArray{W,N},Array{W,N}}) where {W <: Number, N}
  return zeros(R,size(A))
end

function zeros(R::DataType,A::tens{W}) where W <: Number
  return tens{R}(zeros(R,size(A)))
end

import Base.zero
"""
  G = zero(A)

Creates a new tensor `G` of the same size as `A` but filled with zeros (same as internal julia function)

See also: [`zero`](@ref)
"""
function zero(M::tens{W}) where W <: Number
  return tens{W}(zeros(W,size(M)))
end

import Base.zero
"""
  zero(x)

Creates the zero value for a given `dualnum`, `x`
"""
function zero(x::dualnum{W}) where W <: Number
  return dualnum{W}(W(0),W(1))
end





import Base.zeros
function zeros(Qlabels::Array{Array{Q,1},1}, arrows::Array{Bool,1};datatype::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
  return Qtens(newQlabels,datatype=datatype,flux=flux,blockfct=zeros)
end

function zeros(Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels),datatype::DataType=Float64) where Q <: Qnum
  return Qtens(Qlabels,datatype=datatype,flux=flux,currblock=currblock,blockfct=zeros)
end

function zeros(datatype::DataType,Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels)) where Q <: Qnum
  return Qtens(Qlabels,datatype=datatype,flux=flux,currblock=currblock,blockfct=zeros)
end

function zeros(currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,datatype=W,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=zeros)
end

function zeros(datatype::DataType,currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,datatype=datatype,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=zeros)
end

import Base.zero
  """Like the default function zero(t::Array), return an object with the same properties containing only zeros."""
  function zero(Qt::qarray)
  return zeros(Qt)
end
