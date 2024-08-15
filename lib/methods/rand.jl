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
    G = rand(A)

Generates a random tensor `G` from inputting `denstens` `A` (rank-2) or from quantum number labels
"""
function rand(rho::tens{W}) where W <: Number
  return tens{W}(rand(W, size(rho)))
end

"""
    G = rand(A)

Generates a random tensor `G` from inputting `AbstractArray` `A` (rank-2) or from quantum number labels
"""
function rand(rho::AbstractArray{W,N}) where {W <: Number, N}
  return rand(W, size(rho))
end






"""
    rand([DataType,]A[,arrows])

generates a random tensor from inputting another tensor (rank-2) or from quantum number labels; can assign `arrows` for Qtensors if alternate arrow convention is requested
"""
function rand(Qlabels::Array{Array{Q,1},1}, arrows::Array{Bool,1};datatype::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
  return Qtens(newQlabels,datatype=datatype,flux=flux,blockfct=rand)
end

function rand(datatype::DataType,Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels)) where Q <: Qnum
  return Qtens(Qlabels,datatype=datatype,flux=flux,currblock=currblock,blockfct=rand)
end

function rand(Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels),datatype::DataType=Float64) where Q <: Qnum
  return Qtens(Qlabels,datatype=datatype,flux=flux,currblock=currblock,blockfct=rand)
end
#=
"""
  B = rand([DataType,]A)

Creates a random tensor `B` of the same type as `B`
"""
=#
function rand(currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,datatype=W,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=rand)
end

function rand(datatype::DataType,currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,datatype=datatype,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=rand)
end

