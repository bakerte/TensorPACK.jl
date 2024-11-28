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

Generates a random tensor `G` from inputting `AbstractArray` `A` (rank-2)
"""
function rand(rho::AbstractArray{W,N}) where {W <: Number, N}
  return rand(W, size(rho))
end

"""
    G = rand(W,A)

Generates a random tensor `G` from inputting `AbstractArray` `A` (rank-2) with element type `W`
"""
function rand(W::DataType,sizearray::AbstractArray{R,1}) where {R <: intType}
  return rand(W, sizearray...)
end




"""
    rand(A,arrows)

generates a random tensor from inputting another tensor (rank-2); inverts quantum number symmetries according to `arrows` containing a `Bool`
"""
function rand(Qlabels::Array{Array{Q,1},1}, arrows::Array{Bool,1};type::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
  return Qtens(newQlabels,type=type,flux=flux,blockfct=rand)
end

"""
    rand(W,Qlabels)

generates a random tensor from inputting another tensor (rank-2) from `Qlabels` containing `qnum`s with element type `W`
"""
function rand(type::DataType,Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels)) where Q <: Qnum
  return Qtens(Qlabels,type=type,flux=flux,currblock=currblock,blockfct=rand)
end

"""
    rand(Qlabels)

generates a random tensor from inputting another tensor (rank-2) from `Qlabels` containing `qnum`s
"""
function rand(Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels),type::DataType=Float64) where Q <: Qnum
  return Qtens(Qlabels,type=type,flux=flux,currblock=currblock,blockfct=rand)
end
#=
"""
  B = rand([DataType,]A)

Creates a random tensor `B` of the same type as `B`
"""
=#
"""
    rand(Qtens)

generates a random tensor from inputting another tensor (rank-2) from `Qtens`
"""
function rand(currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,type=W,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=rand)
end

"""
    rand(W,Qtens)

generates a random tensor from inputting another tensor (rank-2) from `Qtens` with element type `W`
"""
function rand(type::DataType,currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,type=type,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=rand)
end

