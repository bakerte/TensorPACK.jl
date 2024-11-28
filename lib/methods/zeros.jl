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
    G = zeros(A)

Generates a tensor of zeros `G` an AbstractArray `A`
"""
function zeros(A::AbstractArray{W,N}) where {W <: Number, N}
  return zeros(W,size(A))
end

"""
    G = zeros(A)

Generates a tensor of zeros `G` an `denstens` `A`
"""
function zeros(A::tens{W}) where W <: Number
  return tens{W}(zeros(W,size(A)...))
end

"""
    G = zeros(R,A)

Generates a tensor of zeros `G` an AbstractArray `A` with a different type `R`
"""
function zeros(R::DataType,A::Union{AbstractArray{W,N},Array{W,N}}) where {W <: Number, N}
  return zeros(R,size(A)...)
end

"""
    G = zeros(R,A)

Generates a tensor of zeros `G` an `denstens` `A` with a different type `R`
"""
function zeros(R::DataType,A::tens{W}) where W <: Number
  return tens{R}(zeros(R,size(A)...))
end

"""
    G = zero(A)

Generates a tensor of zeros `G` an `denstens` `A`

See also: [`zero`](@ref)
"""
function zero(M::tens{W}) where W <: Number
  return zeros(M) #tens{W}(zeros(W,size(M)...))
end

"""
    zero(x)

Creates the zero value for a given `dualnum`, `x`
"""
function zero(x::dualnum{W}) where W <: Number
  return dualnum{W}(W(0),W(1))
end



"""
    zeros(Qt,arrows[,type=Float64,flux=Q()])

Creates a `qarray` from input `qarray` `Qt` full of zeros of type `type`; can switch flux of quantum numbers with `false` value in `arrows` (requires one true or false value for each index of `Qt`); optional `flux` parameter changes overall flux of the tensor (default zero element of group)
"""
function zeros(Qlabels::Array{Array{Q,1},1}, arrows::Array{Bool,1};type::DataType=Float64,flux::Q=Q()) where Q <: Qnum
  newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
  return Qtens(newQlabels,type=type,flux=flux,blockfct=zeros)
end

"""
    zeros(Qlabels[,flux=Q()])

Creates a `qarray` from input `Qlabels` (array of array of `Qnum`) full of zeros; optional `flux` parameter changes overall flux of the tensor (default zero element of group)
"""
function zeros(Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels),type::DataType=Float64) where Q <: Qnum
  return Qtens(Qlabels,type=type,flux=flux,currblock=currblock,blockfct=zeros)
end

"""
    zeros(W,Qt[,flux=Q()])

Creates a `qarray` from input `Qlabels` (array of array of `Qnum`) full of zeros of element type `W`; optional `flux` parameter changes overall flux of the tensor (default zero element of group)
"""
function zeros(type::DataType,Qlabels::Array{Array{Q,1},1};flux::Q=Q(),currblock::currblockTypes=equalblocks(Qlabels)) where Q <: Qnum
  return Qtens(Qlabels,type=type,flux=flux,currblock=currblock,blockfct=zeros)
end

"""
    zeros(Qt)

Creates a copy of an input `qarray` `Qt` full of zeros
"""
function zeros(currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,type=W,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=zeros)
end

"""
    zeros(W,Qt)

Creates a copy of an input `qarray` `Qt` full of zeros but of element type `W`
"""
function zeros(type::DataType,currQtens::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Qlabels = fullQnumMat(currQtens)
  return Qtens(Qlabels,type=type,currblock=currQtens.currblock,flux=currQtens.flux,blockfct=zeros)
end

"""
    zero(Qt)

Alias for `zeros` on a `qarray` `Qt`
"""
function zero(Qt::qarray)
  return zeros(Qt)
end
