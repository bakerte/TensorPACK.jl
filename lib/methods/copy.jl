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
    G = copy(M)

Copies a `denstens` (output `G`); `deepcopy` is inherently not type stable, so this function should be used instead

See: [`denstens`](@ref) [`deepcopy`](@ref)
"""
function copy(A::tens{W}) where {W <: Number}
  return tens{W}(A.size,copy(A.T))
end

"""
    G = copy(M)

Copies a `diagonal` (output `G`); `deepcopy` is inherently not type stable, so this function should be used instead

See: [`diagonal`](@ref) [`deepcopy`](@ref)
"""
function copy(A::Diagonal{W}) where {W <: Number}
  return Diagonal{W}(copy(A.T))
end

"""
    G = copy(names,X[,ext=".dmrjulia",copyext=ext])

Copies tensors from `largeEnv` input `X` to a new tensor with a vector of strings `names` representing the new filenames
"""
function copy(names::Array{String,1},X::bigvec;ext::String=file_extension,copyext::String=ext)
  newObj = bigvec([names[(i-1) % length(names) + 1] for i = 1:length(X)],X.type)
  for i = 1:length(X)
    Y = tensorfromdisc(X.V[i],ext=ext)
    tensor2disc(newObj.V[i],Y,ext=copyext)
  end
  return newObj
end

"""
    B = copy(names,A[,ext=file_extension])

copy any large storage type `A` with new names `names`

See also: [`file_extension`](@ref)
"""
function copy(names::String,X::bigvec;ext::String=file_extension,copyext::String=ext)
  return copy(names .* X.V,X,ext=ext,copyext=copyext)
end

"""
    B = copy(A)

Returns a copy of named tensor `A`

See also: [`nametens`](@ref)
"""
function copy(A::nametens)
  return nametens(copy(A.N),copy(A.names))
end

"""
    B = copy(A)

Returns a copy of directed tensor `A`

See also: [`directedtens`](@ref)
"""
function copy(A::directedtens)
  return directedtens(copy(A.N),copy(A.names),copy(A.conj),copy(A.arrows))
end

"""
    B = copy(A)

Returns a copy of network of any `TNnetwork` `A`

See also: [`TNnetwork`](@ref)
"""
function copy(A::TNnetwork)
  return network([copy(A.net[i]) for i = 1:length(A)])
end


"""
    B = copy(A)

Returns a copy of network of `dtens` `A`

See also: [`dtens`](@ref)
"""
function copy(A::dtens)
  return dtens(copy(A[0]),copy(A[1]))
end

"""
    copy!(Qt)

Copies a Qtensor; `deepcopy` is inherently not type stable, so this function should be used instead; `copy!` performs a shallow copy on all fields
"""
function copy!(Qt::Qtens{T,Q}) where {T <: Number, Q <: Qnum}
  return Qtens{T,Q}(Qt.size,Qt.T,Qt.ind,Qt.currblock,Qt.Qblocksum,Qt.QnumMat,Qt.QnumSum,Qt.flux)
end

"""
    copy(Qt)

Copies a Qtensor; `deepcopy` is inherently not type stable, so this function should be used instead
"""
function copy(Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  newsize = [copy(Qt.size[i]) for i = 1:length(Qt.size)]
  copyQtT = Array{Array{eltype(Qt),2},1}(undef,length(Qt.T))
  @inbounds @simd for q = 1:length(copyQtT)
    copyQtT[q] = copy(Qt.T[q])
  end
  copyQtind = [(copy(Qt.ind[q][1]),copy(Qt.ind[q][2])) for q = 1:length(Qt.ind)]
  newcurrblock = (copy(Qt.currblock[1]),copy(Qt.currblock[2]))
  newQblocksum = Qt.Qblocksum
  newQnumSum = [copy(Qt.QnumSum[i]) for i = 1:length(Qt.QnumSum)]
  newQnumMat = [copy(Qt.QnumMat[i]) for i = 1:length(Qt.QnumMat)]
  return Qtens{W,Q}(newsize,copyQtT,copyQtind,newcurrblock,
                    newQblocksum,newQnumMat,newQnumSum,Qt.flux)
end

