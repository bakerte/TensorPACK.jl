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
  eye([,d=2])

creates an identity matrix of integer size `d`
"""
function eye(;d::intType=2)
  return eye(Float64,d) #zeros(d,d)+LinearAlgebra.I
end

"""
  eye(type[,d=2])

creates an identity matrix of integer size `d` with type 'type'
"""
function eye(type::DataType;d::intType=2)
  return eye(type,d) #zeros(d,d)+LinearAlgebra.I
end

"""
  eye(d)

creates an identity matrix of integer size `d`
"""
function eye(d::intType)
  return eye(Float64,d) #zeros(d,d)+LinearAlgebra.I
end

"""
  eye(type,d)

creates an identity matrix of integer size `d` with type 'type'
"""
function eye(type::DataType,d::intType)
  return diagonal(ones(type,d))
end
export eye

#=
function makeIdarray(W::DataType,ldim::Integer;addone::Bool=false,addRightDim::Bool=false,loadleft::Bool=true)
  return makeId(W,ldim,ldim;addone=addone,addRightDim=addRightDim)
end
=#
"""
  G = eye(W,ldim,rdim[;addone=false,addRightDim=false,loadleft=true])

Generates a matrix (`G`) with dimensions `ldim`x`rdim` with data type `W`. Can also add an index of size 1 (`addone`) to either the left (`addRightDim=false`) or right (`addRightDim=true`).

See: [`eye`](@ref)
"""
function eye(W::DataType,ldim::Integer,rdim::Integer;addone::Bool=false,addRightDim::Bool=false)#,loadleft::Bool=true)
  oneval = W(1)
  if addone
    if addRightDim
      newsize = (ldim,rdim,1)
    else
      newsize = (1,ldim,rdim)
    end
  else
    newsize = (ldim,rdim)
  end
  Id = zeros(W,prod(newsize))
  stop = min(ldim,rdim) #loadleft ? ldim : rdim
  @inbounds @simd for i = 1:stop
    Id[i + ldim*(i-1)] = oneval
  end
  return tens(newsize,Id)
end

#=
"""
    G = eye(W,ldim[;addone=false,addRightDim=false])

Generates an identity tensor (`denstens` of output type `W` [default Float64], `G`) that contracts to the identity operator (rank-2) when applied on another tensor traces out that a pair of indices of equal dimension. Parameter `ldim` denotes the size of the index to contract over, `addone` if `true` leaves two indices of size 1 on indices 1 and 4 (rank-4 tensor). Option `addRightDim` adds one index of size 1 on the third index (rank-3).

See also: [`denstens`](@ref)
"""
function eye(W::DataType,ldim::Integer;addone::Bool=false,addRightDim::Bool=false)
  if addone
    return tens(eye(W,ldim,ldim;addone=addone,addRightDim=addRightDim))
  else
    return eye(W,ldim)
  end
end
=#

"""
    G = eye(ldim,rdim[,addone=false,addRightDim=false])

Generates an identity tensor (`denstens` of output type `Float64` [default Float64], `G`) that contracts to the identity operator (rank-2) when applied on another tensor traces out that a pair of indices of equal dimension. Parameter `ldim` x `rdim` denotes the size of the index to contract over, `addone` if `true` leaves two indices of size 1 on indices 1 and 4 (rank-4 tensor). Option `addRightDim` adds one index of size 1 on the third index (rank-3).

See also: [`denstens`](@ref)
"""
function eye(ldim::Integer,rdim::Integer;addone::Bool=false,addRightDim::Bool=false,loadleft::Bool=true)
  return eye(Float64,ldim,rdim;addone=addone,addRightDim=addRightDim)
end
#=
function makeId(ldim::Integer;addone::Bool=false,addRightDim::Bool=false)
  return tens(makeIdarray(Float64,ldim,ldim;addone=addone,addRightDim=addRightDim))
end

function makeId(ldim::Integer,rdim::Integer;addone::Bool=false,addRightDim::Bool=false,loadleft::Bool=true)
  return tens(makeIdarray(Float64,ldim,rdim;addone=addone,addRightDim=addRightDim))
end
=#
"""
    G = eye(A,iA)

Similar to other `eye`, but inputs tensor `A` (`denstens` or `Array`) with a pair of indices to trace over. Indices are selected by vector `iA` that identifies which indices from which to generate the identity matrix `G`.

Example: `eye(A,[[1,2],[3,4],[5,6]])`

See also: [`denstens`](@ref)
"""
function eye(A::tens{W},iA::Array{P,1}) where {W <: Number, P <: Union{Integer,Tuple}}
  ldim = size(A,iA[1][1])
  Id = eye(W,ldim,ldim,addone=true,addRightDim=true)
  for g = 2:length(iA)
    ldim = size(A,iA[g][1])
    addId = eye(W,ldim,ldim,addone=true,addRightDim=false)
    Id = reshape!(Id,prod(size(Id)),1)
    Id = Id * addId #contract(Id,ndims(Id),addId,1)
  end
  newsize = Array{intType,1}(undef,2*length(iA))
  @inbounds @simd for x = 1:length(iA)
    newsize[2*x-1] = size(A,iA[x][1])
    newsize[2*x] = size(A,iA[x][1])
  end
  return reshape!(Id,newsize...)
end

function eye(A::tens{W},iA::Integer) where {W <: Number}
  return eye(A,[iA...])
end

function eye(A::Array{W,N},iA::Array{P,1}) where {N, W <: Number,P <: Union{Integer,Tuple}}
  a = size(A,iA[1][1])
  Id = zeros(W,a,a) + LinearAlgebra.I #makeId(W,size(A,iA[1][1]),addone=true,addRightDim=false)
  Id = reshape!(Id,prod(size(Id)),1)

  for g = 2:length(iA)
    b = size(A,iA[g][1])
    addId = zeros(W,b,b) + LinearAlgebra.I
    addId = reshape!(addId,1,prod(size(addId)))
    Id = reshape!(Id,prod(size(Id)),1)
    Id *= addId #contract(Id,ndims(Id),addId,1)
  end
  newsize = Array{intType,1}(undef,2*length(iA))
  @inbounds @simd for x = 1:length(iA)
    newsize[2*x-1] = size(A,iA[x][1])
    newsize[2*x] = size(A,iA[x][1])
  end
  return reshape!(Id,newsize...)
end

function eye(A::Array{W,N},iA::Integer...) where {W <: Number,N}
  return eye(A,[iA...])
end
export eye






#  import ..tensor.makeId
"""
    eye(A,iA)

generates an identity matrix from tensor `A` with indices `iA`

See also: [`trace`](@ref)
"""
function eye(A::Qtens{W,Q},iA::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}
#  lsize,finalsizes = Idhelper(A,iA)
  lsize = prod(w->size(A,iA[w][1]),1:length(iA))
  leftsizes = ntuple(w->size(A,iA[w][1]),length(iA))

  finalsizes = (leftsizes...,leftsizes...)

  leftQNs = [[getQnum(iA[w],x,A) for x = 1:size(A,iA[w])] for w = 1:length(iA)]
  rightQNs = [inv.(leftQNs[a]) for a = 1:length(leftQNs)]
  newQnumMat = vcat(leftQNs,rightQNs)
  typeA = eltype(A)
  Id = Qtens(typeA,newQnumMat,blockfct=eye)
  return Id
end
