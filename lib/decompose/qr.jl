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
    Q,R,0.,1. = qr(A,vecA[,decomposer=libqr,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2.

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `Q`: a unitary matrix
+ `R`: a weighted matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function qr(AA::TensType,vecA::Array{Array{W,1},1};decomposer::Function=libqr,leftflux::Bool=false,
              a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2])) where W <: Integer
  AB,Lsizes,Rsizes,a,b = getorder(AA,vecA)
  Qmat,Rmat = decomposer(AB,a,b)

  innerdim = size(Qmat,2)

  outU = unreshape!(Qmat,Lsizes...,innerdim)
  outV = unreshape!(Rmat,innerdim,Rsizes...)
  return outU,outV
end

"""
    Q,R,0.,1. = qr!(A,vecA[,decomposer=libqr!,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2. Overwrites some input data from `A`

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `Q`: a unitary matrix
+ `R`: a weighted matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function qr!(AA::TensType,vecA::Array{Array{W,1},1};decomposer::Function=libqr!,leftflux::Bool=false,
              a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2])) where W <: Integer
  return qr(AA,vecA,decomposer=decomposer,a=a,b=b)
end

"""
    Q,R,0.,1. = qr(A[,decomposer=libqr,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` (`denstens`) for `svd` and then unreshapes U and V matrices on return

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `Q`: a unitary matrix
+ `R`: a weighted matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function qr(AA::densTensType;decomposer::Function=libqr,leftflux::Bool=false,a::Integer=size(AA,1),b::Integer=size(AA,2))
  return decomposer(AA,a,b)
end

"""
    Q,R,0.,1. = qr!(A[,decomposer=libqr!,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` (`denstens`) for `svd` and then unreshapes U and V matrices on return

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `Q`: a unitary matrix
+ `R`: a weighted matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function qr!(AA::densTensType;decomposer::Function=libqr!,leftflux::Bool=false,a::Integer=size(AA,1),b::Integer=size(AA,2))
  return qr(AA,decomposer=decomposer,a=a,b=b)
end
export qr,qr!




#       +------------------------+
#>------| Quantum number version |---------<
#       +------------------------+


"""
    Q,R,0.,1. = qr(A[,decomposer=libqr,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` (`Qtens`) for `svd` and then unreshapes U and V matrices on return

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `Q`: a unitary matrix
+ `R`: a weighted matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function qr(QtensA::Qtens{W,Q};a::Integer=1,b::Integer=1,leftflux::Bool=false,decomposer::Function=libqr) where {W <: Number, Q <: Qnum}
  Rsize = QtensA.size
  Linds = Rsize[1]
  Rinds = Rsize[2]

  A = changeblock(QtensA,Linds,Rinds)
  nQN = length(A.T)

  LRinds = 1
  QNsummary = Array{Q,1}(undef,nQN)
  invQNsummary = Array{Q,1}(undef,nQN)
  @inbounds @simd for q = 1:nQN
    invQNsummary[q] = A.Qblocksum[q][LRinds]
    QNsummary[q] = inv(A.Qblocksum[q][LRinds])
  end

  nQN = length(A.T)
  newU = Array{Array{W,2},1}(undef,nQN)
  newV = Array{Array{W,2},1}(undef,nQN)
  for q = 1:nQN
    newU[q],newV[q] = decomposer(A.T[q],size(A.T[q],1),size(A.T[q],2))
  end

  thism = 0
  @inbounds @simd for q = 1:length(newU)
    thism += size(newU[q],2)
  end
#  sumD = mag

  outU,outV = newU,newV


  qstarts = Array{intType,1}(undef,nQN+1)
  qstarts[1] = 0
  @inbounds for q = 2:nQN
    qstarts[q] = 0
    @inbounds @simd for w = 1:q-1
      qstarts[q] += size(newU[w],2)
    end
  end
  qstarts[end] = thism
#  qranges = [UnitRange(qstarts[q]+1,qstarts[q+1]) for q = 1:nQN]
  qranges = Array{UnitRange{intType},1}(undef,nQN)
  @inbounds @simd for q = 1:nQN
    qranges[q] = UnitRange(qstarts[q]+1,qstarts[q+1])
  end

  truncerr = 0.

  newqindexL = Array{intType,1}(undef,thism)
  finalinds = Array{Array{intType,2},1}(undef,nQN)
  @inbounds @simd for q = 1:nQN
    finalinds[q] = Array{intType,2}(undef,1,length(qranges[q]))
    @inbounds @simd for i = 1:length(qranges[q])
      w = qranges[q][i]
      finalinds[q][i] = w-1
    end
  end

  @inbounds for q = 1:nQN
    offset = 0
    @inbounds @simd for w = 1:q-1
      offset += length(finalinds[w])
    end

    @inbounds @simd for i = 1:length(finalinds[q])
      newqindexL[i + offset] = q
    end
  end


  newqindexLsum = QNsummary
  newqindexRsum = invQNsummary
  newqindexR = newqindexL

  keepq = [true for q = 1:nQN]

  U = makeU(nQN,keepq,outU,A,finalinds,newqindexL,newqindexLsum,leftflux,Linds)
  V = makeV(nQN,keepq,outV,A,finalinds,newqindexR,newqindexRsum,leftflux,Rinds)
  return U,V,0.,1. #truncerr,mag
end
export qr

"""
    Q,R,0.,1. = qr!(A[,decomposer=libqr!,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` (`Qtens`) for `svd` and then unreshapes U and V matrices on return; some operations may be in place

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `Q`: a unitary matrix
+ `R`: a weighted matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function qr!(QtensA::Qtens{W,Q};leftflux::Bool=false,a::Integer=1,b::Integer=1,decomposer::Function=libqr) where {W <: Number, Q <: Qnum}
  return qr(QtensA,leftflux=leftflux,decomposer=decomposer,a=a,b=b)
end
export qr!


"""
    Q,R,0.,1. = qr(A,order[,name="qrind",leftadd="L",leftflux=false,decomposer=libqr])

Generates qr decomposition of named tensor `A` according to `order`

# Naming created index:
+ the index to the left of `D` is `vcat(name,leftadd)`

#Inputs:
+ `A`: Any `TensType` in the library
+ `order`: vector containing index groups given as `String`

#Optional additional inputs:
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `Q`: a unitary matrix
+ `R`: a weighted matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function qr(AA::nametens,order::Array{Array{B,1},1};decomposer::Function=libqr,name::String="qrind",leftflux::Bool=false,leftadd::String="L") where B <: String

  left = matchnames(AA,order[1])
  right = matchnames(AA,order[2])
  neworder = [left,right]
  leftname = name*leftadd

  Q,R = qr(AA.N,neworder,leftflux=leftflux,decomposer=decomposer)

  TNobjQ = nametens(Q,vcat(AA.names[left]...,[leftname]))
  TNobjR = nametens(R,[leftname,AA.names[right]...])

  return TNobjQ,TNobjR
end


