#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker and M.P. Thompson (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#
 
"""
  Module: decompositions

Decompose a tensor

See also: [`contractions`](@ref)
"""
#=
module decompositions
using ..tensor
using ..contractions
import LinearAlgebra
=#

#       +---------------------------------------+
#       |                                       |
#>------+  Quantum Number conserving operation  +---------<
#       |                                       |
#       +---------------------------------------+
#using ..QN
#using ..Qtensor

function makeU(nQN::Integer,keepq::Array{Bool,1},outU::Array{Array{W,2},1},QtensA::Qtens{W,Q},
                finalinds::Array{Array{P,2},1},newqindexL::Array{P,1},newqindexLsum::Array{Q,1},
                leftflux::Bool,Linds::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}

  finalnQN = sum(keepq)
  finalUinds = Array{NTuple{2,Array{P,2}},1}(undef,finalnQN)
  newQblocksum = Array{NTuple{2,Q},1}(undef,finalnQN)
  counter = 0
  for q = 1:nQN
    if keepq[q]
      counter += 1
      left = QtensA.ind[q][1]
      right = finalinds[q]
      finalUinds[counter] = (left,right)
      
      newQblocksum[counter] = (QtensA.Qblocksum[q][1],newqindexLsum[q])
    end
  end
  finalUQnumMat = Array{Array{intType,1},1}(undef,length(Linds)+1)
  @inbounds @simd for q = 1:length(Linds)
    finalUQnumMat[q] = QtensA.QnumMat[Linds[q]]
  end
  finalUQnumMat[end] = newqindexL

  Uflux = leftflux ? QtensA.flux : Q()

  leftinds = [i for i = 1:length(Linds)]
  rightinds = [length(Linds) + 1]
  newUQsize = [leftinds,rightinds]
  newUblocks = (leftinds,rightinds)
  finalUQnumSum = Array{Array{Q,1},1}(undef,length(Linds)+1)
  @inbounds for q = 1:length(Linds)
    finalUQnumSum[q] = QtensA.QnumSum[Linds[q]]
  end
  finalUQnumSum[end] = newqindexLsum

  return Qtens{W,Q}(newUQsize,outU,finalUinds,newUblocks,newQblocksum,finalUQnumMat,finalUQnumSum,Uflux)
end

function makeV(nQN::Integer,keepq::Array{Bool,1},outV::Array{Array{W,2},1},QtensA::Qtens{W,Q},
                finalinds::Array{Array{P,2},1},newqindexR::Array{P,1},newqindexRsum::Array{Q,1},
                leftflux::Bool,Rinds::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}
       
  finalnQN = sum(keepq)
  finalVinds = Array{NTuple{2,Array{P,2}},1}(undef,finalnQN)
  newQblocksum = Array{NTuple{2,Q},1}(undef,finalnQN)
  counter = 0
  @inbounds for q = 1:nQN
    if keepq[q]
      counter += 1
      left = finalinds[q]
      right = QtensA.ind[q][2]
      finalVinds[counter] = (left,right)

      newQblocksum[counter] = (newqindexRsum[q],QtensA.Qblocksum[q][2])
    end
  end
  finalVQnumMat = Array{Array{intType,1},1}(undef,length(Rinds)+1)
  finalVQnumMat[1] = newqindexR
  for q = 1:length(Rinds)
    finalVQnumMat[q+1] = QtensA.QnumMat[Rinds[q]]
  end


  Vflux = !leftflux ? QtensA.flux : Q()

  leftinds = [1]
  rightinds = [i+1 for i = 1:length(Rinds)]
  newVQsize = [leftinds,rightinds]
  newVblocks = (leftinds,rightinds)

  finalVQnumSum = Array{Array{Q,1},1}(undef,length(Rinds)+1)
  finalVQnumSum[1] = newqindexRsum
  @inbounds for q = 1:length(Rinds)
    finalVQnumSum[q+1] = QtensA.QnumSum[Rinds[q]]
  end
  return Qtens{W,Q}(newVQsize,outV,finalVinds,newVblocks,newQblocksum,finalVQnumMat,finalVQnumSum,Vflux)
end

function makeD(nQN::Integer,keepq::Array{Bool,1},outD::Array{Diagonal{W},1},QtensA::Qtens{W,Q},
                finalinds::Array{Array{P,2},1},newqindexL::Array{P,1},newqindexR::Array{P,1},
                newqindexRsum::Array{Q,1},newqindexLsum::Array{Q,1}) where {W <: Number, Q <: Qnum, P <: Integer}

  finalnQN = sum(keepq)
  finalDinds = Array{NTuple{2,Array{P,2}},1}(undef,finalnQN)
  newQblocksum = Array{NTuple{2,Q},1}(undef,finalnQN)
  counter = 0
  @inbounds @simd for q = 1:nQN
    if keepq[q]
      counter += 1
      left = finalinds[q]
      right = finalinds[q]
      finalDinds[counter] = (left,right)

      newQblocksum[counter] = (newqindexRsum[q],newqindexLsum[q])
    end
  end
  finalDQnumMat = [newqindexR,newqindexL]

  Dflux = Q()
  leftinds = [1]
  rightinds = [2]
  newDQsize = [leftinds,rightinds]
  newDblocks = (leftinds,rightinds)
  finalDQnumSum = [newqindexRsum,newqindexLsum]

  return Qtens{W,Q}(newDQsize,outD,finalDinds,newDblocks,newQblocksum,finalDQnumMat,finalDQnumSum,Dflux)
end

#=
@inline function threeterm(arr::Array{Array{W,2},1};decomposer::Function=safesvd) where W <: Number
  nQN = length(arr)
  newU = Array{Array{W,2},1}(undef,nQN)
  newD = Array{Array{W,1},1}(undef,nQN)
  newV = Array{Array{W,2},1}(undef,nQN)
  for q = 1:nQN
    newU[q],newD[q],newV[q] = decomposer(arr[q])
  end
  return newU,newD,newV
end
=#
function svd(QtensA::Qtens{W,Q};a::Integer=size(QtensA,1),b::Integer=size(QtensA,2),
              cutoff::Float64 = 0.,m::Integer = 0,minm::Integer=2,nozeros::Bool=true,power::Number=2,leftflux::Bool=false,mag::Float64=0.,
              effZero::Real=defzero,keepdeg::Bool=false,decomposer::Function=libsvd) where {W <: Number, Q <: Qnum}

  Tsize = QtensA.size
  Linds = Tsize[1]
  Rinds = Tsize[2]

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
  newD = Array{Array{W,1},1}(undef,nQN)
  newV = Array{Array{W,2},1}(undef,nQN)
  @inbounds for q = 1:nQN
    U,newD[q],Vt = decomposer(A.T[q])
    a,b = size(A.T[q])
    sizeD = length(newD[q])
    newU[q] = reshape!(U,a,sizeD)
    newV[q] = reshape!(Vt,sizeD,b)
  end

  m_intervals,sizeD,truncerr,sumD = truncate(newD...,m=m,minm=minm,mag=mag,cutoff=cutoff,effZero=effZero,nozeros=nozeros,power=power,keepdeg=keepdeg)

  thism = 0
  @inbounds @simd for q = 1:length(m_intervals)
    thism += length(m_intervals[q])
  end

  newqindexL = Array{intType,1}(undef,max(thism,minm))
  keepq = Array{Bool,1}(undef,nQN)
  tempD = Array{Diagonal{W},1}(undef,nQN)
  finalinds = Array{Array{intType,2},1}(undef,nQN)
  @inbounds for q = 1:nQN
    if length(m_intervals[q]) != 0
      keepq[q] = m_intervals[q][1] != 0 && m_intervals[q][1] <= m_intervals[q][end]

      if keepq[q]
        newU[q] = newU[q][:,m_intervals[q]]

        tempD[q] = Diagonal(newD[q][m_intervals[q]])
        newV[q] = newV[q][m_intervals[q],:]

        offset = 0
        @inbounds @simd for w = 1:q-1
          offset += length(m_intervals[w])
        end

        tempvec = Array{intType,2}(undef,1,length(m_intervals[q]))
        @inbounds @simd for i = 1:length(m_intervals[q])
          tempvec[i] = i + offset - 1
        end
        finalinds[q] = tempvec
        
        @inbounds @simd for i = 1:length(finalinds[q])
          newqindexL[i + offset] = q
        end
      end
    else
      keepq[q] = false
    end
  end

  if length(newqindexL) > thism

    findzeroQN = 1
    zeroQN = Q()
    while findzeroQN < nQN && QNsummary[findzeroQN] != zeroQN
      findzeroQN += 1
    end
    if findzeroQN > nQN
      push!(QNsummary,zeroQN)
      push!(invQNsummary,zeroQN)
    end

    lastind = 0
    @inbounds @simd for q = 1:nQN
      lastind += length(m_intervals[q])
    end

    @inbounds @simd for w = lastind+1:length(newqindexL)
      newqindexL[w] = findzeroQN
    end
  end



  newqindexLsum = QNsummary
  newqindexRsum = invQNsummary
  newqindexR = newqindexL


  if sum(keepq) < nQN
    outU = newU[keepq]
    outD = tempD[keepq]
    outV = newV[keepq]
  else
    outU = newU
    outD = tempD
    outV = newV
  end


  U = makeU(nQN,keepq,outU,A,finalinds,newqindexL,newqindexLsum,leftflux,Linds)
  D = makeD(nQN,keepq,outD,A,finalinds,newqindexL,newqindexR,newqindexRsum,newqindexLsum)
  V = makeV(nQN,keepq,outV,A,finalinds,newqindexR,newqindexRsum,leftflux,Rinds)

  return U,D,V,truncerr,sumD
end

function svd!(QtensA::Qtens{W,Q};a::Integer=size(QtensA,1),b::Integer=size(QtensA,2),
              cutoff::Float64 = 0.,m::Integer = 0,minm::Integer=2,nozeros::Bool=true,power::Number=2,leftflux::Bool=false,mag::Float64=0.,
              effZero::Real=defzero,keepdeg::Bool=false) where {W <: Number, Q <: Qnum}
  return svd(QtensA,a=a,b=b,cutoff=cutoff,m=m,minm=minm,nozeros=nozeros,power=power,
                    leftflux=leftflux,mag=mag,effZero=effZero,keepdeg=keepdeg,decomposer=libsvd!)
end


function eigen!(QtensA::Qtens{W,Q};cutoff::Float64 = 0.,m::Integer = 0,effZero::Real=defzero,
              minm::Integer=2,nozeros::Bool=false,
              power::Number=1,leftflux::Bool=false,mag::Float64=0.,
              decomposer::Function=libeigen!,keepdeg::Bool=false,transpose::Bool=false,rev::Bool=true) where {W <: Number, Q <: Qnum}

  Tsize = QtensA.size
  Linds = Tsize[1]
  Rinds = Tsize[2]

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
  newD = Array{Array{W,1},1}(undef,nQN)
  for q = 1:nQN
    newD[q],newU[q] = decomposer(A.T[q])
  end

  if transpose
    for q = 1:nQN
      newU[q] = permutedims!(newU[q],[2,1])
      if eltype(newU[q]) <: Complex
        conj!(newU[q])
      end
    end
  end

  m_intervals,sizeD,truncerr,sumD = truncate(newD...,m=m,minm=minm,mag=mag,cutoff=cutoff,effZero=effZero,nozeros=nozeros,power=power,keepdeg=keepdeg,rev=rev)

  thism = 0
  @inbounds @simd for q = 1:length(m_intervals)
    thism += length(m_intervals[q])
  end

  newqindexL = Array{intType,1}(undef,max(thism,minm))
  keepq = Array{Bool,1}(undef,nQN)
  tempD = Array{Diagonal{W},1}(undef,nQN)
  finalinds = Array{Array{intType,2},1}(undef,nQN)
  @inbounds for q = 1:nQN
    if length(m_intervals[q]) != 0
      keepq[q] = m_intervals[q][1] != 0 && m_intervals[q][1] <= m_intervals[q][end]

      if keepq[q]
        if transpose
          newU[q] = newU[q][m_intervals[q],:]
        else
          newU[q] = newU[q][:,m_intervals[q]]
        end

        tempD[q] = Diagonal(newD[q][m_intervals[q]])

        offset = 0
        @inbounds @simd for w = 1:q-1
          offset += length(m_intervals[w])
        end

        tempvec = Array{intType,2}(undef,1,length(m_intervals[q]))
        @inbounds @simd for i = 1:length(m_intervals[q])
          tempvec[i] = i + offset - 1
        end
        finalinds[q] = tempvec
        
        @inbounds @simd for i = 1:length(finalinds[q])
          newqindexL[i + offset] = q
        end
      end
    else
      keepq[q] = false
    end
  end

  if length(newqindexL) > thism

    findzeroQN = 1
    zeroQN = Q()
    while findzeroQN < nQN && QNsummary[findzeroQN] != zeroQN
      findzeroQN += 1
    end
    if findzeroQN > nQN
      push!(QNsummary,zeroQN)
      push!(invQNsummary,zeroQN)
    end

    lastind = 0
    @inbounds @simd for q = 1:nQN
      lastind += length(finalinds[q])
    end

    @inbounds @simd for w = lastind+1:length(newqindexL)
      newqindexL[w] = findzeroQN
    end
  end

  newqindexLsum = QNsummary
  newqindexRsum = invQNsummary
  newqindexR = newqindexL

  if sum(keepq) < nQN
    outU = newU[keepq]
    outD = tempD[keepq]
  else
    outU = newU
    outD = tempD
  end

  if transpose
    U = makeV(nQN,keepq,outU,A,finalinds,newqindexR,newqindexRsum,!leftflux,Rinds)
  else
    U = makeU(nQN,keepq,outU,A,finalinds,newqindexL,newqindexLsum,leftflux,Linds)
  end
  D = makeD(nQN,keepq,outD,A,finalinds,newqindexL,newqindexR,newqindexRsum,newqindexLsum)

  return D,U,truncerr,sumD
end

function eigen(QtensA::Qtens{W,Q};cutoff::Float64 = 0.,m::Integer = 0,a::Integer=size(QtensA,1),b::Integer=size(QtensA,2),
                minm::Integer=2,nozeros::Bool=false,
                power::Number=1,leftflux::Bool=false,mag::Float64=0.,effZero::Real=defzero,
                decomposer::Function=libeigen,keepdeg::Bool=false,transpose::Bool=false,rev::Bool=true) where {W <: Number, Q <: Qnum}
  return eigen!(QtensA,cutoff=cutoff,m=m,minm=minm,nozeros=nozeros,power=power,effZero=effZero,
                leftflux=leftflux,mag=mag,decomposer=libeigen,keepdeg=keepdeg,transpose=transpose,rev=rev)
end

function qr(QtensA::Qtens{W,Q};a::Integer=1,b::Integer=1,leftflux::Bool=false,decomposer::Function=libqr,mag::Number=1.) where {W <: Number, Q <: Qnum}
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
  sumD = mag

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
  return U,V,truncerr,mag
end
export qr

function qr!(QtensA::Qtens{W,Q};leftflux::Bool=false,a::Integer=1,b::Integer=1) where {W <: Number, Q <: Qnum}
  return qr(QtensA,leftflux=leftflux,decomposer=libqr!,a=a,b=b)
end
export qr!

function lq(QtensA::Qtens{W,Q};leftflux::Bool=false,a::Integer=1,b::Integer=1) where {W <: Number, Q <: Qnum}
  return qr(QtensA,leftflux=leftflux,decomposer=liblq,a=a,b=b)
end
export lq

function lq!(QtensA::Qtens{W,Q};leftflux::Bool=false,a::Integer=1,b::Integer=1) where {W <: Number, Q <: Qnum}
  return qr(QtensA,leftflux=leftflux,decomposer=liblq!,a=a,b=b)
end
export lq!

function svdvals(QtensA::Qtens{W,Q};mag::Number=1.) where {W <: Number, Q <: Qnum}
  Rsize = QtensA.size
  Linds = Rsize[1]
  Rinds = Rsize[2]

  A = changeblock(QtensA,Linds,Rinds)

  nQN = length(A.T)
  newD = [svdvals(A.T[q]) for q = 1:nQN]
  return vcat(newD...)
end

function nullspace(A::TensType; left::Bool=false,atol::Real = 0.0, rtol::Real = (min(size(A, 1), size(A, 2))*eps(real(float(one(eltype(A))))))*iszero(atol))

  U,D,V = svd(A)
  Dvals = [searchindex(D,i,i) for i = 1:size(D,1)]
  tol = max(atol, Dvals[1]*rtol)
  indstart = sum(s -> s .> tol, Dvals) + 1

  minval = minimum(abs.(Dvals))
  ipos = findfirst(w->isapprox(abs(Dvals[w]),minval),1:length(Dvals))

  if length(ipos) > 1
    g = rand(length(ipos))
    minpos = ipos[g]
  else
    minpos = ipos[1]
  end
  outTens = left ? U[:,minpos:minpos] : V[minpos:minpos,:]
  return outTens
end
export nullspace