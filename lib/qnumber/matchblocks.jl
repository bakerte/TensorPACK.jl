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
    LQNs = genQNs(conjvar,A,Aind)

Generates a set of quantum numbers `LQNs` corresponding to the row or column of the matrix equivalent of `A` (a `qarray`) defined across the indices `Aind`
"""
function genQNs(conjvar::Bool,A::Qtens{W,Q},Aind::intType) where {W <: Number, Q <: Qnum}
  LQNs = Array{Q,1}(undef,length(A.Qblocksum))
  if conjvar
    @inbounds @simd for q = 1:length(A.Qblocksum)
      LQNs[q] = inv(A.Qblocksum[q][Aind])
    end
  else
    @inbounds @simd for q = 1:length(A.Qblocksum)
      LQNs[q] = A.Qblocksum[q][Aind]
    end
  end
  return LQNs
end

"""
    C = matchblocks(conjvar,A,B[,ind=(2,1),matchQN=Q()])

finds matching quantum number blocks in `qarray`s `A` and `B` where an input `conjvar` is a tuple of length 2 to determine which of `A` and `B` are conjugated; `ind` specifies which side of each tensor should be used to check if there is a match such that the number from `A` adds to the number from `B` to equal `matchQN`
"""
function matchblocks(conjvar::NTuple{2,Bool},A::Qtens{W,Q},B::Qtens{S,Q};
                      ind::NTuple{2,P}=(2,1),matchQN::Q=Q()) where {W <: Number, S <: Number, Q <: Qnum, P <: Integer}

  Aind,Bind = ind
  LQNs = genQNs(conjvar[1],A,Aind)
  RQNs = genQNs(conjvar[2],B,Bind)

  return matchblocks(LQNs,RQNs,matchQN=matchQN)
end

"""
   loadtup!(Aorder,q,k,newblock)

loads `Aorder` (an array of length-2 tuples of integer input) for index `q` and new index order `k`
"""
function loadtup!(Aorder::Array{NTuple{2,intType},1},q::intType,k::intType,newblock::intType)
  Aorder[q] = (k,newblock)
  nothing
end

"""
    C = matchblocks(LQNs,RQNs[,matchQN=Q()])

When contracting two `qarrays`, the quantum number blocks are not guaranteed to be in order for contraction, this function finds the matches between a set of left (right) quantum numbers `LQNs` (`RQNs`); all quantum numbers should add to `matchQN` (default zero value of the quantum number); returns as an array of tuples with two integers as elements
"""
function matchblocks(LQNs::Array{Q,1},RQNs::Array{Q,1};matchQN::Q=Q()) where Q <: Qnum
  Aorder = Array{intType,1}(undef,length(LQNs))
  outmatches = 0
  @inbounds for q = 1:length(LQNs)
    matchBool = false
    w = 0
    @inbounds while w < length(RQNs) && !matchBool
      w += 1
      thisQN = LQNs[q] + RQNs[w]
      matchBool = thisQN == matchQN
    end
    if matchBool
      Aorder[q] = w
      outmatches += 1
    else
      Aorder[q] = 0
    end
  end

  outAorder = Array{NTuple{2,intType},1}(undef,outmatches)
  counter = 0
  k = 0
  @inbounds while counter < outmatches
    k += 1
    if Aorder[k] != 0
      counter += 1
      loadtup!(outAorder,counter,k,Aorder[k])
    end
  end
  return outAorder
end
