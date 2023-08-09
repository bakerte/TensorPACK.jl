#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

"""
  Module: contractions

Contract two tensors together

See also: [`decompositions`](@ref)
"""
#=
module contractions
using ..tensor
import LinearAlgebra
=#

function *(X::qarray,Y::qarray)
  if ndims(X) == 1 && ndims(Y) == 2
    X = reshape(X,size(X,1),1)
  end
  return contract(X,ndims(X),Y,1)
end

function dot(C::Qtens{W,Q},D::Qtens{R,Q};Lfct::Function=adjoint,Rfct::Function=identity) where {W <: Number, R <: Number, Q <: Qnum}
  newtype = typeof(W(1)*R(1))

  A = changeblock(C,intType[],intType[i for i = 1:length(C.QnumMat)])
  B = changeblock(D,intType[i for i = 1:length(D.QnumMat)],intType[])
  conjA = Lfct != identity
  conjB = Rfct != identity
  commonblocks = matchblocks((conjA,conjB),A,B,ind=(2,1))

  val = newtype(0)
  @inbounds for q = 1:length(commonblocks)
    Aqind = commonblocks[q][1]
    Bqind = commonblocks[q][2]
    val += dot(A.T[Aqind],B.T[Bqind],Lfct=Lfct,Rfct=Rfct)
  end
  return val
end

#not sure how to assign matrix blocks...or how to get information from that to A and B vectors
#must be set up correctly with changeblock
function dot(A::Qtens{W,Q},H::Qtens{Y,Q},B::Qtens{R,Q};Lfct::Function=adjoint,Rfct::Function=identity) where {W <: Number, Y <: Number, R <: Number, Q <: Qnum}
  Acommonblocks = matchblocks((conjA,false),A,H,ind=(2,1))
  Bcommonblocks = matchblocks((conjA,conjB),A,B,ind=(2,1))
  newtype = typeof(W(1)*R(1))
  val = newtype(0)
  @inbounds for q = 1:length(commonblocks)
    Aqind = Acommonblocks[q][1]
    Hqind = Acommonblocks[q][2]
    Bqind = Bcommonblocks[q][2]
    if length(A.T[Aqind]) * length(B.T[Bqind]) != length(H.T[Hqind])
      error("unequal sizes in dot for quantum number tensors for A block: $Aqind, H block: $Hqind, and B block: $Bqind")
    end
    if Aqind != 0 && Hqind != 0 && Bqind != 0
      val += dot(A.T[Aqind],H.T[Hqind],B.T[Bqind],Lfct=Lfct,Rfct=Rfct)
    end
  end
  return val
end


#       +------------------------+
#>------| Quantum number version |---------<
#       +------------------------+

function checkblocks(Aqind::intType,Bqind::intType,A::Qtens{W,Q},B::Qtens{R,Q},inputA::Union{Diagonal{W},Array{W,2}},inputB::Union{Diagonal{R},Array{R,2}};Aind::intType=2,Bind::intType=1) where {W <: Number, R <: Number, Q <: Qnum}
  checksum = 0
  minAB = min(length(A.ind[Aqind][Aind]),length(B.ind[Bqind][Bind]))
  w = 0
  @inbounds while w < minAB && checksum == 0
    w += 1
    checksum += A.ind[Aqind][Aind][w]-B.ind[Bqind][Bind][w]
  end

  mulblockA = inputA
  mulblockB = inputB

  if !(length(A.ind[Aqind][Aind]) == length(B.ind[Bqind][Bind]) && checksum == 0)

    blocksizes = ntuple(n->length(A.QnumMat[A.currblock[Aind][n]]),length(A.currblock[Aind]))

    indsA = A.ind[Aqind][Aind]
    indsB = B.ind[Bqind][Bind]

    Lrowcols = Array{intType,1}(undef,size(indsA,2))
    Rrowcols = Array{intType,1}(undef,size(indsB,2))

    for p = 1:2
      if p == 1
        G = Lrowcols
        K = indsA
      else
        G = Rrowcols
        K = indsB
      end
      for x = 1:length(G)
        z = K[end,x]
        @inbounds @simd for y = length(blocksizes)-1:-1:1
          z *= blocksizes[y]
          z += K[y,x]
        end
        G[x] = z+1
      end
    end

    equalinds = length(Lrowcols) == length(Rrowcols)
    if equalinds
      k = 0
      while equalinds && w < length(Lrowcols)
        k += 1
        equalinds = Lrowcols[k] == Rrowcols[k]
      end
    end
    if !equalinds
      commoninds = intersect(Lrowcols,Rrowcols)
      if !issorted(commoninds)
        sort!(commoninds)
      end
      orderL = sortperm(Lrowcols)
      orderR = sortperm(Rrowcols)

      keepL = Array{intType,1}(undef,length(commoninds))
      keepR = Array{intType,1}(undef,length(commoninds))

      for p = 1:length(commoninds)
        b = 1
        @inbounds while b < length(orderR) && Lrowcols[orderL[p]] != Rrowcols[orderR[b]]
          b += 1
        end
        keepL[p] = orderL[p]
        keepR[p] = orderR[b]
      end

      mulblockA = mulblockA[:,keepL]
      mulblockB = mulblockB[keepR,:]
    end
  end
  return mulblockA,mulblockB
end

 function genblockinds(offset::intType,firstblock::Array{intType,1})
  @inbounds @simd for w = 1:length(firstblock)
    firstblock[w] = offset + w
  end
  nothing
end

 function loadnewsize(newsize::Array{Array{intType,1},1})
  counter = 0
  for w = 1:length(newsize)
    @inbounds @simd for a = 1:length(newsize[w])
      counter += 1
      newsize[w][a] = counter
    end
  end
  nothing
end

 function loadarraynewsize(newsize::Array{Array{intType,1},1},offset::intType,notconA::Array{intType,1},QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds for w = 1:length(notconA)
    newsize[w+offset] = Array{intType,1}(undef,length(QtensA.size[notconA[w]]))
  end
  nothing
end

 function loadnewQnumMat(newQnumMat::Array{Array{intType,1},1},offset::Integer,notconA::Array{intType,1},A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds @simd for q = 1:length(notconA)
    newQnumMat[q+offset] = A.QnumMat[notconA[q]]
  end
  nothing
end

 function loadnewQnumSum(newQnumMat::Array{Array{Q,1},1},offset::Integer,notconA::Array{intType,1},A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds @simd for q = 1:length(notconA)
    newQnumMat[q+offset] = A.QnumSum[notconA[q]]
  end
  nothing
end

 function loadnewQnumSum_inv(newQnumMat::Array{Array{Q,1},1},offset::Integer,notconA::Array{intType,1},A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds @simd for q = 1:length(notconA)
    newQnumMat[q+offset] = inv.(A.QnumSum[notconA[q]])
  end
  nothing
end

function permq(A::Qtens{W,Q},iA::Array{intType,1}) where {W <: Number, Q <: Qnum} #K
  nopermL = length(iA) == length(A.currblock[1]) && issorted(A.currblock[1])
  w = 0
  @inbounds while nopermL && w < length(iA)
    w += 1
    nopermL = w == A.currblock[1][w] && w == iA[w]
  end

  nopermR = !nopermL && length(iA) == length(A.currblock[2]) && issorted(A.currblock[2])
  #println(nopermR)
  if nopermR
    w =length(iA)
    end_dim = length(A.QnumMat)
    @inbounds while nopermR && w > 0
      nopermR = A.currblock[2][w] == end_dim && iA[w] == end_dim
      end_dim -= 1
      w -= 1
    end
  end
  return nopermL,nopermR
end

function dmul!(X::Qtens{R,Q},Y::Qtens{W,Q}) where {R <: Number, W <: Number, Q <: Qnum}
  return maincontractor(false,false,X,(ndims(X),),Y,(1,),inplace=true)
end

function maincontractor(conjA::Bool,conjB::Bool,QtensA::Qtens{W,Q},vecA::Tuple,QtensB::Qtens{R,Q},vecB::Tuple,Z::Qtens{S,Q}...;alpha::Number=W(1),beta::Number=W(1),inplace::Bool=false) where {W <: Number, R <: Number, S <: Number, Q <: Qnum}
  if QtensA === QtensB && ((conjA && W <: Complex) || (conjB && R <: Complex))
    QtensB = copy(QtensA)
  end

  conA,notconA = getinds(QtensA,vecA)
  conB,notconB = getinds(QtensB,vecB)




  AnopermL,AnopermR = permq(QtensA,conA)
  BnopermL,BnopermR = permq(QtensB,conB)

  Aperm,transA = willperm(conjA,eltype(QtensA),AnopermL,AnopermR)
  Bperm,transB = willperm(conjB,eltype(QtensB),BnopermR,BnopermL)

  if Aperm
    A = QtensA
  else
    A = changeblock(QtensA,notconA,conA)
    transA = 'N'
  end
  if Bperm
    B = QtensB
  else
    B = changeblock(QtensB,conB,notconB)
    transB = 'N'
  end

  Aretind,notAretind = transA == 'N' ? (2,1) : (1,2)
  Bretind,notBretind = transB == 'N' ? (1,2) : (2,1)
  commonblocks = matchblocks((conjA,conjB),A,B,ind=(Aretind,Bretind))


  outType = W == R ? W : typeof(W(1) * R(1))

  usealpha = !isapprox(alpha,1)

  numQNs = length(commonblocks)

  useZ = length(Z) > 0
  if useZ
    Zone = [i for i = 1:length(notconA)]
    Ztwo = [i + length(notconA) for i = 1:length(notconB)]
    Z = changeblock(Zed,Zone,Ztwo)
    Zcommonblocks = matchblocks((conjA,false),A,Zed,ind=(2,1))
    type_beta = eltype(beta) == outType && !isapprox(beta,1) ? beta : convert(outType,beta)
    type_alpha = typeof(alpha) == outType ? alpha : convert(outType,alpha)
  elseif usealpha
    type_alpha = typeof(alpha) == outType ? alpha : convert(outType,alpha)
  end


  ############

  newrowcols = Array{NTuple{2,Array{intType,2}},1}(undef,numQNs)
  newQblocksum = Array{NTuple{2,Q},1}(undef,numQNs)
  outTens = Array{Array{outType,2},1}(undef,numQNs)

  #optional types like alpha input into functions from upstream calls create perhaps a type instability? Disabling alpha saves one allocation

  @inbounds for q = 1:numQNs
    Aqind = commonblocks[q][1]
    Bqind = commonblocks[q][2]

    if conjA && W <: Complex && transA == 'N'
      inputA = conj(A.T[Aqind])
    else
      inputA = A.T[Aqind]
    end

    if conjB && R <: Complex && transB == 'N'
      inputB = conj(B.T[Bqind])
    else
      inputB = B.T[Bqind]
    end

    mulblockA,mulblockB = checkblocks(Aqind,Bqind,A,B,inputA,inputB,Aind=Aretind,Bind=Bretind)

    Lsize,innersizeL = size(mulblockA,notAretind),size(mulblockA,Aretind)
    Rsize,innersizeR = size(mulblockB,notBretind),size(mulblockB,Bretind)

    Adiag = typeof(mulblockA) <: Diagonal
    if Adiag || typeof(mulblockB) <: Diagonal
#      println("IN HERE? ",transA," ",transB)
      if transA == 'N' && transB == 'N'
        if inplace
          outTens[q] = dmul!(mulblockA,mulblockB)
        else
          outTens[q] = mulblockA*mulblockB
        end
      else
        if Adiag
          if eltype(mulblockA) <: Complex
            mulblockA = conj(mulblockA)
          end
        else
          if eltype(mulblockB) <: Complex
            mulblockB = conj(mulblockB)
          end
        end
        if inplace
          outTens[q] = dmul!(mulblockB,mulblockA)
        else
          outTens[q] = mulblockB*mulblockA
        end
        outTens[q] = transpose(outTens[q])
      end
    else
      if useZ
        Zqind = Zcommonblocks[q][2]
        outTens[q] = libmult(transA,transB,type_alpha,mulblockA,mulblockB,type_beta,inputZed,Lsize,innersizeL,innersizeR,Rsize)
      elseif usealpha
        outTens[q] = libmult(transA,transB,type_alpha,mulblockA,mulblockB,Lsize,innersizeL,innersizeR,Rsize)
      else
        outTens[q] = libmult(transA,transB,mulblockA,mulblockB,Lsize,innersizeL,innersizeR,Rsize)
      end
    end

    LQN = conjA ? inv(A.Qblocksum[Aqind][notAretind]) : A.Qblocksum[Aqind][notAretind]
    RQN = conjB ? inv(B.Qblocksum[Bqind][notBretind]) : B.Qblocksum[Bqind][notBretind]
    newQblocksum[q] = (LQN,RQN)

    newrowcols[q] = (A.ind[Aqind][notAretind],B.ind[Bqind][notBretind])
  end
  ############

  newQnumMat = Array{Array{intType,1},1}(undef,length(notconA)+length(notconB))
  loadnewQnumMat(newQnumMat,0,notconA,A)
  loadnewQnumMat(newQnumMat,length(notconA),notconB,B)

  newQnumSum = Array{Array{Q,1},1}(undef,length(notconA)+length(notconB))
  if conjA
    loadnewQnumSum_inv(newQnumSum,0,notconA,A)
  else
    loadnewQnumSum(newQnumSum,0,notconA,A)
  end
  if conjB
    loadnewQnumSum_inv(newQnumSum,length(notconA),notconB,B)
  else
    loadnewQnumSum(newQnumSum,length(notconA),notconB,B)
  end

  newsize = Array{Array{intType,1},1}(undef,length(notconA)+length(notconB))
  loadarraynewsize(newsize,0,notconA,QtensA)
  loadarraynewsize(newsize,length(notconA),notconB,QtensB)

  loadnewsize(newsize)

  keepers = Bool[size(outTens[q],1) > 0 && size(outTens[q],2) > 0 for q = 1:length(outTens)]

  if !conjA && !conjB
    newflux = QtensA.flux + QtensB.flux
  elseif conjA && !conjB
    newflux = QtensB.flux - QtensA.flux
  elseif !conjA && conjB
    newflux = QtensA.flux - QtensB.flux
  elseif conjA && conjB
    newflux = -(QtensA.flux,QtensB.flux)
  end
  firstblock = Array{intType,1}(undef,length(notconA))
  genblockinds(0,firstblock)

  secondblock = Array{intType,1}(undef,length(notconB))
  genblockinds(length(notconA),secondblock)
  newcurrblocks = (firstblock,secondblock)

  if sum(keepers) < length(keepers)
    newT = outTens[keepers]
    newinds = newrowcols[keepers]
    newQblocks = newQblocksum[keepers]
  else
    newT = outTens
    newinds = newrowcols
    newQblocks = newQblocksum
  end
  return Qtens{outType,Q}(newsize,newT,newinds,newcurrblocks,newQblocks,newQnumMat,newQnumSum,newflux)
end


#         +-----------------+
#>--------|  Check contract |------<
#         +-----------------+

function checkcontract(A::TensType,iA::intvecType,B::TensType,iB::intvecType,Z::TensType...;alpha::Number=eltype(A)(1),beta::Number=eltype(A)(1))
  mA,mB = checkType(A,B)

  iA = convIn(iA)
  iB = convIn(iB)

  if size(mA)[[iA...]] == size(mB)[[iB...]]
    println("contracting over indices with equal sizes")
  else
    error("some indices in A or B are not equal size; A->",size(mA)[[iA...]],", B->",size(mB)[[iB...]])
  end
  if typeof(mA) <: qarray
    println("checking flux left:")
    checkflux(mA)
    println("checking flux right:")
    checkflux(mB)
    for a = 1:length(iA)
      AQNs = recoverQNs(iA[a],A)
      BQNs = recoverQNs(iB[a],B)
      println("contracted index $a (A's index: ",iA[a],", B's index: ",iB[a],")")
      if length(AQNs) == length(BQNs)
        for w = 1:length(AQNs)
          if AQNs[w] != inv(BQNs[w])
            error("non-matching quantum numbers on indices on index ",iA[a]," of A and index ",iB[a]," of B: values ")
            println(A.QnumSum[iA[a]])
            println(B.QnumSum[iB[a]])
          end
        end
      else
        error("unmatching quantum number vector lengths")
      end
      println("matching quantum numbers on both indices")
      println("FULL PASS")
    end
  end
  nothing
end
export checkcontract
#end

