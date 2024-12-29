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
    C = libmult([alpha,]C,D[,beta,Z])

Chooses the best matrix multiply function for tensor contraction (dense tensor or sub-block) withoutput matrix `C`

+ Outputs `C` * `D` if `alpha`=1
+ Outputs `alpha` * `C` * `D` if `Z` is not input
+ Outputs `alpha` * `C` * `D` + `beta` * `Z` if `Z` is input
"""
function libmult! end

"""
    libmult(tA, tB, alpha, A, B)

Return `alpha*A*B` or the other three variants according to [`tA`](@ref stdlib-blas-trans) and `tB`.
"""
libmult(tA, tB, alpha, A, B)

"""
    libmult(tA, tB, alpha, A, B, beta, C)

Return `alpha*A*B+beta*C` or the other three variants according to [`tA`](@ref stdlib-blas-trans) and `tB`.
"""
libmult(tA, tB, alpha, A, B,beta,C)

"""
    libmult(tA, tB, A, B)

Return `A*B` or the other three variants according to [`tA`](@ref stdlib-blas-trans) and `tB`.
"""
libmult(tA, tB, A, B)

"""
    permL,permR = permq(A,iA)

Answers the question of whether to permute the dense tensor `A` with contracted indices `iA` with a boolean value; `permL` will permute `A`'s left indices while `permR` will permute the right-indices
"""
function permq(A::densTensType,iA::Union{Array{intType,1},NTuple{K,intType}}) where K
  nopermL = true
  w = 0
  while nopermL && w < length(iA)
    w += 1
    nopermL = iA[w] == w
  end
  nopermR = !nopermL
  if nopermR
    w = length(iA)
    end_dim = ndims(A)
    while nopermR && w > 0
      nopermR = iA[w] == end_dim
      end_dim -= 1
      w -= 1
    end
  end
  return nopermL,nopermR
end

"""
    Aperm,transA = willperm(conjA,W,AnopermL,AnopermR)

Determines LAPACK flag (`transA`) based on output of `permq` and whether to conjugate `conjA` and what type `W` and whether to permute `Aperm`
"""
function willperm(conjA::Bool,W::DataType,AnopermL::Bool,AnopermR::Bool)
  if AnopermL
    transA = conjA && W <: Complex ? 'C' : 'T'
    Aperm = true
  else
    transA = 'N'
    Aperm = AnopermR
  end
  return Aperm,transA
end

"""
    pA = prepareT(A,Lvec,Rvec,conjvar)

Converts input tensor `A` to its matrix equivalent `pA` with left indices contained in `Lvec` and right indices contained in `Rvec` and whether to conjugate (`conjvar`)
"""
function prepareT(A::densTensType,Lvec::Union{Array{intType,1},NTuple{K,intType}},Rvec::Union{Array{intType,1},NTuple{P,intType}},conjvar::Bool) where {K,P}

  newdimsA = Memory{intType}(undef,ndims(A))
  counter = 0
  @inbounds @simd for w = 1:length(Lvec)
    counter += 1
    newdimsA[counter] = Lvec[w]
  end
  @inbounds @simd for w = 1:length(Rvec)
    counter += 1
    newdimsA[counter] = Rvec[w]
  end
  pA = permutedims(A,newdimsA)
  if conjvar && eltype(A) <: Complex
    conj!(pA)
  end
  return pA
end

"""
    consize,notconsize = getsizes(A,iA)

Finds sizes of matrix equivvalent of `A` based on contracted indices `iA` (`consize`) and not contracted indices size (`notconsize`)
"""
function getsizes(A::TensType,iA::intvecType)
  Lsize = innersizeL = 1
  @inbounds for w = 1:ndims(A)
    if !(w in iA)
      Lsize *= size(A,w)
    else
      innersizeL *= size(A,w)
    end
  end
  return Lsize,innersizeL
end

"""
    a = makesize(w,A,iA,B,iB)

based on index `w` of resulting tensor `C`, will find output size for use in tuples based on input `TensTypes` `A` and `B` and also their contractd indices `iA` and `iB`
"""
function makesize(w,A,iA,B,iB)
  counter = 0
  a = 0
  if w <= ndims(A)-length(iA)
    while counter < w
      a += 1
      notinset = true
      p = 0
      while notinset && p < length(iA)
        p += 1
        notinset = a != iA[p]
      end
      if notinset
        counter += 1
      end
    end
    return size(A,a)
  else
    while counter < w-ndims(A)+length(iA)
      a += 1
      notinset = true
      p = 0
      while notinset && p < length(iB)
        p += 1
        notinset = a != iB[p]
      end
      if notinset
        counter += 1
      end
    end
    return size(B,a)
  end
end

"""
    G = findnotcons(nA,iA)

Generates the complement set of an input `iA` (`G`) for a total number of elements `nA`.  Used for contractions and other functions.
"""
function findnotcons(nA::Integer,iA::NTuple{N,intType}) where N
  notconvec = Array{intType,1}(undef,nA-length(iA))
  counter = 0
  @inbounds for w = 1:nA
    k = 0
    notmatchinginds = true
    @inbounds while k < length(iA) && notmatchinginds
      k += 1
      notmatchinginds = iA[k] != w
    end
    if notmatchinginds
      counter += 1
      notconvec[counter] = w
    end
  end
  return notconvec
end

"""
    C = maincontractor!(conjA,conjB,A,iA,B,iB,Z...;alpha=1,beta=1)

Contraction function (`alpha`*`A`*`B`+`beta`*`Z`) and can conjugate A (`conjA`) or B (`conjB`) for input `densTensType`s. Does not construct return tensor, just outputs a vector.

See also: [`maincontractor`](@ref)
"""
function maincontractor!(conjA::Bool,conjB::Bool,A::densTensType,iA::intvecType,B::densTensType,iB::intvecType,Z::densTensType...;alpha::Number=1,beta::Number=1)
  
  if ndims(A) < maximum(iA)
    error("too many indices input ",size(A)," for ",iA)
  end

  if ndims(B) < maximum(iB)
    error("too many indices input ",size(B)," for ",iB)
  end

  if prod(w->size(A,iA[w]),1:length(iA)) != prod(w->size(B,iB[w]),1:length(iB))
    error("not matching sizes for input tensors to contraction [A size ",size(A)," on indices ",iA," and B size ",size(B)," on ",iB,"]")
  end
  
  AnopermL,AnopermR = permq(A,iA)
  BnopermL,BnopermR = permq(B,iB)

  Aperm,transA = willperm(conjA,eltype(A),AnopermL,AnopermR)
  Bperm,transB = willperm(conjB,eltype(B),BnopermR,BnopermL)

  Aremain = ndims(A)-length(iA)
  Bremain = ndims(B)-length(iB)

  Lsize,innersizeL = getsizes(A,iA)
  Rsize,innersizeR = getsizes(B,iB)

  if Aremain == 0 && Bremain == 0
    AAsizes = nullsize
  else
    AAsizes = Array{intType,1}(undef,Aremain+Bremain)
    for w = 1:Aremain+Bremain
      AAsizes[w] = makesize(w,A,iA,B,iB)
    end
#    intType[makesize(w,A,iA,B,iB) for w = 1:Aremain+Bremain] #ntuple(w->makesize(w,A,iA,B,iB),Aremain+Bremain)
  end
  
  if Aperm && Bperm
    mulA = conjA && eltype(A) <: Complex && !AnopermL ? conj(A) : A
    mulB = conjB && eltype(B) <: Complex && !BnopermR ? conj(B) : B
  elseif Aperm
    mulA = conjA && eltype(A) <: Complex && !AnopermL ? conj(A) : A
    notvecB = findnotcons(ndims(B),iB)
    mulB = prepareT(B,iB,notvecB,conjB)
  elseif Bperm
    notvecA = findnotcons(ndims(A),iA)
    mulA = prepareT(A,notvecA,iA,conjA)
    mulB = conjB && eltype(B) <: Complex && !BnopermR ? conj(B) : B
  else
    notvecA = findnotcons(ndims(A),iA)
    mulA = prepareT(A,notvecA,iA,conjA)
    notvecB = findnotcons(ndims(B),iB)
    mulB = prepareT(B,iB,notvecB,conjB)
  end

  if length(Z) > 0
    outType = typeof(eltype(A)(1)*eltype(B)(1)*eltype(Z[1])(1)*typeof(alpha)(1)*typeof(beta)(1))
    type_alpha = convert(outType,alpha)
    type_beta = convert(outType,beta)
    out = libmult(transA,transB,type_alpha,mulA,mulB,type_beta,Z[1],Lsize,innersizeL,innersizeR,Rsize)
  elseif abs(alpha-1) < 1E-8 #isapprox(alpha,1)

    out = libmult(transA,transB,mulA,mulB,Lsize,innersizeL,innersizeR,Rsize)
  else
    outType = typeof(eltype(A)(1)*eltype(B)(1)*typeof(alpha)(1))
    if outType != typeof(alpha)
      type_alpha = convert(outType,alpha)
    else
      type_alpha = alpha
    end
    out = libmult(transA,transB,type_alpha,mulA,mulB,Lsize,innersizeL,innersizeR,Rsize)
  end
  return AAsizes,out
end

"""
    C = maincontractor(conjA,conjB,A,iA,B,iB,Z...;alpha=1,beta=1)

Contraction function (`alpha`*`A`*`B`+`beta`*`Z`) and can conjugate A (`conjA`) or B (`conjB`) for input `densTensType`s. This outputs a tensor `C`

See also: [`maincontractor!`](@ref)
"""
function maincontractor(conjA::Bool,conjB::Bool,A::densTensType,iA::intvecType,B::densTensType,iB::intvecType,Z::TensType...;alpha::Number=1,beta::Number=1)
  AAsizes,out = maincontractor!(conjA,conjB,A,iA,B,iB,Z...,alpha=alpha,beta=beta)
  if typeof(out) <: Number
    outTens = out
  else
    if typeof(A) <: denstens || typeof(B) <: denstens
      outTens = tens(AAsizes,out)
    else
      outTens = reshape!(out,AAsizes)
    end
  end
  return outTens
end


#       +------------------------+
#>------| Quantum number version |---------<
#       +------------------------+



"""
    c = getinds_loop_one!(con,Rsize,vec)

populates `con` with sizes of groups indices; used for `qarray` contraction to find sizes of the contracted group of indices

See also: [`getinds`](@ref)
"""
function getinds_loop_one!(con::Array{intType,1},Rsize::Array{Array{intType,1},1},vec::Union{NTuple{N,intType},Array{intType,1}}) where N
  counter = 0
  for j = 1:size(vec, 1)
    @inbounds @simd for p in Rsize[vec[j]]
      counter += 1
      con[counter] = p
    end
  end
  nothing
#  return counter  #; outputs a counter (current unused) for checking purposes
end

"""
    getinds_loop_two!(notcon,Rsize,vec)

populates `notcon` with sizes of groups indices; used for `qarray` contraction to find sizes of the non-contracted group

See also: [`getinds`](@ref)
"""
function getinds_loop_two!(notcon::Array{intType,1},Rsize::Array{Array{intType,1},1},vec::Union{NTuple{N,intType},Array{intType,1}}) where N
  altcounter = 0
  for j = 1:length(Rsize)
    condition = true
    k = 0
    @inbounds while k < size(vec,1) && condition
      k += 1
      condition = !(j == vec[k])
    end
    if condition
      @inbounds @simd for p in Rsize[j]
        altcounter += 1
        notcon[altcounter] = p
      end
    end
  end
  nothing
end

"""
    con,notcon = getinds(A,iA)

Sub-function for quantum number contraction.  A Qtensor `A` with indices to contract `iA` generates all contracted indices (if, for example, a joined index was called by a single index number) as `con`, and also the un-contracted indices (`notcon`)
"""
function getinds(currQtens::qarray, vec::Union{Array{intType,1},NTuple{N,intType}}) where N
  Rsize = currQtens.size
  consize = 0
  @inbounds @simd for w = 1:length(vec)
    consize += length(Rsize[w])
  end
  con = Array{intType,1}(undef,consize)  

  getinds_loop_one!(con,Rsize,vec)
  
  notcon = Array{intType,1}(undef,length(currQtens.QnumMat)-consize)

  getinds_loop_two!(notcon,Rsize,vec)

  return con, notcon
end

"""
    mulblockA,mulblockB = checkblocks(Aqind,Bqind,A,B,inputA,inputB,Aind=2,Bind=1)

helper function for `qarray` contraction; checks that blocks `Aqind` of `qarray` `A` and `Bqind` of `qarray` `B` have the same size (sometimes blocks are stored without matching sizes because of a row of zeros); this function truncates blocks that do not match after checking the quantum numbers on each index (defined by indices `Aind` and `Bind`) and outputs new blocks `mulblockA` and `mulblockB`
"""
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

"""
    genblockinds!(offset,firstblock)

iterates an inpute vector `firstblock` by a counter (1 to the length of the input vector) plus the `offset`
"""
function genblockinds!(offset::intType,firstblock::Array{intType,1})
  @inbounds @simd for w = 1:length(firstblock)
    firstblock[w] = offset + w
  end
  nothing
end

"""
    loadnewsize!(newsize)

corrects size field input `newsize` to be in order after contraction and count to number of base dimensions of the `qarray`
"""
function loadnewsize!(newsize::Array{Array{intType,1},1})
  counter = 0
  for w = 1:length(newsize)
    @inbounds @simd for a = 1:length(newsize[w])
      counter += 1
      newsize[w][a] = counter
    end
  end
  nothing
end

"""
    loadarraynewsize!(newsize,offset,notconA,QtensA)

corrects array size field (`.ind`) input `newsize` to be in order after contraction and count to number of base dimensions of the `qarray`

See also: [`Qtens`](@ref)
"""
function loadarraynewsize!(newsize::Array{Array{intType,1},1},offset::intType,notconA::Array{intType,1},QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds for w = 1:length(notconA)
    newsize[w+offset] = Array{intType,1}(undef,length(QtensA.size[notconA[w]]))
  end
  nothing
end

"""
    loadnewQnumMat!(newQnumMat,offset,notconA,A)

fills resulting `QnumMat` as a result of contraction

See also: [`Qtens`](@ref)
"""
function loadnewQnumMat!(newQnumMat::Array{Array{intType,1},1},offset::Integer,notconA::Array{intType,1},A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds @simd for q = 1:length(notconA)
    newQnumMat[q+offset] = A.QnumMat[notconA[q]]
  end
  nothing
end

"""
    loadnewQnumSum!(newQnumMat,offset,notconA,A)

fills resulting `QnumSum` as a result of contraction

See also: [`Qtens`](@ref)
"""
function loadnewQnumSum!(newQnumMat::Array{Array{Q,1},1},offset::Integer,notconA::Array{intType,1},A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds @simd for q = 1:length(notconA)
    newQnumMat[q+offset] = A.QnumSum[notconA[q]]
  end
  nothing
end

"""
    loadnewQnumSum_inv!(newQnumMat,offset,notconA,A)

fills resulting inverse of the `QnumSum` as a result of contraction (different index)

See also: [`Qtens`](@ref)
"""
function loadnewQnumSum_inv!(newQnumMat::Array{Array{Q,1},1},offset::Integer,notconA::Array{intType,1},A::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  @inbounds @simd for q = 1:length(notconA)
    newQnumMat[q+offset] = inv.(A.QnumSum[notconA[q]])
  end
  nothing
end

"""
    nopermL,nopermR = permq(A,iA)

Determines whether to permute a `qarray` `A` based on contraction indices `iA` and whether the first (`nopermL`) or last (`nopermR`) group of indices should be permuted
"""
function permq(A::Qtens{W,Q},iA::Array{intType,1}) where {W <: Number, Q <: Qnum} #K
  nopermL = length(iA) == length(A.currblock[1]) && issorted(A.currblock[1])
  w = 0
  @inbounds while nopermL && w < length(iA)
    w += 1
    nopermL = w == A.currblock[1][w] && w == iA[w]
  end

  nopermR = !nopermL && length(iA) == length(A.currblock[2]) && issorted(A.currblock[2])

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

"""
    C = maincontractor(conjA,conjB,A,iA,B,iB,Z...;alpha=1,beta=1)

Contraction function (`alpha`*`A`*`B`+`beta`*`Z`) and can conjugate A (`conjA`) or B (`conjB`) for input `qarray`s. This outputs a tensor `C`

See also: [`maincontractor!`](@ref)
"""
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

  usealpha = abs(alpha-1) > 1E-8 #!isapprox(alpha,1)

  numQNs = length(commonblocks)

  useZ = length(Z) > 0
  if useZ
    Zone = [i for i = 1:length(notconA)]
    Ztwo = [i + length(notconA) for i = 1:length(notconB)]
    Z = changeblock(Zed,Zone,Ztwo)
    Zcommonblocks = matchblocks((conjA,false),A,Zed,ind=(2,1))
    type_beta = eltype(beta) == outType && #=!isapprox(beta,1)=# abs(alpha-1) > 1E-8  ? beta : convert(outType,beta)
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
  loadnewQnumMat!(newQnumMat,0,notconA,A)
  loadnewQnumMat!(newQnumMat,length(notconA),notconB,B)

  newQnumSum = Array{Array{Q,1},1}(undef,length(notconA)+length(notconB))
  if conjA
    loadnewQnumSum_inv!(newQnumSum,0,notconA,A)
  else
    loadnewQnumSum!(newQnumSum,0,notconA,A)
  end
  if conjB
    loadnewQnumSum_inv!(newQnumSum,length(notconA),notconB,B)
  else
    loadnewQnumSum!(newQnumSum,length(notconA),notconB,B)
  end

  newsize = Array{Array{intType,1},1}(undef,length(notconA)+length(notconB))
  loadarraynewsize!(newsize,0,notconA,QtensA)
  loadarraynewsize!(newsize,length(notconA),notconB,QtensB)

  loadnewsize!(newsize)

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
  genblockinds!(0,firstblock)

  secondblock = Array{intType,1}(undef,length(notconB))
  genblockinds!(length(notconA),secondblock)
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



