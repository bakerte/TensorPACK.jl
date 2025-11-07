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
    G = joinindex!(vec,A,B)

In-place joinindexenatation of tensors `A` (either `denstens` or Array type) and `B` along indices specified in `vec`
"""
function joinindex!(bareinds::intvecType,A::Union{tens{W},Array{W,N}},B::Union{tens{W},Array{W,N}}) where {W <: Number, N}
  inds = convIn(bareinds)

  nA = ndims(A)

  #=
  if typeof(A) <: denstens || typeof(B) <: denstens
    finalsize = Array{intType,1}(undef,nA)
    @inbounds @simd for w = 1:nA
      finalsize[w] = size(A,w)
    end
    for w = 1:length(inds)
      finalsize[inds[w]] += size(B,inds[w])
    end
  else
    =#
    #=
    finalsize = Array{intType,1}(undef,nA)#ntuple(w-> w in inds ? size(A,w) + size(B,w) : size(A,w),nA)
    for w = 1:nA
      finalsize[w] = w in inds ? size(A,w) + size(B,w) : size(A,w)
    end
    =#
#  end
  finalsize = intType[w in inds ? size(A,w) + size(B,w) : size(A,w) for w = 1:nA] #ntuple(w-> w in inds ? size(A,w) + size(B,w) : size(A,w),nA)

  Asize = size(A)
  Bsize = size(B)
  

  if length(inds) > 1
    if typeof(A) <: denstens || typeof(B) <: denstens
      Csize = 1
      for w = 1:nA
        Csize *= finalsize[w]
      end
      C = zeros(W,Csize)
    else
      C = zeros(W,finalsize...)
    end
  else
    if typeof(A) <: denstens || typeof(B) <: denstens
      Csize = 1
      for w = 1:nA
        Csize *= finalsize[w]
      end
      C = Array{W,1}(undef,Csize)
    else
      C = Array{W,nA}(undef,finalsize...)
    end
  end

  Aloop!(C,A,finalsize,Asize)
  Bloop!(C,B,finalsize,Bsize,inds,Asize)

  if typeof(A) <: denstens || typeof(B) <: denstens
    C = tens{W}(finalsize,C)
  end
  return C
end

"""
    Aloop!(C,A,finalsize,Asize)

Function boundary to load in first tensor in a direct sum into C (Array) from `denstens` or Array `A`; `finalsize` is the size of the resulting tensor as a vector or tuple; `Asize` is the size of `A` as a vector or tuple
"""
function Aloop!(C::Union{Array{W,G},Array{W,1}},A::Union{tens{W},Array{W,G}},finalsize::Union{NTuple{G,intType},Array{intType,1}},Asize::Union{Array{intType,1},NTuple{G,intType},Memory{intType}}) where {G, W <: Number}

  nA = ndims(A)

  pos = makepos(nA)
  d1 = 1
#  @inbounds while d1 < nA && Asize[d1] == 1
#    d1 += 1
#  end
  dimA = size(A,d1)

  Aysize = 1
  @inbounds @simd for w = d1+1:nA
    Aysize *= size(A,w)
  end

  p = 0
  for y = 1:Aysize
    position_incrementer!(pos,Asize)

    backZ = pos[nA]
    @inbounds @simd for w = nA-1:-1:d1
      backZ -= 1
      backZ *= finalsize[w]
      backZ += pos[w]
    end
    @inbounds @simd for x = 0:dimA-1
      p += 1
      C[x + backZ] = A[p]
    end
    pos[d1] += dimA
  end
  nothing
end

"""
    Bloop!(C,B,finalsize,inds,Asize)

Function boundary to load in second tensor in a direct sum into C (Array) from `denstens` or Array `B`; `finalsize` is the size of the resulting tensor as a vector or tuple; `inds` are the indices that are expanded; `Asize` is the size of `A` as a vector or tuple
"""
function Bloop!(C::Union{Array{W,R},Array{W,1}},B::Union{tens{W},Array{W,R}},finalsize::Union{NTuple{R,intType},Array{intType,1}},Bsize::Union{Array{intType,1},NTuple{R,intType},Memory{intType}},inds::NTuple{G,intType},Asize::Union{Array{intType,1},NTuple{R,intType},Memory{intType}}) where {R, G, W <: Number}



  nA = ndims(B)



  pos = makepos(nA) #makepos!(pos)
  Bpos = makepos(nA)

  d1 = 1
  dimB = size(B,d1)

  Bysize = 1
  @inbounds @simd for w = d1+1:nA
    Bysize *= size(B,w)
  end
  p = 0
  for y = 1:Bysize

    position_incrementer!(Bpos,Bsize)

    @inbounds @simd for w = 1:nA
      pos[w] = Bpos[w]
    end

    @inbounds @simd for w = 1:length(inds)
      pos[inds[w]] += Asize[inds[w]]
    end

    backZ = pos[nA]
    @inbounds @simd for w = nA-1:-1:d1
      backZ -= 1
      backZ *= finalsize[w]
      backZ += pos[w]
    end
    @inbounds @simd for x = 0:dimB-1
      p += 1
      C[x + backZ] = B[p]
    end
    Bpos[d1] += dimB
  end
  nothing
end


"""
    G = joinindex!(vec,A,B)

In-place joinindexenatation of tensors `A` (Array type) and `B` over all indices (most useful for rank-2 tensors as a matrix direct sum)
"""
function joinindex!(A::Array{S,N},B::Array{W,N}) where {W <: Number, S <: Number, N}
  return joinindex!(A,B,[i for i = 1:N])
end

"""
    G = joinindex!(vec,A,B)

In-place joinindexenatation of tensors `A` (`denstens`) and `B` over all indices (most useful for rank-2 tensors as a matrix direct sum)
"""
function joinindex!(A::tens{S},B::tens{W}) where {W <: Number, S <: Number}
  return joinindex!(A,B,[i for i = 1:ndims(A)])
end

"""
  G = joinindex(vec,A,B...)

Concatenatation of tensors `A` and any number of `B` along indices specified in `vec`
"""
function joinindex(inds::intvecType,A::W,B::R) where {W <: TensType, R <: TensType}
  if typeof(A) <: densTensType
    C = A
  else
    C = copy(A)
  end
  return joinindex!(inds,C,B)
end

"""
  G = joinindex(A,B,vec)

Concatenatation of tensors `A` and any number of `B` along indices specified in `vec` to output `G`
"""
function joinindex(A::W,B::R,inds::intvecType) where {W <: TensType, R <: TensType}
  mA,mB = checkType(A,B)
  return joinindex(inds,mA,mB)
end

"""
  G = joinindex!(A,B,vec)

In-place joinindexenatation of tensors `A` and any number of `B` along indices specified in `vec` to output `G`
"""
function joinindex!(A::W,B::R,inds::intvecType) where {W <: TensType, R <: TensType}
  mA,mB = checkType(A,B)
  return joinindex!(inds,mA,mB)
end

"""
  directsum(A,B...[,group=[1,2]])

Takes the direct sum of rank-2 tensors `A` and any number of tensors `B`. One can extend beyond matrices by modifying `group` which is the extra argument in `joinindex!`

See also: [`joinindex`](@ref) [`joinindex!`](@ref) [`directsum!`](@ref)
"""
function directsum(A::TensType,B::TensType...;group::Array{W,1}=[1,2],fct::Function=joinindex) where W <: Integer
  if length(B) > 0
    out = fct(group,A,B[1])
    for w = 2:length(B)
      out = joinindex!(group,out,B[w])
    end
  else
    out = A
  end
  return out
end

"""
  directsum!(A,B...[,group=[1,2]])

Takes the direct sum of rank-2 tensors `A` and any number of tensors `B`. One can extend beyond matrices by modifying `group` which is the extra argument in `joinindex!`. Modifies `A` in-place

See also: [`joinindex`](@ref) [`joinindex!`](@ref) [`directsum!`](@ref)
"""
function directsum!(A::TensType,B::TensType...;group::Array{W,1}=[1,2],fct::Function=joinindex!) where W <: Integer
  return directsum!(A,B...,group=group,fct=fct)
end


"""
  ⊕(A,B...)

Computes `A` ⊕ `B` (direct sum) of two input matrices (can extend `B` to any number of inputs)
"""
function ⊕(A::TensType,B::TensType...)
  return directsum(A,B...)
end







"""
    getblockrows(A,Aqind,leftAblock)

Obtains effective rows of an input `qarray` `A` for a given block `Aqind` and dimensions of the blocks `leftAblock`
"""
function getblockrows(A::qarray,Aqind::Integer,leftAblock::Array{P,1}) where P <: Integer
  Arows = Array{intType,1}(undef,size(A.ind[Aqind][1],2))
  if size(A.ind[Aqind][1],1) > 0
    @inbounds for i = 1:size(A.ind[Aqind][1],2)
      Arows[i] = A.ind[Aqind][1][1,i]
      @inbounds @simd for j = 2:size(A.ind[Aqind][1],1)
        Arows[i] *= leftAblock[j-1]
        Arows[i] += A.ind[Aqind][1][j,i]
      end
    end
  end
  return Arows
end





"""
    joinloop!(A,B,commonblocks,origAsize...)

Function boundary for `joinindex!` taking input `qarray`s `A` and `B` and quantum number blocks with the same quantum numbers `commonblocks` (array of size 2 tuples); `origAsize` is the base size of the input `A` taken from .QnumMat
"""
function joinloop!(A::Qtens{W,Q},B::Qtens{R,Q},commonblocks::Array{NTuple{2,intType},1},origAsize::intType...) where {W <: Number, R <: Number, Q <: Qnum}
  tup_Aleftsize = ntuple(w->A.currblock[1][w],length(A.currblock[1]))
  leftAblock = Array{intType,1}(undef,length(A.currblock[1])-1)
  @inbounds @simd for i = 1:length(leftAblock)
    leftAblock[i] = length(A.QnumMat[A.currblock[1][i]])
  end
  leftBblock = Array{intType,1}(undef,length(B.currblock[1])-1)
  @inbounds @simd for i = 1:length(leftBblock)
    leftBblock[i] = length(B.QnumMat[B.currblock[1][i]]) 
  end


  @inbounds for q = 1:length(commonblocks)
    Aqind = commonblocks[q][1]
    Bqind = commonblocks[q][2]


    Arows = getblockrows(A,Aqind,leftAblock)
    Brows = getblockrows(B,Bqind,leftBblock)

    if Arows != Brows 

      newrow = sort!(unique(vcat(Arows,Brows)))

      Anewrows = Array{intType,1}(undef,length(Arows))
      @inbounds for i = 1:length(Arows)
        Anewrows[i] = findfirst(w->newrow[w]==Arows[i],1:length(newrow))
      end
      Bnewrows = Array{intType,1}(undef,length(Brows))
      @inbounds for i = 1:length(Brows)
        Bnewrows[i] = findfirst(w->newrow[w]==Brows[i],1:length(newrow))
      end

      Ttype = typeof(W(1)*R(1))
      newcolsize = size(A.T[Aqind],2) + size(B.T[Bqind],2)

      newT = zeros(Ttype,length(newrow),newcolsize)

      newT[Anewrows,1:size(A.T[Aqind],2)] = A.T[Aqind]

      newT[Bnewrows,size(A.T[Aqind],2)+1:end] = B.T[Bqind]

      A.T[Aqind] = newT
      leftinds = ind2zeropos(newrow,tup_Aleftsize)

    else
      A.T[Aqind] = joinindex!([2],A.T[Aqind],B.T[Bqind])
      leftinds = A.ind[Aqind][1]
    end

    Ainds = A.ind[Aqind][2]
    Binds = B.ind[Bqind][2]

    Asize = size(Ainds,2)
    Bsize = size(Binds,2)

    newlength = Asize + Bsize
    newind = Array{intType,2}(undef,length(origAsize),newlength)
    for g = 1:Asize 
      @inbounds @simd for r = 1:length(origAsize)
        newind[r,g] = Ainds[r,g]
      end
    end
    for g = 1:Bsize
      modg = Asize + g
      @inbounds @simd for r = 1:length(origAsize)
        newind[r,modg] = Binds[r,g] + origAsize[r]
      end
    end

    A.ind[Aqind] = (leftinds,newind)
  end
  nothing
end

"""
    firstloop!(w,origAsize,A,index,newQnums,newQnumSum)

Function boundary for `joinindex!` taking input `qarray` `A` examining a given row or column in .ind `w`; `index` is the current index to get a quantum number; `newQnums` is the new quantum number after joining (.QnumMat); `newQnumSum` is the new summary (.QnumSum); `origAsize` is the base size of the input `A` taken from .QnumMat as either a vector or tuple
"""
function firstloop!(w::Integer,origAsize::Union{Array{P,1},NTuple{G,P}},A::Qtens{W,Q},index::Integer,newQnums::Array{intType,1},newQnumSum::Array{Q,1}) where {W <: Number, Q <: Qnum, P <: Integer, G}
  @inbounds for j = 1:origAsize[w] #better serial on small systems (for sequential memory access?)
    thisQN = getQnum(index,j,A)
    notmatchQN = true
    b = 0
    @inbounds while b < length(newQnumSum) && notmatchQN
      b += 1
      notmatchQN = thisQN != newQnumSum[b]
    end
    newQnums[j] = b
  end
  nothing
end

"""
    matchloop(g,B,index,deltaflux,newQnums,newQnumSum)

Function boundary that takes original size `g` of input tensor `A` in `joinindex!` and adds to sizes in second tensor `B`; `index` is the current base index (.QnumMat) that is being investigated; `deltaflux` is the change in flux between the two tensors to ensure matching; `newQnums` is the new quantum number after joining (.QnumMat); `newQnumSum` is the new summary (.QnumSum); `origAsize` is the base size of the input `A` taken from .QnumMat as either a vector or tuple
"""
function matchloop(g::Integer,B::Qtens{W,Q},index::Integer,deltaflux::Q,newQnums::Array{intType,1},newQnumSum::Array{Q,1}) where {W <: Number, Q <: Qnum}
  @inbounds for j = 1:size(B,index)
    g += 1
    thisQN = getQnum(index,j,B) + deltaflux
    notmatchQN = true
    b = 0
    @inbounds while b < length(newQnumSum) && notmatchQN
      b += 1
      notmatchQN = thisQN != newQnumSum[b]
    end
    newQnums[g] = b
  end
  nothing
end

"""
    matchloop(g,B,index,deltaflux,newQnums,newQnumSum)

Function boundary that adds in any non-matching blocks between `A` and `B` in a `joinindex!` call; `inds` is the indices to be expanded; `A` and `B` are `qarray`s; `Bleftover` is the leftover quantum numbers that were not already accounted for; `newT` is the resulting tensor that `A` and `B` are joined into; `newindexlist` (array or tuple) is a list of indices (.ind field) that will be needed for the new block; `inputsize` is a tuple that each index has and therefore must be added to each appropriate index; `newQblocksum` is the new summary of quantum numbers on each block
"""
function Bextraloop!(inds::Array{intType,1},A::Qtens{W,Q},B::Qtens{R,Q},Bleftover::Array{intType,1},
                      newT::Array{Array{P,2},1},newindexlist::Array{NTuple{2,Array{intType,2}},1},
                      inputsize::Tuple,newQblocksum::Array{NTuple{2,Q},1}) where {W <: Number, R <: Number, Q <: Qnum, P <: Number}
  @inbounds for q = 1:length(Bleftover)
    addq = Bleftover[q]
    thisind = q + length(A.T)
    newT[thisind] = B.T[addq]
    newindexlist[thisind] = B.ind[addq]
    for i = 1:2
      @inbounds for a = 1:length(A.currblock[i])
        index = A.currblock[i][a]
        if index in inds
          @inbounds @simd for j = 1:size(newindexlist[thisind][i],2)
            newindexlist[thisind][i][a,j] += inputsize[index]
          end
        end
      end
    end
    newQblocksum[thisind] = B.Qblocksum[addq]
  end
  nothing
end

"""
    makerowcol(Lposes,Lsizes,A,q,LR)

Generates rows of the effective matrix-equivalent based on the .ind field of an input `qarray` `A`; `Lposes` is a vector of integers that is used to store the integers; `Lsizes` is a tuple of sizes; `q` is an integer corresponding to a particular block of `A`; `LR` is either the row or column of the block that is required
"""
function makerowcol(Lposes::Array{P,1},Lsizes::Tuple,A::Qtens{W,Q},q::Integer,LR::Integer) where {P <: Integer, W <: Number, Q <: Qnum}
  @inbounds rows = Array{intType,1}(undef,size(A.ind[q][LR],2))
  for x = 1:length(rows)
    @inbounds @simd for i = 1:size(A.ind[q][LR],1)
      Lposes[i] = A.ind[q][LR][i,x]
    end
    pos2ind!(rows,x,Lposes,Lsizes)
  end
  return rows
end

"""
    rowcolstart(rows,A,LR)

Useful for reordering a joined tensor by checking to see if the return tensor is ordered for its rows and columns (better for efficiency later); takes input `qarray` `A` and a vector containing the rows `rows` computing either the left (`LR` = 1) or right (`LR` = 2) of the matrix-equivalent of the blocks

If `rows` is sorted, nothing happens, else it re-sorts them
"""
function rowcolsort(rows::Array{P,1},A::Qtens{W,Q},LR::Integer) where {P <: Integer, W <: Number, Q <: Qnum}
  rowsort = issorted(rows)
  newroworder = rowsort ? sortperm(rows) : intType[0] #1:size(A.ind[q][LR],2)
  return rowsort,newroworder
end

"""
    orderloop!(A,Lsizes,Rsizes,Lposes,Rposes)

loads the existing .ind fields from `qarray` `A` into a new tensor (in-place) according to the sizes of the of the new tensor blocks given by `Lsizes` x `Rsizes`; inputs `Lposes` and `Rposes` are a vector of integers providing a rule to map integers from one .ind field to another
"""
function orderloop!(A::Qtens{W,Q},Lsizes::Tuple,Rsizes::Tuple,Lposes::Array{P,1},Rposes::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}
  @inbounds for q = 1:length(A.ind)
    rows = makerowcol(Lposes,Lsizes,A,q,1)
    cols = makerowcol(Rposes,Rsizes,A,q,2)

    rowsort,newroworder = rowcolsort(rows,A,1)
    colsort,newcolorder = rowcolsort(cols,A,2)
    if rowsort && colsort
      A.T[q] = A.T[q][newroworder,newcolorder]
      loadM!(A.ind[q][1],A.ind[q][1][:,newroworder])
      loadM!(A.ind[q][2],A.ind[q][2][:,newcolorder])
    elseif rowsort
      A.T[q] = A.T[q][newroworder,:]
      loadM!(A.ind[q][1],A.ind[q][1][:,newroworder])
    else
      A.T[q] = A.T[q][:,newcolorder]
      loadM!(A.ind[q][2],A.ind[q][2][:,newcolorder])
    end
  end
  nothing
end

"""
    joinindex(inds,A,B[,ordered=false])

performs a direct sum for `qarray`s `A` and `B` expanding indices `inds`; `ordered` will order the blocks inside of the tensor if required
"""
function joinindex(bareinds::intvecType,QtensA::Qtens{R,Q},QtensB::Qtens{S,Q};ordered::Bool=false) where {R <: Number, S <: Number, Q <: Qnum}
  return joinindex!(bareinds,copy(QtensA),QtensB,ordered=ordered)
end

"""
    joinindex!(inds,A,B[,ordered=false])

performs a direct sum for `qarray`s `A` and `B` expanding indices `inds`; `ordered` will order the blocks inside of the tensor if required
"""
function joinindex!(bareinds::intvecType,QtensA::Qtens{R,Q},QtensB::Qtens{S,Q};ordered::Bool=false) where {R <: Number, S <: Number, Q <: Qnum}
  preinds = convIn(bareinds)
  inds = Array{intType,1}(undef,length(preinds))
  @inbounds @simd for i = 1:length(inds)
    inds[i] = preinds[i]
  end

  inputsize = size(QtensA)

  notcommoninds = findnotcons(length(QtensA.QnumMat),preinds)

  A = changeblock(QtensA,notcommoninds,inds)
  B = changeblock(QtensB,notcommoninds,inds)


  origAsize = Array{intType,1}(undef,length(inds))
  @inbounds @simd for w = 1:length(inds)
    origAsize[w] = length(A.QnumMat[inds[w]])
  end

  commonblocks = matchblocks((false,false),A,B,ind=(2,1),matchQN=A.flux)

  Bleftover = findextrablocks(B,commonblocks)

  joinloop!(A,B,commonblocks,origAsize...)



  Ttype = typeof(eltype(QtensA)(0)*eltype(QtensB)(0))
  newT = Array{Array{Ttype,2},1}(undef,length(A.T)+length(Bleftover))
  newindexlist = Array{NTuple{2,Array{intType,2}},1}(undef,length(newT))
  newQblocksum = Array{NTuple{2,Q},1}(undef,length(newT))


  newT[1:length(A.T)] = A.T
  newindexlist[1:length(A.T)] = A.ind
  newQblocksum[1:length(A.T)] = A.Qblocksum

  Bextraloop!(inds,A,B,Bleftover,newT,newindexlist,inputsize,newQblocksum)

  A.T = newT
  A.ind = newindexlist
  A.Qblocksum = newQblocksum

  zeroQN = Q()

  deltaflux = A.flux - B.flux
  @inbounds for w = 1:length(inds)
    index = inds[w]

    thisvec = Array{Q,1}(undef,length(A.QnumSum[index])+length(B.QnumSum[index]))
    @inbounds @simd for w = 1:length(A.QnumSum[index])
      thisvec[w] = A.QnumSum[index][w]
    end
    counter = length(A.QnumSum[index])
    if deltaflux == zeroQN || w > 1
      @inbounds @simd for w = 1:length(B.QnumSum[index])
        counter += 1
        thisvec[counter] = B.QnumSum[index][w] + deltaflux
      end
    else
      @inbounds @simd for w = 1:length(B.QnumSum[index])
        counter += 1
        thisvec[counter] = B.QnumSum[index][w]
      end
    end
    newQnumSum = unique!(thisvec)

    newQnums = Array{intType,1}(undef,origAsize[w] + size(B,index))
    firstloop!(w,origAsize,A,index,newQnums,newQnumSum)
    
    g = origAsize[w]
    matchloop(g,B,index,deltaflux,newQnums,newQnumSum)

    A.QnumMat[index] = newQnums
    A.QnumSum[index] = newQnumSum
  end

  return A
end


