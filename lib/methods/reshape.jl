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
  G = reshape!(M,S[,merge=])

In-place reshape for `denstens` from new size `S`--a tuple--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape`](@ref)
"""
function reshape!(M::tens{W}, S::NTuple{N,intType};merge::Bool=false) where {N, W <: Number}
  if ndims(M) != length(S)
    M.size = intType[S[w] for w = 1:length(S)]
  else
    @inbounds @simd for w = 1:length(S)
      M.size[w] = S[w]
    end
  end
  return M
end

"""
  G = reshape!(M,S...[,merge=])

In-place reshape for `denstens` from new size `S`--a list of numbers--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape`](@ref)
"""
function reshape!(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
  return reshape!(M,S)
end

"""
  G = reshape!(M,S[,merge=])

In-place reshape for `denstens` from new size `S`--an array--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape`](@ref)
"""
function reshape!(M::tens{W}, S::Array{intType,1};merge::Bool=false) where W <: Number
  return reshape(M,S...)
end

"""
  G = reshape!(M,S[,merge=])

In-place reshape for `Array` from new size `S`--a tuple--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape`](@ref)
"""
function reshape!(M::Array{W,P}, S::NTuple{N,intType};merge::Bool=false) where {N,P, W <: Number}
  return reshape!(M,S...)
end

"""
  G = reshape!(M,S...[,merge=])

In-place reshape for `Array` from new size `S`--a list of numbers--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape`](@ref)
"""
function reshape!(M::Array{W,P}, S::intType...;merge::Bool=false) where {P,W <: Number}
  return reshape(M,S...)
end

"""
  G = reshape!(M,S[,merge=])

In-place reshape for `Array` from new size `S`--an array--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape`](@ref)
"""
function reshape!(M::Array{W,P}, S::Array{intType,1};merge::Bool=false) where {P,W <: Number}
  return reshape!(M,S...)
end

"""
  G = reshape!(M,a[,merge=])

Similar to `reshape!` with an input tuple for new sizes of the tensor `G`, but can also reshape indices explicitly by specifying them in a vector of vectors.

# Example:

```julia
julia> A = rand(20,40,30);
julia> reshape!(A,800,30); #reshapes first indices together
julia> reshape!(A,[[1,2],[3]]); #same as above
```

See also: [`reshape`](@ref)
"""
function reshape!(M::tens{W}, S::Union{Array{Array{P,1},1},Tuple};merge::Bool=false) where {W <: Number, P <: Integer}
  newsize = ntuple(a->prod(b->size(M,b),S[a]),length(S))
  order = vcat(S...)
  pM = issorted(order) ? M : permutedims!(M,order)
  return reshape!(pM,newsize)
end

"""
  G = reshape(M,S[,merge=])

Reshape for `denstens` from new size `S`--a tuple--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape`](@ref)
"""
function reshape(M::tens{W}, S::NTuple{N,intType};merge::Bool=false) where {N,W <: Number}
  return reshape!(copy(M),S)
end

"""
  G = reshape(M,S...[,merge=])

Reshape for `denstens` from new size `S`--a list of numbers--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape`](@ref)
"""
function reshape(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
  return reshape!(copy(M),S)
end

"""
  G = reshape(M,S[,merge=])

Reshape for `denstens` from new size `S`--an array--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape`](@ref)
"""
function reshape(M::tens{W}, S::Array{intType,1};merge::Bool=false) where W <: Number
  return reshape(copy(M),S...)
end

"""
  G = reshape(M,S[,merge=])

Reshape for `denstens` from new size `S`--an array--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape!`](@ref)
"""
function reshape(M::Memory{W}, S::Memory{intType};merge::Bool=false) where W <: Number
  return reshape(convert(Array{W,1},M),S...)
end

"""
  G = reshape!(M,S[,merge=])

Reshape for `denstens` from new size `S`--an array--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape`](@ref)
"""
function reshape!(M::Memory{W}, S::intType...;merge::Bool=false) where W <: Number
  return reshape(convert(Array{W,1},M),S...)
end

"""
  G = reshape(M,S[,merge=])

Reshape for `Array` from new size `S`--an array--with output `G`

Note: `merge` is not useful for this input but is useful to have for flexibility of code for using quantum number symmetries

See also: [`reshape`](@ref)
"""
function reshape(M::Array{W,N}, S::Array{intType,1};merge::Bool=false) where {W <: Number, N}
  return reshape(copy(M),S...)
end

"""
  G = reshape(M,a[,merge=])

Similar to `reshape` with an input tuple for new sizes of the tensor `G`, but can also reshape indices explicitly by specifying them in a vector of vectors.

# Example:

```julia
julia> A = rand(20,40,30);
julia> reshape!(A,800,30); #reshapes first indices together
julia> reshape!(A,[[1,2],[3]]); #same as above
```

See also: [`reshape!`](@ref)
"""
function reshape(M::tens{W}, S::Union{Array{Array{P,1},1},Tuple};merge::Bool=false) where {W <: Number, P <: intType}
  return reshape!(copy(M),S)
end

"""
  G = unreshape!(M,S[,merge=false])

Same as `reshape!` for `Array` but used for ease of reading code and also has new context with quantum numbers

See also: [`reshape!`](@ref)
"""
function unreshape!(M::AbstractArray, S::intType...;merge::Bool=false)
  return reshape(M,S...)
end

"""
  G = unreshape!(M,S[,merge=false])

Same as `reshape!` for `denstens` but used for ease of reading code and also has new context with quantum numbers

See also: [`reshape!`](@ref)
"""
function unreshape!(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
  return reshape!(M,S)
end

"""
  G = unreshape(M,S)

Same as `reshape` for `Array` but used for ease of reading code and also has new context with quantum numbers

See also: [`reshape`](@ref)
"""
function unreshape(M::AbstractArray, S::intType...;merge::Bool=false)
  return reshape(M,S...)
end

"""
  G = unreshape(M,S)

Same as `reshape` for `denstens` but used for ease of reading code and also has new context with quantum numbers

See also: [`reshape`](@ref)
"""
function unreshape(M::tens{W}, S::intType...;merge::Bool=false) where W <: Number
  newM = tens{W}(M.size,M.T)
  return unreshape!(newM,S...)
end




"""
   reshape!(A,S...)

In-place reshape of input `dtens` with integers `S`
"""
function reshape!(A::dtens,a::Integer...)
  A[0] = reshape!(A[0],a...)
  A[1] = reshape!(A[1],a...)
  return A
end

"""
   reshape!(A,S)

In-place reshape of input `dtens` with vector `S`
"""
function reshape!(A::dtens,order::Array)
  A[0] = reshape!(A[0],order)
  A[1] = reshape!(A[1],order)
  return A
end

"""
   reshape!(A,S)

In-place reshape of input `dtens` with tuple `S`
"""
function reshape!(A::dtens,order::Tuple)
  A[0] = reshape!(A[0],order)
  A[1] = reshape!(A[1],order)
  return A
end

"""
   reshape(A,S)

Reshape of input `dtens` with Array `S`
"""
function reshape(A::dtens,order::Array)
  A[0] = reshape!(A[0],order)
  A[1] = reshape!(copy(A[1]),order)
  return A
end

"""
   reshape(A,S)

Reshape of input `dtens` with tuple `S`
"""
function reshape(A::dtens,order::Tuple)
  A[0] = reshape!(A[0],order)
  A[1] = reshape!(copy(A[1]),order)
  return A
end









"""
    reshape!(M,a...[,merge=false])

In-place reshape for Qtensors for a list of numbers `a`; can also make Qtensor unreshapable with `merge = true`, joining all grouped indices together

# Warning
If the Qtensor size is (10,1,2) and we want to reshape in to a (10,2) tensor, this algorithm will reshape to a 
[[1,2],[3]] be default instead of [[1],[2,3]], so beware.

See also: [`reshape`](@ref)
"""
function reshape!(Qt::Qtens{W,Q}, S::Integer...;merge::Bool=false) where {W <: Number, Q <: Qnum}
  Rsize = recoverShape(Qt,S...)
  outQt = reshape!(Qt,Rsize,merge=merge)
  newindexsizeone!(outQt,S...)
  return outQt
end

"""
    reshape!(M,vec[,merge=false])

In-place reshape for Qtensors for a group of indices `vec` (ex: [[1,2],[3,4],[5]]); can also make Qtensor unreshapable with `merge = true`, joining all grouped indices together

See also: [`reshape`](@ref)
"""
function reshape!(Qt::Qtens{W,Q}, newQsize::Union{Array{Array{P,1},1},Tuple};merge::Bool=false) where {W <: Number, Q <: Qnum, P <: intType}
  order = vcat(newQsize...)

  if !issorted(order)
    permutedims!(Qt,order)
  end

  Qt.size = [[newQsize[q][w] for w = 1:length(newQsize[q])] for q = 1:length(newQsize)]

  if merge
    outQt = mergereshape!(Qt)
  else
    outQt = Qt
  end
  return outQt
end

"""
    reshape!(M,a...[,merge=false])

In-place reshape for Qtensors for a tuple of indices (ex: [1,2],[3,4],[5]); can also make Qtensor unreshapable with `merge = true`, joining all grouped indices together

See also: [`reshape`](@ref)
"""
function reshape!(Qt::Qtens{W,Q}, newQsize::Array{P,1}...;merge::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  order = vcat(newQsize)
  if !issorted(order)
    permutedims!(Qt,order)
  end
  return reshape!(Qt,[newQsize[i] for i = 1:length(newQsize)],merge=merge)
end




"""
    reshape(M,S...[,merge=false])

Reshape for Qtensors (makes a copy) with integer inputs `S`; can also make Qtensor unreshapable with `merge`, joining all grouped indices together

# Warning
If the Qtensor size is (10,1,2) and we want to reshape in to a (10,2) tensor, this algorithm will reshape to a 
[[1,2],[3]] be default instead of [[1],[2,3]], so beware.

See also: [`reshape!`](@ref)
"""
function reshape(Qt::Qtens{W,Q}, S::Integer...;merge::Bool=false) where {W <: Number, Q <: Qnum}
  return reshape!(copy(Qt),S...,merge=merge)
end

"""
    reshape(M,S[,merge=false])

Reshape for Qtensors (makes a copy) with groups of indices `S` (ex: [[1,2],[3],[4,5]]); can also make Qtensor unreshapable with `merge`, joining all grouped indices together

# Warning
If the Qtensor size is (10,1,2) and we want to reshape in to a (10,2) tensor, this algorithm will reshape to a 
[[1,2],[3]] be default instead of [[1],[2,3]], so beware.

See also: [`reshape!`](@ref)
"""
function reshape(Qt::Qtens{W,Q}, newQsize::Array{Array{P,1},1};merge::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  return reshape!(copy(Qt), newQsize...,merge=merge)
end

"""
    reshape(M,S...[,merge=false])

Reshape for Qtensors (makes a copy) with a list of indices `S` (ex: [1,2],[3],[4,5]); can also make Qtensor unreshapable with `merge`, joining all grouped indices together

# Warning
If the Qtensor size is (10,1,2) and we want to reshape in to a (10,2) tensor, this algorithm will reshape to a 
[[1,2],[3]] be default instead of [[1],[2,3]], so beware.

See also: [`reshape!`](@ref)
"""
function reshape(Qt::Qtens{W,Q}, newQsize::Array{P,1}...;merge::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  return reshape!(copy(Qt), newQsize...,merge=merge)
end

"""
    reshape(M,S)

Reshape for Array `M` (makes a copy) with groups of indices `S` (ex: [[1,2],[3],[4,5]])

See also: [`reshape!`](@ref)
"""
function (reshape(Qt::Array{W,N}, newQsize::Array{Array{P,1},1};merge::Bool=false)::Array{W,length(newQsize)}) where {W <: Number, N, P <: Integer}
  M = Array{intType,1}(undef,length(newQsize))
  counter = 0
  @inbounds for g = 1:length(newQsize)
    counter += 1
    if length(newQsize[g]) > 0
      M[counter] = prod(b->size(Qt,b),newQsize[g])
    else
      M[counter] = 1
    end
  end
  return reshape(copy(Qt), M...)
end

"""
    getQnum(a,b,QnumMat,QnumSum)

Finds `Qnum` from `QnumSum` (vector of vectors of `Qnum`) based on index from `QnumMat` (vector of vectors of integers); `a` indexes the index represented on the `qarray`; `b` represents the value of the index
"""
function getQnum(a::Integer,b::Integer,QnumMat::Array{Array{P,1},1},QnumSum::Array{Array{Q,1},1}) where {Q <: Qnum, P <: intType}
  Qnumber = QnumMat[a][b]
  return QnumSum[a][Qnumber]
end

"""
    getQnum(a,b,Qt)

Finds `Qnum` from `qarray` `Qt`; `a` indexes the index represented on the `qarray`; `b` represents the value of the index
"""
function getQnum(a::Integer,b::Integer,Qt::Qtens{W,Q}) where {Q <: Qnum, W <: Number}
  return getQnum(a,b,Qt.QnumMat,Qt.QnumSum)
end

"""
    makenewindsL(LR,newQt,Qt,Rsize,offset)

Generates new index labels for `Qt`, a `qarray`, specifically the `.ind` field (left field; 1)
"""
function makenewindsL(LR::Integer,newQt::qarray,Qt::qarray,Rsize::Array{Array{P,1},1},offset::Integer) where P <: Integer
  newindsL = Array{Array{intType,2},1}(undef,length(newQt.T)) #[Array{intType,2}(undef,length(newQt.currblock[1]),size(newQt.ind[q][LR],2)) 
  @inbounds for q = 1:length(newQt.T)
    newindsL[q] = Array{intType,2}(undef,length(newQt.currblock[1]),size(newQt.ind[q][LR],2))
  end
  for q = 1:length(newindsL)
    for i = 1:size(newindsL[q],1)
      b = i + offset
      @inbounds for x = 1:size(newindsL[q],2)
        val = 0
        @inbounds @simd for a = length(Rsize[b]):-1:1
          index = Rsize[b][a]
          val *= length(Qt.QnumMat[index])
          val += newQt.ind[q][LR][index,x]
        end
        newindsL[q][i,x] = val
      end
    end
  end
  return newindsL
end

"""
    makenewindsR(LR,newQt,Qt,Rsize,offset)

Generates new index labels for `Qt`, a `qarray`, specifically the `.ind` field (right field; 2)
"""
function makenewindsR(LR::Integer,newQt::qarray,Qt::qarray,Rsize::Array{Array{P,1},1},offset::Integer) where P <: Integer
  newindsR = Array{Array{intType,2},1}(undef,length(newQt.T)) #[Array{intType,2}(undef,length(newQt.currblock[1]),size(newQt.ind[q][LR],2)) 
  @inbounds for q = 1:length(newQt.T)
    newindsR[q] = Array{intType,2}(undef,length(newQt.currblock[LR]),size(newQt.ind[q][LR],2))
  end
  for q = 1:length(newindsR)
    for i = 1:size(newindsR[q],1)
      b = i + offset
      @inbounds for x = 1:size(newindsR[q],2)
        val = 0
        @inbounds @simd for a = length(Rsize[b]):-1:1
          index = Rsize[b][a]
          val *= length(Qt.QnumMat[index])
          val += newQt.ind[q][LR][a,x]
        end
        newindsR[q][i,x] = val
      end
    end
  end
  return newindsR
end

"""
    mergeQNloop!(ninds,numElements,vec,pos,sizes,currflux,QNsummary,current_QnumMat,Qt)

Takes elements in `qarray` `Qt` to merge together indices and their quantum numbers together into a new index that is not unreshapeable


#Inputs:
+ `ninds`: number of indices  on the tensor
+ `numElements`: number of elements in the block
+ `vec`: indices to contract over
+ `pos`: vector to record where in the tensor the current position is
+ `sizes`: sizes of the `qarray`
+ `currflux`: flux of the `qarray`
+ `QNsummary`: quantum number summaries for each index of the `qarray`
+ `current_QnumMat`: `QnumMat` field of the resulting `qarray` (updated)
+ `Qt`: Input quantum number tensor
"""
function mergeQNloop!(ninds::Integer,numElements::Integer,vec::Array{P,1},pos::Array{P,1},
                      sizes::Tuple,currflux::Q,QNsummary::Array{Q,1},
                      current_QnumMat::Array{P,1},Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum, P <: Integer}
  @inbounds for y = 1:numElements
    position_incrementer!(pos,sizes)
    currQN = currflux
    @inbounds for b = 1:ninds
      currQN += getQnum(vec[b],pos[b],Qt)
    end
    b = 0
    findmatch = true
    @inbounds while findmatch && b < length(QNsummary)
      b += 1
      findmatch = QNsummary[b] != currQN
    end
    current_QnumMat[y] = b
  end
  nothing
end



"""
    multi_indexsummary(QnumSum,vec)

Generates the quantum numbers for a reshaped quantum number tensor by inputing the `QnumSum` from a `qarray` and the reshape rule as a vector of integers `vec`
"""
function multi_indexsummary(QnumSum::Array{Array{Q,1},1},vec::Array{P,1}) where {Q <: Qnum, P <: Integer}
  ninds = length(vec)
  if ninds > 0
    QsumSizes = ntuple(a->length(QnumSum[vec[a]]),ninds)
    Qsumel = 1
    @inbounds @simd for a = 1:ninds
      Qsumel *= QsumSizes[a]
    end
    Qsumvec = Array{Q,1}(undef,Qsumel)

    counter = 0
    pos = makepos(ninds)

    @inbounds for g = 1:Qsumel
      position_incrementer!(pos,QsumSizes)

      currQN = Q()
      @inbounds for b = 1:ninds
        currQN += QnumSum[vec[b]][pos[b]]
      end
      addQ = true
      w = 0
      @inbounds while w < counter && addQ
        w += 1
        addQ = currQN != Qsumvec[w]
      end
      if addQ
        counter += 1
        Qsumvec[counter] = currQN
      end
    end
    outQNsum = Qsumvec[1:counter]
  else
    outQNsum = [Q()]
  end
  return outQNsum
end

"""
    multi_indexsummary(Qt,vec)

Generates the quantum numbers for a reshaped quantum number tensor by inputing the `qarray` `Qt` and the reshape rule as a vector of integers `vec`
"""
function multi_indexsummary(Qt::Qtens{W,Q},vec::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}
  QnumSum = Qt.QnumSum
  return multi_indexsummary(QnumSum,vec)
end



####################################################
####################################################

"""
    C = countload(A,B)

Creates a vector `C` of all sizes from input arrays `A` and `B` (integers)
"""
function countload(A::Array{W,1},B::Array{W,1}) where W #{W <: Integer, R <: Integer}
  loadvec = Array{W,1}(undef,length(A)+length(B))
  counter = 0
  @inbounds @simd for k = 1:length(A)
    counter += 1
    loadvec[counter] = A[k]
  end
  @inbounds @simd for k = 1:length(B)
    counter += 1
    loadvec[counter] = B[k]
  end
  return loadvec
end


####################################################
####################################################

"""
    newindexsizeone!(Qt,S...)

Modifies size of output `qarray` `Qt` with sizes `S` (integers) that creates new indices of size 1 on a reshaped tensor
"""
function newindexsizeone!(Qt::Qtens{W,Q},S::Integer...) where {W <: Number, Q <: Qnum}
  size_ones = 0
  @inbounds for w = 1:length(S)
    size_ones += S[w] == 1
  end
  base_ones = 0
  @inbounds for w = 1:length(Qt.QnumMat)
    base_ones += length(Qt.QnumMat[w]) == 1
  end

  newindices = size_ones - base_ones

  if newindices > 0
    @inbounds for q = 1:length(Qt.ind)
      newind = Array{intType,2}(undef,size(Qt.ind[q][2],1)+newindices,size(Qt.ind[q][2],2))  #vcat(Qt.ind[q][2],zeros(intType,newindices,size(Qt.ind[q][2],2)))
      newind[1:size(Qt.ind[q][2],1),:] = Qt.ind[q][2]
      newind[size(Qt.ind[q][2],1)+1:end,:] = zeros(intType,newindices,size(Qt.ind[q][2],2))

      Qt.ind[q] = (Qt.ind[q][1],newind)
    end
    newinds = Array{intType,1}(undef,newindices)
    @inbounds @simd for w = 1:newindices
      newinds[w] = length(Qt.QnumMat) + w
    end

    newcurrblock = countload(Qt.currblock[2],newinds)

    Qt.currblock = (Qt.currblock[1],newcurrblock)

    zeroQN = Q()
    newindex = Array{Array{Q,1},1}(undef,newindices) #[[zeroQN] for w = 1:newindices]    
    @inbounds @simd for q = 1:newindices
      newindex[q] = [zeroQN]
    end

    Qt.QnumSum = countload(Qt.QnumSum,newindex)

    newQnum = Array{Array{intType,1},1}(undef,newindices)
    @inbounds @simd for w = 1:newindices
      newQnum[w] = [1]
    end

    Qt.QnumMat = countload(Qt.QnumMat,newQnum)
  end
  nothing
end


"""
    mergereshape!(M[,currblock=equalblocks(M)])

Groups all joined indices together to make one index that is unreshapable.  Dense tensors are unaffected.

See also: [`reshape!`](@ref)
"""
function mergereshape!(Qt::Qtens{W,Q};currblock::currblockTypes=equalblocks(Qt)) where {W <: Number, Q <: Qnum}

  Rsize = [Array{intType,1}(undef,length(Qt.size[w])) for w = 1:length(Qt.size)]
  counter = 0
  for w = 1:length(Rsize)
    for a = 1:length(Rsize[w])
      counter += 1
      Rsize[w][a] = counter
    end
  end
  newdim = length(Rsize)

  newQnumMat = Array{Array{intType,1},1}(undef,newdim)
  newQnumSum = Array{Array{Q,1},1}(undef,newdim)

  zeroQN = Q()
  truesize = basesize(Qt)

  @inbounds for a = 1:length(Rsize)
    if length(Rsize[a]) > 1

      sizes = truesize[Rsize[a]]
      thisflux = a == length(Rsize) ? Qt.flux : zeroQN

      Linds = Rsize[a]
      vec = Linds
      QNsummary = multi_indexsummary(Qt,Linds)

      ninds = length(vec)
      pos = makepos(ninds)
    
      numElements = prod(sizes)
      current_QnumMat = Array{intType,1}(undef,numElements)

      mergeQNloop!(ninds,numElements,vec,pos,sizes,thisflux,QNsummary,current_QnumMat,Qt)

      newQnumMat[a],newQnumSum[a] = current_QnumMat,QNsummary
    else
      thisind = Rsize[a][1]
      newQnumMat[a] = Qt.QnumMat[thisind]
      newQnumSum[a] = Qt.QnumSum[thisind]
    end
  end


  Linds = [i for i = 1:Rsize[end][1]-1]
  Rinds = Rsize[end]

  newQt = changeblock(Qt,Linds,Rinds)
  newQt.size = [[i] for i = 1:length(Rsize)]
  merged_currblock = ([i for i = 1:length(Rsize)-1],[length(Rsize)])

  newQt.currblock = merged_currblock

  newindsL = makenewindsL(1,newQt,Qt,Rsize,0)
  newindsR = makenewindsR(2,newQt,Qt,Rsize,length(newQt.currblock[1]))

  newQt.ind = [(newindsL[q],newindsR[q]) for q = 1:length(newindsL)]

  newQt.QnumMat = newQnumMat
  newQt.QnumSum = newQnumSum

  return newQt
end

"""
    mergereshape(M[,currblock=equalblocks(M)])

Groups all joined indices together to make one index that is unreshapable.  Dense tensors are unaffected.

See also: [`reshape`](@ref)
"""
function mergereshape(Qt::Qtens{W,Q};currblock::currblockTypes=equalblocks(Qt)) where {W <: Number, Q <: Qnum}
  cQt = copy(Qt)
  return mergereshape!(cQt,currblock=currblock)
end

"""
    unreshape!(Qt,sizes...)

Equivalent to `reshape!` for a `densTensType` for a string of integers of new sizes `sizes`
"""
function unreshape!(Qt::densTensType,sizes::W...) where {W <: Integer}
  return reshape!(Qt,sizes...)
end

"""
    unreshape!(Qt,sizes)

Equivalent to `reshape!` for a `densTensType` for an array of new sizes `sizes`
"""
function unreshape!(Qt::densTensType,sizes::Array{W,1}) where {W <: Integer}
  return reshape!(Qt,sizes...)
end

"""
    unreshape!(Qt,sizes...)

Equivalent to `reshape!` for a `qarray` for a string of integers of new sizes `sizes`
"""
function unreshape!(Qt::qarray,sizes::W...) where W <: Integer
  return unreshape(Qt)
end

"""
    unreshape!(Qt,sizes)

Equivalent to unreshape for a `qarray` for an array of new sizes `sizes`
"""
function unreshape!(Qt::qarray,sizes::Array{W,1}) where W <: Integer
  return unreshape(Qt)
end

"""
    unreshape!(Qt)

Unreshapes `qarray` to base sizes (taken from `Qt.QnumMat`)

See also: [`unreshape`](@ref)
"""
function unreshape!(Qt::qarray)
  return unreshape(Qt)
end

"""
    unreshape(Qt)

Unreshapes `qarray` to base sizes (taken from `Qt.QnumMat`)
"""
function unreshape(Qt::qarray)
  Qt.size = [[i] for i = 1:length(Qt.QnumMat)]
  return Qt
end

"""
    unreshape(Qt,sizes)

Equivalent to reshape for a `densTensType` for an array of new sizes `sizes`
"""
function unreshape(Qt::densTensType,sizes::Array{W,1}) where W <: Integer
  return reshape(Qt,sizes...)
end

"""
    unreshape(Qt,sizes...)

Equivalent to reshape for a `densTensType` for a string of integers of new sizes `sizes`
"""
function unreshape(Qt::densTensType,sizes::W...) where W <: Integer
  return reshape(Qt,sizes...)
end

"""
    unreshape(Qt,sizes...)

Equivalent to reshape for a `qarray` for a string of integers of new sizes `sizes`
"""
function unreshape(Qt::qarray,sizes::W...) where W <: Integer
  return unreshape!(copy!(Qt),sizes...)
end

"""
    unreshape(Qt,sizes)

Equivalent to reshape for a `qarray` for an array of new sizes `sizes`
"""
function unreshape(Qt::qarray,sizes::Array{W,1}) where W <: Integer
  return unreshape!(copy!(Qt),sizes...)
end

"""
    recoverShape(Qt,S...)

Creates size fields for an input `qarray` `Qt`
"""
function recoverShape(Qt::Qtens{W,Q},S::Integer...) where {W <: Number, Q <: Qnum}
  Rsize = Array{Array{intType,1},1}(undef,length(S))
  count = 1
  wstart = 1
  for a = 1:length(Rsize)
    if count > length(Qt.QnumMat)
      Rsize[a] = [count]
      count += 1
    else
      currdim = length(Qt.QnumMat[count])
      while count < length(Qt.QnumMat) && currdim < S[a]
        count += 1
        currdim *= length(Qt.QnumMat[count])
      end
      while length(Qt.QnumMat) > count && a == length(Rsize) && currdim*length(Qt.QnumMat[count+1]) == S[a] #&& length(Qt.QnumMat[count+1]) == 1
        count += 1
      end
      Rsize[a] = [i for i = wstart:count]
      count += 1
      wstart = count
    end
  end
  return Rsize
end
