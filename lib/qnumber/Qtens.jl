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
    leftsummary,rightsummary = LRsummary_invQ(QNsummary,flux)

generates quantum number summaries of a given `QNsummary` (=`leftsummary`) and the inverse `rightsummary` given a certain `flux`
"""
function LRsummary_invQ(QNsummary::Array{Q,1},flux::Q) where Q <: Qnum
  leftSummary = QNsummary
  rightSummary = Array{Q,1}(undef,length(leftSummary))
  @inbounds for q = 1:length(rightSummary)
    rightSummary[q] = flux - QNsummary[q]
  end
  return leftSummary,rightSummary
end

"""
    QNsummary,leftSummary,rightSummary,newQblocksum = makeQNsummaries(Qlabels,Linds,Rinds,LR,flux)

Generates quantum number summaries `QNsummary` for each block in the `Qtens` object (two summaries, `leftSummary` and `rightSummary` for two index groups `Linds` and `Rinds`, respectively). `LR` determines whether the computation should be applied to the left or right group where the other group is found by using the `flux` instead of computation of the quantum numbers; the field `newQblocksum` is the resulting quantum number summary for the summaries for the rows and colums of the blocks

See also: [`Qtens`](@ref)
"""
function makeQNsummaries(Qlabels::Array{Array{Q,1},1},Linds::Array{P,1},Rinds::Array{P,1},LR::Bool,flux::Q) where {Q <: Qnum, P <: Integer}
  if LR
    QNsummary = multi_indexsummary(Qlabels,Linds)
    leftSummary,rightSummary = LRsummary_invQ(QNsummary,flux)
  else
    QNsummary = multi_indexsummary(Qlabels,Rinds)
    rightSummary,leftSummary = LRsummary_invQ(QNsummary,flux)
  end
  newQblocksum = Array{NTuple{2,Q},1}(undef,length(QNsummary))
  @inbounds for q = 1:length(QNsummary)
    newQblocksum[q] = (leftSummary[q],rightSummary[q])
  end
  return QNsummary,leftSummary,rightSummary,newQblocksum
end

"""
    QNsummary,leftSummary,rightSummary,newQblocksum = makeQNsummaries(Qtens,Linds,Rinds,LR)

Generates quantum number summaries `QNsummary` for a `Qtens` object based on a group of `Linds` and `Rinds` and whether to compute the left (`LR` = true) or right (`LR` = false) groups

See also: [`Qtens`](@ref)
"""
function makeQNsummaries(Qt::Qtens{W,Q},Linds::Array{P,1},Rinds::Array{P,1},LR::Bool) where {W <: Number, Q <: Qnum, P <: Integer}
  return makeQNsummaries(Qt.QnumSum,Linds,Rinds,LR,Qt.flux)
end

"""
    QnumMat,QnumSum = convertQnumMat(Qlabels)

Converts an `Array{Array{Qnum,1},1}` into a `QnumMat` of an `Array{Array{intType,1},1}` (arrays of integers) that relate to the summaries of `Qnum`s in `QnumSum`

See also: [`Qnum`](@ref) [`intType`](@ref)
"""
function convertQnumMat(QnumMat::Array{Array{Q,1},1}) where Q <: Qnum
  QnumSum = unique.(QnumMat)
  return convertQnumMat(QnumMat,QnumSum),QnumSum
end

"""
    QnumMat = convertQnumMat(Qlabels,QnumSum)

Converts an `Array{Array{Qnum,1},1}` into a `QnumMat` of an `Array{Array{intType,1},1}` (arrays of integers) that relate to the summaries of `Qnum`s in the input `QnumSum`

See also: [`Qnum`](@ref) [`intType`](@ref)
"""
function convertQnumMat(QnumMat::Array{Array{Q,1},1},QnumSum::Array{Array{Q,1},1}) where Q <: Qnum
  finalQnumMat = Array{Array{intType,1},1}(undef,length(QnumMat))
  @inbounds for i = 1:length(QnumMat)
    finalQnumMat[i] = Array{intType,1}(undef,length(QnumMat[i]))
    @inbounds for w = 1:length(QnumMat[i])
      y = 0
      notmatchingQN = true
      @inbounds while notmatchingQN
        y += 1
        notmatchingQN = QnumMat[i][w] != QnumSum[i][y]
      end
      finalQnumMat[i][w] = y
    end
  end
  return finalQnumMat
end
export convertQnumMat

"""
    Qt = Qtens(Qlabels[,arrows,datatype=Float64,flux=Q(),currblock=equalblocks,blockfct=undefMat])

Creates empty `Qtens` with array type `datatype` (default Float64), quantum number labels given by `Qlabels` and arrow convention `arrows`; initial blocks of the `qarray` are given by `equalblocks` function and blocks are initialized with `undefMat`; initial flux value is whatever zero is defined for the input quantum number type

See also: [`equalblocks`](@ref) [`undefMat`](@ref)
"""
function Qtens(Qlabels::Array{Array{Q,1},1};type::DataType=Float64,currblock::currblockTypes=equalblocks(Qlabels),flux::Q=Q(),blockfct::Function=undefMat) where Q <: Qnum

  pLinds = currblock[1]
  pRinds = currblock[2]

  Lsizes = Array{intType,1}(undef,length(pLinds))
  @inbounds for w = 1:length(pLinds)
    Lsizes[w] = length(Qlabels[pLinds[w]])
  end

  Rsizes = Array{intType,1}(undef,length(pRinds))
  @inbounds for w = 1:length(pRinds)
    Rsizes[w] = length(Qlabels[pRinds[w]])
  end

  Lsize = prod(Lsizes)
  Rsize = prod(Rsizes)
  LR = Lsize < Rsize

  QNsummary,leftSummary,rightSummary,newQblocksum = makeQNsummaries(Qlabels,pLinds,pRinds,LR,flux)
  
  finalQnumMat, QnumSum = convertQnumMat(Qlabels)

  leftQNs,Lbigtosub,rows,Lindexes,rightQNs,Rbigtosub,columns,Rindexes = QnumList(Lsizes,Rsizes,finalQnumMat,QnumSum,pLinds,pRinds,leftSummary,rightSummary)

  finalQNs = 0
  @inbounds @simd for q = 1:length(rows)
    finalQNs += rows[q] > 0 && columns[q] > 0
  end

  newblocks = Array{Array{type,2},1}(undef,finalQNs)
  newind = Array{NTuple{2,Array{intType,2}},1}(undef,finalQNs)
  finalQblocksum = Array{NTuple{2,Q},1}(undef,finalQNs)
  counter = 0
  @inbounds for q = 1:length(rows)
    if rows[q] > 0 && columns[q] > 0
      counter += 1
      newblocks[counter] = blockfct(type,rows[q],columns[q])
      newind[counter] = (Lindexes[q],Rindexes[q])
      finalQblocksum[counter] = newQblocksum[q]
    end
  end

  newsize = Array{Array{intType,1},1}(undef,length(Qlabels))
  @inbounds @simd for i = 1:length(newsize)
    newsize[i] = [i]
  end

  newcurrblock = (currblock[1],currblock[2])
  return Qtens{type,Q}(newsize, newblocks, newind, newcurrblock, finalQblocksum, finalQnumMat, QnumSum, flux)
end

"""
    Qt = Qtens(Qlabels,arrows[,type=Float64,flux=Q(),currblock=equalblocks,blockfct=undefMat])

Creates empty `Qtens` with array type `type` (default Float64), quantum number labels given by `Qlabels` and arrow convention `arrows`; initial blocks of the `qarray` are given by `equalblocks` function and blocks are initialized with `undefMat`; initial flux value is whatever zero is defined for the input quantum number type

See also: [`equalblocks`](@ref) [`undefMat`](@ref)
"""
function Qtens(Qlabels::Array{Array{Q,1},1}, arrows::Array{Bool,1};type::DataType=Float64,currblock::currblockTypes=equalblocks(Qlabels),flux::Q=Q(),blockfct::Function=undefMat) where Q <: Qnum
  newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
  return Qtens(newQlabels,type=type,currblock=currblock,flux=flux,blockfct=blockfct)
end

"""
    Qt = Qtens(type,Qlabels,arrows[,flux=Q(),currblock=equalblocks,blockfct=undefMat])

Creates empty `Qtens` with array type `type` (default Float64), quantum number labels given by `Qlabels` and arrow convention `arrows`; initial blocks of the `qarray` are given by `equalblocks` function and blocks are initialized with `undefMat`; initial flux value is whatever zero is defined for the input quantum number type

See also: [`equalblocks`](@ref) [`undefMat`](@ref)
"""
function Qtens(type::DataType,Qlabels::Array{Array{Q,1},1}, arrows::Array{Bool,1};currblock::currblockTypes=equalblocks(Qlabels),flux::Q=Q(),blockfct::Function=undefMat) where Q <: Qnum
  return Qtens(Qlabels,arrows,type=type,currblock=currblock,flux=flux,blockfct=blockfct)
end

"""
    Qt = Qtens(type,Qlabels[,flux=Q(),currblock=equalblocks,blockfct=undefMat])

Creates empty `Qtens` with array type `type` (default Float64), quantum number labels given by `Qlabels` and arrow convention `arrows`; initial blocks of the `qarray` are given by `equalblocks` function and blocks are initialized with `undefMat`; initial flux value is whatever zero is defined for the input quantum number type

See also: [`equalblocks`](@ref) [`undefMat`](@ref)
"""
function Qtens(type::DataType,Qlabels::Array{Array{Q,1},1};currblock::currblockTypes=equalblocks(Qlabels),flux::Q=Q(),blockfct::Function=undefMat) where Q <: Qnum
  return Qtens(Qlabels,type=type,currblock=currblock,flux=flux,blockfct=blockfct)
end

"""
    leftinds,rightinds = equalblocks(vectQN)

Proposes a block structure that makes the matrix equivalent approximate equal in both number of rows and columns for a given Array of Array of quantum numbers `A`
"""
function equalblocks(vectQN::Array{Array{Q,1},1}) where Q <: Qnum
  sizes = ntuple(q->length(vectQN[q]),length(vectQN))
  return equalblocks(sizes)
end

"""
    leftinds,rightinds = equalblocks(sizes)

Proposes a block structure that makes the matrix equivalent approximate equal in both number of rows and columns for a given set of sizes `sizes`
"""
function equalblocks(sizes::Tuple)
  row = sizes[1]
  column = prod(sizes) ÷ row
  i = 1
  @inbounds while row < column && i < length(sizes) - 1
    i += 1
    row *= sizes[i]
    column ÷= sizes[i]
  end
  return ([w for w = 1:i],[w for w = i+1:length(sizes)])
end

"""
    leftinds,rightinds = equalblocks(A)

Proposes a block structure that makes the matrix equivalent approximate equal in both number of rows and columns for a given `qarray` `A`
"""
function equalblocks(A::qarray)
  Rsize = A.size
  sizes = ntuple(q->prod(w->length(A.QnumMat[w]),Rsize[q]),length(Rsize))
  return equalblocks(sizes)
end

"""
    B = recoverQNs(q,QnumMat,QnumSum)

Returns `Qnum` `B` associated with the `q`th value of the `QnumMat` (vectors of integers) according to the `QnumSum`

See also: [`Qnum`](@ref)
"""
function recoverQNs(q::Integer,QnumMat::Array{Array{intType,1},1},QnumSum::Array{Array{Q,1},1}) where Q <: Qnum
  return [getQnum(q,i,QnumMat,QnumSum) for i = 1:length(QnumMat[q])]
end

"""
    B = recoverQNs(q,Qt)

Returns `Qnum` `B` associated with the `q`th value of the .QnumMat (vectors of integers) according to the .QnumSum of a tensor `Qt`

See also: [`Qnum`](@ref)
"""
function recoverQNs(q::Integer,Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return recoverQNs(q,Qt.QnumMat,Qt.QnumSum)
end
export recoverQNs

"""
    QnumMat = fullQnumMat(QnumMat,QnumSum)

Converts the quantum number entries `QnumSum` and `QnumMat` the full format

See also: [`Qnum`](@ref) [`intType`](@ref)
"""
function fullQnumMat(QnumMat::Array{Array{intType,1},1},QnumSum::Array{Array{Q,1},1}) where Q <: Qnum
  finalQnumMat = [recoverQNs(q,QnumMat,QnumSum) for q = 1:length(QnumMat)]
  return finalQnumMat
end

"""
    QnumMat = fullQnumMat(A)

Converts the quantum number entries `QnumSum` and `QnumMat` of `Qtens` `A` into the full format

See also: [`Qnum`](@ref) [`intType`](@ref)
"""
function fullQnumMat(Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return fullQnumMat(Qt.QnumMat,Qt.QnumSum)
end
export fullQnumMat

"""
    Qt = Qtens(operator,QnumMat[,Arrows,zero=])

Creates a dense `operator` (`denstens`) `Qt` as a Qtensor with quantum numbers `QnumMat` on each index (`Array{Array{Q,1},1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
"""
function Qtens(optens::denstens,Qtensor::qarray;zero::Number=0.,currblock::currblockTypes=Qtensor.currblock)
  Op = Array(optens)
  finalQnumMat = fullQnumMat(Qtensor.QnumMat,Qtensor.QnumSum)
  return Qtens(Op,finalQnumMat,zero=zero,currblock=currblock)
end

"""
    Qt = Qtens(operator,QnumMat[,Arrows,zero=])

Creates a dense `operator` (`Array`) `Qt` as a Qtensor with quantum numbers `QnumMat` on each index (`Array{Array{Q,1},1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
"""
function Qtens(optens::Array,Qtensor::qarray;zero::Number=0.,currblock::currblockTypes=Qtensor.currblock)
  finalQnumMat = fullQnumMat(Qtensor.QnumMat,Qtensor.QnumSum)
  return Qtens(optens,finalQnumMat,zero=zero,currblock=currblock)
end

"""
    Qt = Qtens(operator,QnumMat[,Arrows,zero=])

Creates a dense `operator` `Qt` as a Qtensor with quantum numbers `QnumMat` on each index (`Array{Array{Q,1},1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
"""
function Qtens(Op::R,Qlabels::Array{Array{Q,1},1},Arrows::U...;zero::Number=0.,currblock::currblockTypes=([i for i = 1:ndims(Op)-1],[ndims(Op)]),type::DataType=eltype(Op)) where {Q <: Qnum, R <: densTensType, U <: Union{Bool,Array{Bool,1}}}
  theseArrows = typeof(Arrows) <: Bool ? Arrows : (Arrows[1]...,)
  newQnumMat = [theseArrows[q] ? Qlabels[q] : inv.(Qlabels[q]) for q = 1:length(Qlabels)]
  return Qtens(Op,newQnumMat;zero=zero,currblock=currblock,type=type)
end

"""
    Qt = Qtens(QnumMat,operator...[,Arrows,zero=])

Creates set of dense `operator`s (`Qt`) as a Qtensor with quantum numbers `QnumMat` on each index (`Array{Array{Q,1},1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)
"""
function Qtens(Qlabels::Array{Array{Q,1},1},Op...;zero::Number=0.,currblock::currblockTypes=([i for i = 1:ndims(Op[1])-1],[ndims(Op[1])])) where Q <: Qnum
  if length(Op) > 1
    out = ntuple(w->Qtens(Op[w],Qlabels,zero=zero,currblock=currblock),length(Op))
  else
    out = Qtens(Op[1],Qlabels,zero=zero,currblock=currblock)
  end
  return out
end

"""
    Qt = Qtens(Qlabel,operator...[,Arrows,zero=])

Creates set of dense `operator`s (`Qt`) as a Qtensor with identical quantum numbers `Qlabel` on each index (`Array{Q,1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0). Will conjugate the second index (assume rank-2 input for `operator`)
"""
function Qtens(Qlabels::Array{Q,1},Op...;zero::Number=0.,currblock::currblockTypes=([i for i = 1:ndims(Op[1])-1],[ndims(Op[1])])) where {Q <: Qnum}
  Qnumvec = [Qlabels,inv.(Qlabels)]
  return Qtens(Qnumvec,Op...,zero=zero,currblock=currblock)
end

"""
    Qt = Qtens(operator,Qlabel,[,Arrows,zero=])

Creates set of dense `operator`s (`Qt`) as a Qtensor with identical quantum numbers `Qlabel` on each index (`Array{Q,1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0). Will conjugate the second index (assume rank-2 input for `operator`)
"""
function Qtens(Op::densTensType,Qlabels::Array{Q,1};zero::R=eltype(Op)(0),currblock::currblockTypes=([i for i = 1:ndims(Op)-1],[ndims(Op)]),type::DataType=eltype(Op),blockfct::Function=undefMat) where {Q <: Qnum, R <: Number}

  if ntuple(w->length(Qlabels),ndims(Op)) != size(Op)
    error("Input correct quantum numbers as a vector of vector of quantum numbers")
  end

  return Qtens(Op,[Qlabels],zero=zero,currblock=currblock,type=type,blockfct=blockfct)
end

"""
    Qt = Qtens(operator,Qlabel,[,Arrows,zero=])

Creates set of dense `operator`s (`Qt`) as a Qtensor with identical quantum numbers `Qlabel` on each index (`Array{Q,1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0). Will conjugate the second index (assume rank-2 input for `operator`)
"""
function Qtens(Op::densTensType,Qlabels::Array{Array{Q,1},1};zero::R=eltype(Op)(0),currblock::currblockTypes=([i for i = 1:ndims(Op)-1],[ndims(Op)]),type::DataType=eltype(Op),blockfct::Function=undefMat) where {Q <: Qnum, R <: Number}
  if isapprox(norm(Op),0)
    return zeros(Qlabels,type=type,currblock=currblock)
  end

  pLinds = currblock[1]
  pRinds = currblock[2]
  
  sizes = tupsize(Op)

  Op_mat = reshape(Array(Op),[pLinds,pRinds])

  pos = makepos(length(sizes))

  currval = 0
  saveval = 0
  savepos = Array{Int64,1}(undef,length(pos))
  for x = 1:length(Op_mat)
    position_incrementer!(pos,sizes) #could be faster, but isn't necessary here)
    if abs(Op_mat[x]) > currval
      currval = abs(Op_mat[x])
      saveval = x
      @inbounds for w = 1:length(pos)
        savepos[w] = pos[w]
      end
    end
  end
  x = saveval
  pos = savepos
  invflux = sum(w->Qlabels[w][pos[w]],1:length(pos))

  Qt = Qtens(Qlabels,type=type,flux=invflux,currblock=currblock)

  Lsizes = sizes[pLinds]
  Rsizes = sizes[pRinds]

  pos = ntuple(i->Array{intType,1}(undef,length(Qt.currblock[i])),2)
  for q = 1:length(Qt.T)
    for b = 1:size(Qt.T[q],2)
      @inbounds for a = 1:size(Qt.T[q],1)
        for i = 1:2
          blockindex = i == 1 ? a : b
          @inbounds for w = 1:length(Qt.currblock[i])
            pos[i][w] = Qt.ind[q][i][w,blockindex] + 1
          end
        end
        x = pos2ind(pos[1],Lsizes)
        y = pos2ind(pos[2],Rsizes)
        Qt.T[q][a,b] = Op_mat[x,y]
      end
    end
  end

  keepers = Array{Bool,1}(undef,length(Qt.T))
  for q = 1:length(keepers)
    allzero = (size(Qt.T[q],1) > 0 && size(Qt.T[q],2) > 0)
    w = 0
    while allzero && w < length(Qt.T[q])
      w += 1
      allzero &= isapprox(Qt.T[q][w],zero)
    end
    keepers[q] = !allzero
  end

  if sum(keepers) < length(Qt.T)
    Qt.T = Qt.T[keepers]
    Qt.ind = Qt.ind[keepers]
    Qt.Qblocksum = Qt.Qblocksum[keepers]
  end
  return Qt
end

"""
    Qtens(operator,Qlabels[,Arrows,zero=])

Creates a dense `operator` as a Qtensor with quantum numbers `Qlabels` common to all indices (`Array{Q,1}`) according to arrow convention `Arrows` and all elements equal to `zero` are not included (default 0)

See also: [`densTensType`](@ref)
"""
function Qtens(operator::Union{tens{W},Array{W,N}},Qlabels::Array{Q,1},arrows::U;zero::Number=0.,currblock::currblockTypes=([i for i = 1:ndims(Op)-1],[ndims(Op)]),type::DataType=eltype(Op),blockfct::Function=undefMat) where {Q <: Qnum, U <: Union{Bool,Array{Bool,1}}, W <: Number, N}
  newQlabels = Array{Q,1}[arrows[a] ? Qlabels[a] : inv.(Qlabels[a]) for a = 1:length(arrows)]
  return Qtens(operator,newQlabels,zero=zero,currblock=currblock,type=type,blockfct=blockfct)
end

"""
    Qtens(A)

`A` is a Qtensor; makes shell of a `A` with only meta-data (empty blocks); used mainly for copies
"""
function Qtens(A::Qtens{W,Q};zero::Number=0.,currblock::currblockTypes=A.currblock,type::DataType=eltype(A),blockfct::Function=undefMat) where {W <: Number, Q <: Qnum}
  newQlabels = fullQnumMat(A)
  return Qtens(newQlabels,#=zero=zero,=#currblock=currblock,type=type,blockfct=blockfct,flux=A.flux)
end

"""
    Qtens{Z,Q}()

Default initializer to an empty tensor with type `Z` and quantum number `Q`
"""
function Qtens{Z,Q}() where {Z <: Number,Q <: Qnum}
  return zeros(Z,[[Q()]])
end

"""
    Qtens(T,Qt)

Convert Qtensor `Qt` to type `T`
"""
function Qtens(T::DataType, Qt::Qtens{Z,Q}) where {Z <: Number, Q <: Qnum}
  if Z == T
    return Qt
  else
    newsize = Qt.size #[copy(Qt.size[i]) for i = 1:length(Qt.size)]
    newQblocksum = Qt.Qblocksum
    newT = [convert(Array{T,2},Qt.T[w]) for w = 1:length(Qt.T)]
    newcurrblock = Qt.currblock #(Qt.currblock[1],Qt.currblock[2]) #(copy(Qt.currblock[1]),copy(Qt.currblock[2]))
    return Qtens{T,Q}(newsize,newT,copy(Qt.ind),newcurrblock,newQblocksum,
                      Qt.QnumMat,Qt.QnumSum,Qt.flux)
  end
end

