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
    leftQNs,Lbigtosub,rows,Lindexes,rightQNs,Rbigtosub,columns,Rindexes = QnumList(Qtens,Lind,Rinds,leftSummary,rightSummary)

Generates quantum numbers along rows (`Linds`) and columns (`Rinds`) with quantum number summaries `leftSummary` and `rightSummary` from `Qtens`
"""
function QnumList(Qt::Qtens{W,Q},pLinds::Array{P,1},pRinds::Array{P,1},
                leftSummary::Array{Q,1},rightSummary::Array{Q,1}) where {P <: Integer, W <: Number, Q <: Qnum}

  Lsizes = [length(Qt.QnumMat[pLinds[a]]) for a = 1:length(pLinds)]
  Rsizes = [length(Qt.QnumMat[pRinds[a]]) for a = 1:length(pRinds)]

  return QnumList(Lsizes,Rsizes,Qt.QnumMat,Qt.QnumSum,pLinds,pRinds,leftSummary,rightSummary)
end

"""
    leftQNs,Lbigtosub,rows,Lindexes,rightQNs,Rbigtosub,columns,Rindexes = QnumList(Lsizes,Rsizes,QnumMat,QnumSum,Lind,Rinds,leftSummary,rightSummary)

Generates quantum numbers along rows (`Linds`) and columns (`Rinds`) with quantum number summaries `leftSummary` and `rightSummary` from `QnumMat` and `QnumSum` and sizes of indices `Lsizes` and `Rsizes`
"""
function QnumList(Lsizes::Array{P,1},Rsizes::Array{P,1},QnumMat::Array{Array{P,1},1},QnumSum::Array{Array{Q,1},1},pLinds::Array{P,1},pRinds::Array{P,1},
                  leftSummary::Array{Q,1},rightSummary::Array{Q,1}) where {P <: Integer, Q <: Qnum}

  leftQNs,Lbigtosub,rows = QnumList(Lsizes,QnumMat,QnumSum,pLinds,leftSummary)
  rightQNs,Rbigtosub,columns = QnumList(Rsizes,QnumMat,QnumSum,pRinds,rightSummary)

  Lindexes,Rindexes,leftQNs,rightQNs,rows,columns = makeIndexes(pLinds,Lsizes,leftQNs,Lbigtosub,rows,pRinds,Rsizes,rightQNs,Rbigtosub,columns)

  return leftQNs,Lbigtosub,rows,Lindexes,rightQNs,Rbigtosub,columns,Rindexes
end

"""
    loadindex!(nsizes,Linds,Lsizes,Lindexes,leftQNs,Lbigtosub)

Populates `Lindexes` with values from a vector corresponding to indexes `Linds`. `nszies` is a vector of booleans corresponding to values that are non-zero values in the values. `Lbigtosub` is a vector of the row and column of the dense matrix-equivalent for each row of the quantum number block. `leftQNs` is the quantum numbers (integers corresponding to a vector of `Qnum`s) along that side. Generates the `.ind` field for a Qtensor.
"""
function loadindex!(nonzero_sizes::Array{Bool,1},Linds::intType,Lsizes::Array{intType,1},Lindexes::Array{Array{intType,2},1},
                           leftQNs::Array{intType,1},Lbigtosub::Array{intType,1},rows::Array{intType,1})
  Lpos = makepos(Linds)
  @inbounds for y = 1:length(Lbigtosub)
    position_incrementer!(Lpos,Lsizes)
    q = leftQNs[y]
    if q > 0 && nonzero_sizes[q]
      w = Lbigtosub[y]
      if rows[q] > 0
        @inbounds @simd for r = 1:Linds
          Lindexes[q][r,w] = Lpos[r] - 1
        end
      end
    end
  end
  nothing
end

"""
  QnumList(sizes,QnumMat,vec,QnumSum)

Generates quantum number list (integers) based on `QnumSum` and `QnumMat` for indices in `vec` with total sizes `sizes` (these are input instead of pulled from `QnumMat` because it is recyclable in other parts of the contraction routine)
"""
function QnumList(sizes::Union{Array{P,1},NTuple{G,P}},QnumMat::Array{Array{P,1},1},QnumSum::Array{Array{Q,1},1},
                        vec::Array{P,1},QnumSummary::Array{Q,1}) where {Q <: Qnum, G, P <: Integer}
  ninds = length(sizes)
  if ninds == 0
    matchingQNs,returnvector,QNblocksizes = [1],[1],[1]
  else
    pos = makepos(ninds)

    numElements = 1
    @inbounds @simd for w = 1:length(sizes)
      numElements *= sizes[w]
    end
    matchingQNs = Array{intType,1}(undef,numElements)
    returnvector = Array{intType,1}(undef,numElements)

    numQNs = length(QnumSummary)
    QNblocksizes = zeros(intType,numQNs)

    @inbounds for y = 1:numElements
      position_incrementer!(pos,sizes)
      currQN = Q()
      @inbounds for b = 1:ninds
        currQN += getQnum(vec[b],pos[b],QnumMat,QnumSum)
      end
      notmatchQNs = true
      q = 0
      @inbounds while (q < numQNs) && notmatchQNs
        q += 1
        notmatchQNs = currQN != QnumSummary[q]
      end

      if notmatchQNs
        matchingQNs[y] = 0
        returnvector[y] = 0
      else
        matchingQNs[y] = q

        QNblocksizes[q] += 1
        returnvector[y] = QNblocksizes[q]
      end
    end
  end
  return matchingQNs,returnvector,QNblocksizes
end


"""
    Lindexes,Rindexes,leftQNs,rightQNs,rows,columns = makeIndexes(pLinds,Lsizes,leftQNs,Lbigtosub,rows,pRinds,Rsizes,rightQNs,Rbigtosub,columns)

Generates the index fields in a `Qtens` from 
"""
function makeIndexes(pLinds::Array{intType,1},Lsizes::Array{intType,1},leftQNs::Array{intType,1},Lbigtosub::Array{intType,1},rows::Array{intType,1},
                     pRinds::Array{intType,1},Rsizes::Array{intType,1},rightQNs::Array{intType,1},Rbigtosub::Array{intType,1},columns::Array{intType,1})
  Linds = length(pLinds)
  Rinds = length(pRinds)
  numQNs = length(rows)
  nonzero_sizes = Array{Bool,1}(undef,numQNs)
  @inbounds @simd for q = 1:numQNs
    nonzero_sizes[q] = rows[q] > 0 && columns[q] > 0
  end

#  finalQNs = sum(nonzero_sizes)

  Lindexes = Array{Array{intType,2},1}(undef,numQNs)
  Rindexes = Array{Array{intType,2},1}(undef,numQNs)
  @inbounds @simd for q = 1:numQNs
    if nonzero_sizes[q]
      Lindexes[q] = Array{intType,2}(undef,Linds,rows[q])
      Rindexes[q] = Array{intType,2}(undef,Rinds,columns[q])
    end
  end

  loadindex!(nonzero_sizes,Linds,Lsizes,Lindexes,leftQNs,Lbigtosub,rows)
  loadindex!(nonzero_sizes,Rinds,Rsizes,Rindexes,rightQNs,Rbigtosub,columns)
  return Lindexes,Rindexes,leftQNs,rightQNs,rows,columns
end
