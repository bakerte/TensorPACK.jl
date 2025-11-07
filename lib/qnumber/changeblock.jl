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
    B = basesize(A)

Generates the base size `B` of `A` no matter how it was reshaped
"""
function basesize(Qtensor::qarray)
  return ntuple(i->length(Qtensor.QnumMat[i]),length(Qtensor.QnumMat))
end

"""
    B = basesize(A)

Generates the base size `B` of quantum number labels `A` (vector of a vector of `Qnum`)
"""
function basesize(Qlabels::Array{Array{Q,1},1}) where Q <: Union{Qnum,Integer}
  return ntuple(i->length(Qlabels[i]),length(Qlabels))
end
export basesize

"""
    B = basesize(A,Linds)

Generates the base size `B` of `qarray` `A` over indices `Linds` (vector of a vector of `Qnum`)
"""
function findsizes(Qt::Qtens{W,Q},Linds::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}
  nind = length(Linds)
  Lnonzero = nind > 0
  Lsize = Lnonzero ? prod(Linds) : 1
  return Lsize,nind,Lnonzero
end

"""
    checkorder!(Qt,blockside)

checks .currblock (`blockside` group: 1 or 2) of `Qt` fields to be in the correct order and reorders .currblock and .ind fields to match
"""
function checkorder!(Qt::Qtens{W,Q},side::Integer) where {W <: Number, Q <: Qnum}
  xsorted = issorted(Qt.currblock[side])  
  Lindmat = Array{Array{intType,2},1}(undef,length(Qt.ind))
  if !xsorted
    xorder = sortperm(Qt.currblock[side])
    sort!(Qt.currblock[side])
    @inbounds for q = 1:length(Qt.ind)
      Lindmat[q] = Qt.ind[q][side][xorder,:]
    end
  else
    @inbounds for q = 1:length(Qt.ind)
      Lindmat[q] = Qt.ind[q][side]
    end
  end
  return Lindmat
end

"""
    checkorder!(Qt)

checks both .currblock of `Qt` fields to be in the correct order and reorders .currblock and .ind fields to match
"""
function checkorder!(Qt::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  Lindmat = checkorder!(Qt,1)
  Rindmat = checkorder!(Qt,2)
  @inbounds for q = 1:length(Qt.ind)
    Qt.ind[q] = (Lindmat[q],Rindmat[q])
  end
  nothing
end

"""
    checkzeroblocks!(newblocks,keepblocks)

Finds blocks that are only full of zeros for removal; inputs `newblocks` are the blocks to check and `keepblocks` is a vector of booleans that is updated with whether to truncate or not
"""
function checkzeroblocks!(newblocks::Array{Array{W,2},1},keepblocks::Array{Bool,1}) where W <: Number
  if length(newblocks) > 1
    @inbounds for q = 1:length(keepblocks)
      if keepblocks[q]
        w = 0
        allzero = true
        @inbounds while allzero && w < length(newblocks[q])
          w += 1
          allzero &= isapprox(newblocks[q][w],0)
        end
        keepblocks[q] &= !allzero
      end
    end
  end
  nothing
end




"""
    A = changeblock(Qt,Linds,Rinds[,zeroblocks=false,minelements=parallel_trigger])

changes quantum blocks to `Linds` and `Rinds` of `qarray` `Qt` to an equiavlent `qarray` `A`; `zeroblocks` either keep (true) or removes (false) blocks with all zeros; `minelements` is the threshold to use parallelization in the function
"""
function changeblock(Qt::Qtens{W,Q},Linds::Array{P,1},Rinds::Array{P,1};minelements::Integer=parallel_trigger,zeroblocks::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  return changeblock(Qt,(Linds,Rinds),zeroblocks=zeroblocks,minelements=minelements)
end

"""
    A = changeblock(Qt,newblock[,zeroblocks=false,minelements=parallel_trigger])

changes quantum blocks to `newblocks`=`[Lind,Rind]` with two vectors `Linds` and `Rinds` of `qarray` `Qt` to an equiavlent `qarray` `A`; `zeroblocks` either keep (true) or removes (false) blocks with all zeros; `minelements` is the threshold to use parallelization in the function
"""
function changeblock(Qt::Qtens{W,Q},newblock::Array{Array{P,1},1};minelements::Integer=parallel_trigger,zeroblocks::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  return changeblock(Qt,(newblock[1],newblock[2]),zeroblocks=zeroblocks,minelements=minelements)
end

"""
    parallel_trigger

size of a quantum number block where it becomes worth it to use parallelization methods for changeblock function
"""
const parallel_trigger = 10_000 #set as of DMRjulia v0.8.6

"""
    A = changeblock(Qt,newblock[,zeroblocks=false,minelements=parallel_trigger])

changes quantum blocks to `newblocks`=`(Lind,Rind)` (tuple) with two vectors `Linds` and `Rinds` of `qarray` `Qt` to an equiavlent `qarray` `A`; `zeroblocks` either keep (true) or removes (false) blocks with all zeros; `minelements` is the threshold to use parallelization in the function
"""
function changeblock(Qt::Qtens{W,Q},newblock::NTuple{2,Array{P,1}};minelements::Integer=parallel_trigger,zeroblocks::Bool=false) where {W <: Number, Q <: Qnum, P <: Integer}
  if Qt.currblock == newblock
    newQt = Qt
  else
    checkorder!(Qt)

    Linds = newblock[1]
    Lsize,Lnind,Lnonzero = findsizes(Qt,Linds)
    Rinds = newblock[2]
    Rsize,Rnind,Rnonzero = findsizes(Qt,Rinds)

    ninds = Lnind + Rnind

    LR = Lsize <= Rsize
    QNsummary,leftSummary,rightSummary,newQblocksum = makeQNsummaries(Qt,Linds,Rinds,LR)
    leftQNs,Lbigtosub,rows,Lindexes,rightQNs,Rbigtosub,columns,Rindexes = QnumList(Qt,Linds,Rinds,leftSummary,rightSummary)

    outQNs = LR ? leftQNs : rightQNs

    Tcheck = 0
    @inbounds @simd for q = 1:length(Qt.T)
      Tcheck += length(Qt.T[q])
    end
    rowcolcheck = 0
    @inbounds @simd for q = 1:length(rows)
      rowcolcheck += rows[q]*columns[q]
    end

    fulltens = Tcheck == rowcolcheck
    newblocks = Array{Array{W,2},1}(undef,length(rows))
    keepblocks = Array{Bool,1}(undef,length(rows))
    Matfct = fulltens ? undefMat : zeros
    @inbounds for q = 1:length(rows)
      keepblocks[q] = rows[q] > 0 && columns[q] > 0
      if keepblocks[q]
        x,y = rows[q],columns[q]
        newblocks[q] = Matfct(W,x,y)
      end
    end

    reblock!(newblocks,Qt,outQNs,LR,Linds,Rinds,Lbigtosub,Rbigtosub,minelements=minelements)

    if !zeroblocks
      checkzeroblocks!(newblocks,keepblocks)
    end

    outTens = Array{Array{W,2},1}(undef,sum(keepblocks))
    newrowcols = Array{NTuple{2,Array{intType,2}},1}(undef,length(outTens))

    count = 0
    @inbounds for q = 1:length(keepblocks)
      if keepblocks[q]
        count += 1
        outTens[count] = newblocks[q]
        newrowcols[count] = (Lindexes[q],Rindexes[q])
      end
    end

    if sum(keepblocks) != length(newQblocksum)
      finalQblocksum = newQblocksum[keepblocks]
    else
      finalQblocksum = newQblocksum
    end
    finalblocks = (newblock[1],newblock[2])

    newQt = Qtens{W,Q}(Qt.size,outTens,newrowcols,finalblocks,finalQblocksum,Qt.QnumMat,Qt.QnumSum,Qt.flux)
  end
  return newQt
end
export changeblock

"""
    innerloadpos!(y,Rorigsize,thispos,thiscurrblock_two,thisind_two)

Copies elements of `thisind_two` (matrix) `thispos` (vector, second position given by `y`) according to reordering rule in `thiscurrblock_two`; size of the for loop is `Rorigsize` (i.e., number of columns of `thisind_two` to be copied)
"""
function innerloadpos!(y::P,Rorigsize::P,thispos::Array{P,1},thiscurrblock_two::Array{P,1},thisind_two::Array{P,2}) where P <: Integer
  @inbounds @simd for i = 1:Rorigsize
    thispos[thiscurrblock_two[i]] = thisind_two[i,y]
  end
  nothing
end

"""
    innerloop!(x,y,newblocks,inputval,leftQNs,LRpos,basesize,thispos)

Copies elements of `thisind_two` (matrix) `thispos` (vector, second position given by `y`) according to reordering rule in `thiscurrblock_two`; size of the for loop is `Rorigsize` (i.e., number of columns of `thisind_two` to be copied); updates `newblocks` with new block structure
"""
function innerloop!(x::S,y::S,newblocks::Array{Array{W,2},1},inputval::W,leftQNs::Array{S,1},LRpos::Bool,basesize::NTuple{P,S},thispos::Array{S,1},#thisthread::S,
          thisind_one::Array{S,2},thisind_one_sizetwo::S,Linds::NTuple{Z,S},Lbigtosub::Array{S,1},
          Rinds::NTuple{G,S},Rbigtosub::Array{S,1}) where {P, Z, G, W <: Number, S <: Integer}

  smallx = zeropos2ind(thispos,basesize,Linds)
  smally = zeropos2ind(thispos,basesize,Rinds)
    
  @inbounds newq = leftQNs[LRpos ? smallx : smally]
  @inbounds xval = Lbigtosub[smallx]
  @inbounds yval = Rbigtosub[smally]

  setindex!(newblocks[newq],inputval,xval,yval)
  nothing
end

"""
    doubleloop_right(newblocks,thisTens,leftQNs,LR,basesize,posvecs,thisind_one,thisind_one_sizetwo,Lorigsize,thiscurrblock_one,tup_Linds,Lbigtosub,thisind_two,thisind_two_sizetwo,Rorigsize,thiscurrblock_two,tup_Rinds,Rbigtosub)

copies parameters from one block structure into another, parallelizing over the columns

#Inputs:
+ `newblocks`: an array of new blocks of numbers in each quantum number sector
+ `thisTens`: input tensor block from another choice for the quantum number tensor representation
+ `leftQNs`: quantum number summary for one side of the `Qtens`
+ `LR`: compute the quantities from the left or right side (boolean)
+ `basesize`: size of totally unreshaped tensor
+ `posvecs`: a single vector to store the position of an element in the vector
+ `thisind_one`: positions on side 1 of the quantum number block
+ `thisind_one_sizetwo`: size of the rows in total
+ `Lorigsize`: size of the current block (index 1)
+ `thiscurrblock_one`: reordering rule for position change in reblocking (rows)
+ `tup_Linds`: indices in tuple form for the left group
+ `Lbigtosub`: a vector mapping the rows to their position in the full tensor
+ `thisind_two`: vector of positions for tensor
+ `thisind_two_sizetwo`: size of the columns in total
+ `Rorigsize`: size of the current block (index 2)
+ `thiscurrblock_two`: reordering rule for position change in reblocking (columns)
+ `tup_Rinds`:  indices in tuple form for the right group
+ `Rbigtosub`: a vector mapping the columns to their position in the full tensor
"""
function doubleloop_right(newblocks::Array{Array{W,2},1},thisTens::Array{W,2},leftQNs::Array{S,1},LRpos::Bool,basesize::NTuple{P,S},posvecs::Array{Array{S,1},1},
                          thisind_one::Array{S,2},thisind_one_sizetwo::S,Lorigsize::S,thiscurrblock_one::Array{S,1},Linds::NTuple{Z,S},Lbigtosub::Array{S,1},
                          thisind_two::Array{S,2},thisind_two_sizetwo::S,Rorigsize::S,thiscurrblock_two::Array{S,1},Rinds::NTuple{G,S},Rbigtosub::Array{S,1}) where {P, Z, G, W <: Number, S <: Integer}

  Threads.@threads for x = 1:thisind_one_sizetwo
    thisthread = Threads.threadid()-1
    thispos = posvecs[thisthread]

    innerloadpos!(x,Lorigsize,thispos,thiscurrblock_one,thisind_one)

    @inbounds for y = 1:thisind_two_sizetwo

      inputval = thisTens[x,y]
      innerloadpos!(y,Rorigsize,thispos,thiscurrblock_two,thisind_two)
      
      innerloop!(x,y,newblocks,inputval,leftQNs,LRpos,basesize,thispos,
              thisind_one,thisind_one_sizetwo,Linds,Lbigtosub,Rinds,Rbigtosub)
    end
  end
  nothing
end

"""
    doubleloop_right(newblocks,thisTens,leftQNs,LR,basesize,posvecs,thisind_one,thisind_one_sizetwo,Lorigsize,thiscurrblock_one,tup_Linds,Lbigtosub,thisind_two,thisind_two_sizetwo,Rorigsize,thiscurrblock_two,tup_Rinds,Rbigtosub)

copies parameters from one block structure into another, parallelizing over the rows

#Inputs:
+ `newblocks`: an array of new blocks of numbers in each quantum number sector
+ `thisTens`: input tensor block from another choice for the quantum number tensor representation
+ `leftQNs`: quantum number summary for one side of the `Qtens`
+ `LR`: compute the quantities from the left or right side (boolean)
+ `basesize`: size of totally unreshaped tensor
+ `posvecs`: a single vector to store the position of an element in the vector
+ `thisind_one`: positions on side 1 of the quantum number block
+ `thisind_one_sizetwo`: size of the rows in total
+ `Lorigsize`: size of the current block (index 1)
+ `thiscurrblock_one`: reordering rule for position change in reblocking (rows)
+ `tup_Linds`: indices in tuple form for the left group
+ `Lbigtosub`: a vector mapping the rows to their position in the full tensor
+ `thisind_two`: vector of positions for tensor
+ `thisind_two_sizetwo`: size of the columns in total
+ `Rorigsize`: size of the current block (index 2)
+ `thiscurrblock_two`: reordering rule for position change in reblocking (columns)
+ `tup_Rinds`:  indices in tuple form for the right group
+ `Rbigtosub`: a vector mapping the columns to their position in the full tensor
"""
function doubleloop_left(newblocks::Array{Array{W,2},1},thisTens::Array{W,2},leftQNs::Array{S,1},LRpos::Bool,basesize::NTuple{P,S},posvecs::Array{Array{S,1},1},
                          thisind_one::Array{S,2},thisind_one_sizetwo::S,Lorigsize::S,thiscurrblock_one::Array{S,1},Linds::NTuple{Z,S},Lbigtosub::Array{S,1},
                          thisind_two::Array{S,2},thisind_two_sizetwo::S,Rorigsize::S,thiscurrblock_two::Array{S,1},Rinds::NTuple{G,S},Rbigtosub::Array{S,1}) where {P, Z, G, W <: Number, S <: Integer}

  Threads.@threads for y = 1:thisind_two_sizetwo

    thisthread = Threads.threadid()-1

    thispos = posvecs[thisthread]

    innerloadpos!(y,Rorigsize,thispos,thiscurrblock_two,thisind_two)

    @inbounds for x = 1:thisind_one_sizetwo

      inputval = thisTens[x,y]
      innerloadpos!(x,Lorigsize,thispos,thiscurrblock_one,thisind_one)
      
      innerloop!(x,y,newblocks,inputval,leftQNs,LRpos,basesize,thispos,
              thisind_one,thisind_one_sizetwo,Linds,Lbigtosub,Rinds,Rbigtosub)
    end
  end
  nothing
end

"""
    doubleloop_right(newblocks,thisTens,leftQNs,LR,basesize,posvecs,thisind_one,thisind_one_sizetwo,Lorigsize,thiscurrblock_one,tup_Linds,Lbigtosub,thisind_two,thisind_two_sizetwo,Rorigsize,thiscurrblock_two,tup_Rinds,Rbigtosub)

copies parameters from one block structure into another, parallelizing nothing

#Inputs:
+ `newblocks`: an array of new blocks of numbers in each quantum number sector
+ `thisTens`: input tensor block from another choice for the quantum number tensor representation
+ `leftQNs`: quantum number summary for one side of the `Qtens`
+ `LR`: compute the quantities from the left or right side (boolean)
+ `basesize`: size of totally unreshaped tensor
+ `posvecs`: a single vector to store the position of an element in the vector
+ `thisind_one`: positions on side 1 of the quantum number block
+ `thisind_one_sizetwo`: size of the rows in total
+ `Lorigsize`: size of the current block (index 1)
+ `thiscurrblock_one`: reordering rule for position change in reblocking (rows)
+ `tup_Linds`: indices in tuple form for the left group
+ `Lbigtosub`: a vector mapping the rows to their position in the full tensor
+ `thisind_two`: vector of positions for tensor
+ `thisind_two_sizetwo`: size of the columns in total
+ `Rorigsize`: size of the current block (index 2)
+ `thiscurrblock_two`: reordering rule for position change in reblocking (columns)
+ `tup_Rinds`:  indices in tuple form for the right group
+ `Rbigtosub`: a vector mapping the columns to their position in the full tensor
"""
function doubleloop_reg(newblocks::Array{Array{W,2},1},thisTens::Array{W,2},leftQNs::Array{S,1},LRpos::Bool,basesize::NTuple{P,S},posvec::Array{S,1},
                          thisind_one::Array{S,2},thisind_one_sizetwo::S,Lorigsize::S,thiscurrblock_one::Array{S,1},Linds::NTuple{Z,S},Lbigtosub::Array{S,1},
                          thisind_two::Array{S,2},thisind_two_sizetwo::S,Rorigsize::S,thiscurrblock_two::Array{S,1},Rinds::NTuple{G,S},Rbigtosub::Array{S,1}) where {P, Z, G, W <: Number, S <: Integer}
  for x = 1:thisind_one_sizetwo
    innerloadpos!(x,Lorigsize,posvec,thiscurrblock_one,thisind_one)

    @inbounds for y = 1:thisind_two_sizetwo
      inputval = thisTens[x,y]
      innerloadpos!(y,Rorigsize,posvec,thiscurrblock_two,thisind_two)
      
      innerloop!(x,y,newblocks,inputval,leftQNs,LRpos,basesize,posvec,
              thisind_one,thisind_one_sizetwo,Linds,Lbigtosub,Rinds,Rbigtosub)
    end
  end
  nothing
end

"""
    reblock!(newblocks,Qt,leftQNs,LR,Linds,Rinds,Lbigtosub,Rbigtosub[,minelements=parallel_trigger])

function to perform the transfer of elements from one `Qtens` with a particular block to another equivalent `Qtens` with another block structure

#Inputs:
+ `newblocks`: an array of new blocks of numbers in each quantum number sector
+ `Qt`: input `qarray`
+ `leftQNs`: quantum number summary for one side of the `Qtens`
+ `LR`: compute the quantities from the left or right side (boolean)
+ `Linds`: indices for the rows of the blocks from the tensor
+ `Rinds`: indices for the columns of the blocks from the tensor
+ `Lbigtosub`: a vector mapping the rows to their position in the full tensor
+ `Rbigtosub`: a vector mapping the columns to their position in the full tensor

Optional named inputs:
+ `minelements`: size of arrays before using parallelization

See also: [`minelements`](@ref)
"""
function reblock!(newblocks::Array{Array{W,2},1},Qt::Qtens{W,Q},leftQNs::Array{S,1},LR::Bool,
                          Linds::Array{S,1},Rinds::Array{S,1},Lbigtosub::Array{S,1},Rbigtosub::Array{S,1};minelements::Integer=parallel_trigger) where {W <: Number, Q <: Qnum, S <: Integer}
  thiscurrblock_one = Qt.currblock[1]
  thiscurrblock_two = Qt.currblock[2]

  Lorigsize = length(Qt.currblock[1])
  Rorigsize = length(Qt.currblock[2])

  posvec = Array{intType,1}(undef,length(Qt.QnumMat))
  numthreads = Threads.nthreads()
  posvecs = Array{Array{intType,1},1}(undef,numthreads)
  @inbounds @simd for q = 1:numthreads
    posvecs[q] = Array{intType,1}(undef,length(Qt.QnumMat)) #copy(posvec)
  end
                        
  basesize = ntuple(i->length(Qt.QnumMat[i]),length(Qt.QnumMat))
  tup_Linds = ntuple(i->Linds[i],length(Linds))
  tup_Rinds = ntuple(i->Rinds[i],length(Rinds))
  nQNs = length(Qt.ind)
  @inbounds for q = 1:nQNs

    thisTens = Qt.T[q]
    thisind = Qt.ind[q]

    thisind_one_sizetwo = size(thisind[1],2)
    thisind_one = thisind[1]

    thisind_two_sizetwo = size(thisind[2],2)
    thisind_two = thisind[2]

    if max(thisind_two_sizetwo,thisind_one_sizetwo) > minelements && Threads.nthreads() > 1
      if thisind_two_sizetwo < thisind_one_sizetwo
        doubleloop_right(newblocks,thisTens,leftQNs,LR,basesize,posvecs,
                          thisind_one,thisind_one_sizetwo,Lorigsize,thiscurrblock_one,tup_Linds,Lbigtosub,
                          thisind_two,thisind_two_sizetwo,Rorigsize,thiscurrblock_two,tup_Rinds,Rbigtosub)
      else
        doubleloop_left(newblocks,thisTens,leftQNs,LR,basesize,posvecs,
                        thisind_one,thisind_one_sizetwo,Lorigsize,thiscurrblock_one,tup_Linds,Lbigtosub,
                        thisind_two,thisind_two_sizetwo,Rorigsize,thiscurrblock_two,tup_Rinds,Rbigtosub)
      end
    else
      doubleloop_reg(newblocks,thisTens,leftQNs,LR,basesize,posvec,
                        thisind_one,thisind_one_sizetwo,Lorigsize,thiscurrblock_one,tup_Linds,Lbigtosub,
                        thisind_two,thisind_two_sizetwo,Rorigsize,thiscurrblock_two,tup_Rinds,Rbigtosub)
    end
  end
  nothing
end



