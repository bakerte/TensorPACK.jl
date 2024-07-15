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
    checkflux(Qt[,silent=true])

Debug tool: checks all non-zero elements obey flux conditions in Qtensor (`Qt`); print element by element with `silent`
"""
function checkflux(Qt::Qtens{W,Q},;silent::Bool = true) where {W <: Number, Q <: Qnum}
  condition = true

  if length(Qt.T) == 0
    println("WARNING: zero elements detected in tensor")
  end

  #meta-data checked first
  Rsize = Qt.size
  checksizes = [prod(x->length(Qt.QnumMat[x]),Rsize[w]) for w = 1:length(Rsize)]

  subcondition = sum(w->size(Qt,w) - checksizes[w],1:length(Rsize)) != 0
  condition = condition && !subcondition
  if subcondition
    println("size field does not match QnumMat")
  end

  A = Qt
  p = [size(A.T[q]) == (size(A.ind[q][1],2),size(A.ind[q][2],2)) for q = 1:length(A.T)]
  g = [((size(A.ind[q][1],1),size(A.ind[q][2],1))) == (length(A.currblock[1]),length(A.currblock[2])) for q = 1:length(A.T)]
  h = length(A.Qblocksum) == length(A.T)
  subcondition = !(sum(p) == sum(g) == length(A.T) && h)
  condition = condition && !subcondition
  if subcondition
    println("Sector by sector ($(length(A.T)) sectors):")
    println("T sizes ",p)
    println("ind sizes ",g)
    println("sizes of fields is not correct (.T and .ind) and/or length of Qblocksum")
  end

  firstnorm = norm(Qt)
  secondnorm = norm(Array(Qt))
  subcondition = !isapprox(firstnorm,secondnorm)
  condition = condition && !subcondition
  if subcondition
    println(firstnorm," ",secondnorm)
    error("ill-defined position (.ind) fields...did not return detectably same tensor on dense conversion")
  end

  subcondition = length(Qt.currblock[1]) + length(Qt.currblock[2]) != sum(w->length(Qt.size[w]),1:length(Qt.size))
  condition = condition && !subcondition
  if subcondition
    println("currblock is not correct for sizes")
  end


  numQNs = length(Qt.T)
  LQNs = Array{Q,1}(undef,numQNs)
  RQNs = Array{Q,1}(undef,numQNs)
  matchingQNs = Array{Bool,1}(undef,numQNs)

  for q = 1:numQNs
    LQNs[q] = Q()
    for w = 1:length(Qt.currblock[1])
      thispos = Qt.currblock[1][w]
      thisdim = Qt.ind[q][1][w,1] + 1
      LQNs[q] += getQnum(thispos,thisdim,Qt)
    end

    RQNs[q] = Q()
    for w = 1:length(Qt.currblock[2])
      thispos = Qt.currblock[2][w]
      thisdim = Qt.ind[q][2][w,1] + 1
      RQNs[q] += getQnum(thispos,thisdim,Qt)
    end
    matchingQNs[q] = LQNs[q] + RQNs[q] == Qt.flux
  end

  subcondition = sum(matchingQNs) != numQNs && numQNs > 0
  condition = condition && !subcondition
  if subcondition
    println("not matching quantum numbers...probably issue in defininig (.ind) field in Qtensor")
  end

  subcondition = !(sort(LQNs) == sort([Qt.Qblocksum[q][1] for q = 1:length(Qt.Qblocksum)]))
  condition = condition && !subcondition
  if subcondition
    println(LQNs)
    println([Qt.Qblocksum[q][1] for q = 1:length(Qt.Qblocksum)])
    println("error in left QN block definitions")
  end

  subcondition = !(sort(RQNs) == sort([Qt.Qblocksum[q][2] for q = 1:length(Qt.Qblocksum)]))
  condition = condition && !subcondition
  if subcondition
    println(RQNs)
    println([Qt.Qblocksum[q][2] for q = 1:length(Qt.Qblocksum)])
    println("error in right QN block definitions")
  end


  totalLcheck = Array{Bool,1}(undef,numQNs)
  totalRcheck = Array{Bool,1}(undef,numQNs)
  for q = 1:numQNs
    Lcheck = [true for w = 1:size(Qt.ind[q][1],2)]
    for w = 1:size(Qt.ind[q][1],2)
      checkLQN = Q()
      for x = 1:size(Qt.ind[q][1],1)
        thisrow = Qt.currblock[1][x]
        thisdim = Qt.ind[q][1][x,w]+1
        checkLQN += getQnum(thisrow,thisdim,Qt)
      end
      Lcheck[w] = checkLQN == LQNs[q]
    end
    totalLcheck[q] = sum(Lcheck) == size(Qt.ind[q][1],2)



    Rcheck = [true for w = 1:size(Qt.ind[q][2],2)]
    for w = 1:size(Qt.ind[q][2],2)
      checkRQN = Q()
      for x = 1:size(Qt.ind[q][2],1)
        thisrow = Qt.currblock[2][x]
        thisdim = Qt.ind[q][2][x,w]+1
        checkRQN += getQnum(thisrow,thisdim,Qt)
      end
      Rcheck[w] = checkRQN == RQNs[q]
    end
    totalRcheck[q] = sum(Rcheck) == size(Qt.ind[q][2],2)
  end

  subcondition = sum(totalLcheck) != numQNs
  condition = condition && !subcondition
  if subcondition
    println("wrong quantum number on some rows; quantum numbers: ",totalLcheck)
  end

  subcondition = sum(totalRcheck) != numQNs
  condition = condition && !subcondition
  if subcondition
    println("wrong quantum number on some columns; quantum numbers: ",totalRcheck)
  end



  for q = 1:numQNs
    subcondition = sum(isnan.(Qt.T[q])) > 0
    condition = condition && !subcondition
    if subcondition
      println("element of q = ",q," is not a number")
    end
  end



  if condition
    println("PASSED \n")
  else
    error("problems \n")
  end
  nothing
end
export checkflux

"""
    checkflux(A[,silent=true])

Trivial instantiation of `checkflux` for `densTensType`; useful for debugging purposes
"""
function checkflux(Qt::densTensType;silent::Bool = true)
  nothing
end

