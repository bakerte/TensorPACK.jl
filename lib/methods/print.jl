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
  showTens(Qt[,show=])

Prints fields of the Qtensor (`Qt`) out to a number of elements `show`; can also be called with `print` or `println`

See also: [`Qtens`](@ref) [`print`](@ref) [`println`](@ref)
"""
function showTens(M::denstens;show::Integer = 4)
  println("printing regular tensor of type: ", typeof(M))
  if typeof(M) <: tens
    println("size = ", M.size)
  end
  maxshow = min(show, size(M.T, 1))
  maxBool = show < size(M.T, 1)
  println("T = ", M.T[1:maxshow], maxBool ? "..." : "")
  nothing
end
export showTens

import Base.print
"""
    print(A[,show=])

Idential to `showQtens`

See also: [`showQtens`](@ref) [`println`](@ref)
"""
function print(A::denstens;show::Integer = 4)
  showTens(A, show = show)
  nothing
end

import Base.println
"""
    println(A[,show=])

Idential to `showQtens`

See also: [`showQtens`](@ref) [`print`](@ref)
"""
function println(A::denstens;show::Integer = 4)
  showTens(A, show = show)
  print("\n")
  nothing
end

import Base.println
"""
    println(A[,show=])

Prints named tensor `A`

# Outputs:
+ `size`: size of `A`
+ `index names`: current names on `A`
+ `arrowss`: fluxes for each index on `A`
+ `elements`: elements of `A` if reshaped into a vector (out to `show`)
"""
function println(A::TNobj;show::Integer=10)

  println("size = ",size(A))
  println("index names = ",A.names)
  if typeof(A.N) <: denstens ||  typeof(A.N) <: qarray
    temp = length(A.N.T)
    maxshow = min(show,temp)
    println("elements = ",A.N.T[1:maxshow])
  else
    rAA = reshape(A.N,prod(size(A)))
    temp = length(rAA)
    maxshow = min(show,temp)
    if length(rAA) > maxshow
      println("elements = ",rAA[1:maxshow],"...")
    else
      println("elements = ",rAA[1:maxshow])
    end
  end
  println()
  nothing
end





"""
    showQtens(Qt[,show=])

Prints fields of the Qtensor (`Qt`) out to a number of elements `show`; can also be called with `print` or `println`

See also: [`Qtens`](@ref) [`print`](@ref) [`println`](@ref)
"""
function showQtens(Qtens::qarray;show::Integer = 4)
  println("printing Qtens of type: ", typeof(Qtens))
  println("size of tensor: ",size(Qtens))
  println("size = ", Qtens.size)
  maxshow = min(show, size(Qtens.T, 1))
  maxBool = show < size(Qtens.T, 1)
  println("block tensor: ")#Qtens.T[1:maxshow], maxBool ? "..." : "")
  if length(Qtens.T) == 0
    println("<null tensor>")
  else
    for q = 1:length(Qtens.T)
      maxshow = min(show, length(Qtens.T[q]))
      maxBool = show < length(Qtens.T[q])
      if typeof(Qtens.T[q]) <: Diagonal
        println("block $q size = ",size(Qtens.T[q]),", ",Qtens.Qblocksum[q],", values  (Diagonal type) = ",[Qtens.T[q][i,i] for i = 1:min(size(Qtens.T[q],1),maxshow)], maxBool ? "..." : "")
      else
        println("block $q size = ",size(Qtens.T[q]),", ",Qtens.Qblocksum[q],", values = ",Qtens.T[q][1:maxshow], maxBool ? "..." : "")
      end
      println("inds: block $q")
      maxshow = min(show, length(Qtens.ind[q][1]))
      maxBool = show < length(Qtens.ind[q][1])
      println("  row: ",Qtens.ind[q][1][1:maxshow], maxBool ? "..." : "")
      maxshow = min(show, length(Qtens.ind[q][2]))
      maxBool = show < length(Qtens.ind[q][2])
      println("  col: ",Qtens.ind[q][2][1:maxshow], maxBool ? "..." : "")
    end
  end
  println("currblock = ",Qtens.currblock)
  println("Qblocksum = ",Qtens.Qblocksum)
  println("QnumMat = ")
  for i = 1:size(Qtens.QnumMat, 1)
    maxshow = min(show, size(Qtens.QnumMat[i], 1))
    maxBool = show < size(Qtens.QnumMat[i], 1)
    println("  ",i, ": ", Qtens.QnumMat[i][1:maxshow], maxBool ? "..." : "")
  end
  println("QnumSum = ")
  for i = 1:size(Qtens.QnumSum, 1)
    maxshow = min(show, size(Qtens.QnumSum[i], 1))
    maxBool = show < size(Qtens.QnumSum[i], 1)
    println("  ",i, ": ", Qtens.QnumSum[i][1:maxshow], maxBool ? "..." : "")
  end
  println("flux = ", Qtens.flux)
  nothing
end
export showQtens

import Base.print
"""
    print(A[,show=])

Idential to `showQtens`

See also: [`showQtens`](@ref) [`println`](@ref)
"""
function print(A::qarray...;show::intType = 4)
  showQtens(A, show = show)
  nothing
end

import Base.println
"""
    println(A[,show=])

Idential to `showQtens`

See also: [`showQtens`](@ref) [`print`](@ref)
"""
function println(A::qarray;show::Integer = 4)
  showQtens(A, show = show)
  print("\n")
  nothing
end


