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

#         +-----------------+
#>--------|  Check contract |------<
#         +-----------------+
"""
   checkcontract(A,iA,B,iB[,Z,alpha=1,beta=1])

Returns nothing if contraction between two `TensType` is valid
"""
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
