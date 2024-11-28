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

#         +--------------------------+
#>--------|     library multiply     |---------<
#         +--------------------------+


function libmult(transA::AbstractChar,transB::AbstractChar,A::TensType,B::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer)
  alpha = typeof(eltype(A)(1)*eltype(B)(1))(1)
  return libmult(transA,transB,alpha,A,B,Lsize,innersizeL,innersizeR,Rsize)
end

const strassen_bool = true
const min_strassensize_singlethread = 100
const safe_innerdim = 0.5
const strass_crossover = 2^12

function strassen_choice(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::TensType,B::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer)
  numthreads = Threads.nthreads()
  if numthreads == 1
    Aouterdim = size(A,1) > min_strassensize_singlethread
    Ainnerdim = size(A,2) > min_strassensize_singlethread
    Bouterdim = size(B,1) > min_strassensize_singlethread
    Binnerdim = size(B,2) > min_strassensize_singlethread
    res = Aouterdim && Ainnerdim && Bouterdim && Binnerdim
  else
    if transA == 'N' && transB == 'N'
      res = (size(A,1) > strass_crossover || size(B,2) > strass_crossover) && size(A,2) > size(A,1)*safe_innerdim
    elseif transA != 'N' && transB == 'N'
      res = (size(A,2) > strass_crossover || size(B,2) > strass_crossover) && size(A,1) > size(A,2)*safe_innerdim
    elseif transA == 'N' && transB != 'N'
      res = (size(A,1) > strass_crossover || size(B,1) > strass_crossover) && size(A,2) > size(A,1)*safe_innerdim
    end
  end
  return res
end

function libmult(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::TensType,B::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer)
  if strassen_bool
    out = matmul(transA,transB,alpha,A,B,Lsize,innersizeL,innersizeR,Rsize)
  else
    out = strassen(transA,transB,alpha,A,B,Lsize,innersizeL,innersizeR,Rsize)
  end
  return out
end

function libmult(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::TensType,B::TensType,beta::Number,Z::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer)
  if strassen_bool
    out = matmul(transA,transB,alpha,A,B,beta,Z,Lsize,innersizeL,innersizeR,Rsize)
  else
    out = strassen(transA,transB,alpha,A,B,Lsize,innersizeL,innersizeR,Rsize)
    tensorcombination!(out,Z,alpha=(typeof(beta)(1),beta))
  end
  return out
end


