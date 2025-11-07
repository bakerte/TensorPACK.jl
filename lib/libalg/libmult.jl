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
const main_use_strassen = false

function libmult(transA::AbstractChar,transB::AbstractChar,A::TensType,B::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer;use_strassen::Bool=main_use_strassen,safe_innerdim::Float64=0.5,strass_crossover::Int64=4096,level::Int64=1)
  alpha = typeof(eltype(A)(1)*eltype(B)(1))(1)
  return libmult(transA,transB,alpha,A,B,Lsize,innersizeL,innersizeR,Rsize,use_strassen=use_strassen,safe_innerdim=safe_innerdim,strass_crossover=strass_crossover,level=level)
end

function strassen_choice(transA::AbstractChar,transB::AbstractChar,A::TensType,B::TensType,safe_innerdim::Float64=0.5,strass_crossover::Int64=4096)
  if transA == 'N' && transB == 'N'
    return ((size(A,1) >= strass_crossover || size(B,2) >= strass_crossover) && size(A,2) > size(A,1)*safe_innerdim)
  elseif transA != 'N' && transB == 'N'
    return ((size(A,2) >= strass_crossover || size(B,2) >= strass_crossover) && size(A,1) > size(A,2)*safe_innerdim)
  elseif transA == 'N' && transB != 'N'
    return ((size(A,1) >= strass_crossover || size(B,1) >= strass_crossover) && size(A,2) > size(A,1)*safe_innerdim)
  else
    return ((size(A,2) >= strass_crossover || size(B,1) >= strass_crossover) && size(A,1) > size(A,2)*safe_innerdim)
  end
end

function libmult(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::TensType,B::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer;use_strassen::Bool=main_use_strassen,safe_innerdim::Float64=0.5,strass_crossover::Int64=4096,level::Int64=1)
  if use_strassen
    if falsestrassen_choice(transA,transB,A,B,safe_innerdim,strass_crossover)
      return tens(StrassOPen.strassen(transA,transB,alpha,Matrix(reshape(A.T, (Lsize, innersizeL))),Matrix(reshape(B.T, (innersizeR, Rsize))),n=level)).T
    else
      return matmul(transA,transB,alpha,A,B,Lsize,innersizeL,innersizeR,Rsize)
    end
  else
    return matmul(transA,transB,alpha,A,B,Lsize,innersizeL,innersizeR,Rsize)
  end
end

function libmult(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::TensType,B::TensType,beta::Number,Z::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer;use_strassen::Bool=main_use_strassen,safe_innerdim::Float64=0.5,strass_crossover::Int64=4096,level::Int64=1)
  if use_strassen
    if strassen_choice(transA,transB,A,B,safe_innerdim,strass_crossover)
      out = tens(StrassOPen.strassen(transA,transB,alpha,Matrix(reshape(A.T, (Lsize, innersizeL))),Matrix(reshape(B.T, (innersizeR, Rsize))),n=level)).T
      tensorcombination!(out,Z,alpha=(typeof(beta)(1),beta))
      return out
    else
      return matmul(transA,transB,alpha,A,B,beta,Z,Lsize,innersizeL,innersizeR,Rsize)
    end
  else
    return matmul(transA,transB,alpha,A,B,beta,Z,Lsize,innersizeL,innersizeR,Rsize)
  end
end


