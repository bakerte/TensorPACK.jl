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

#         +---------------+
#>--------|     mult!     |---------<
#         +---------------+

import .StrassOPen.matmul
import .StrassOPen.matmul!

for (gemm, elty) in
  ((:dgemm_,:Float64),
   (:sgemm_,:Float32),
   (:zgemm_,:ComplexF64),
   (:cgemm_,:ComplexF32))
  @eval begin
        # SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
        # *     .. Scalar Arguments ..
        #       DOUBLE PRECISION ALPHA,BETA
        #       INTEGER K,LDA,LDB,LDC,M,N
        #       CHARACTER TRANSA,TRANSB
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
        #=
    function matmul!(transA::AbstractChar, transB::AbstractChar,
                  alpha::Union{($elty), Bool},
                  A::AbstractArray{$elty,N},
                  B::AbstractArray{$elty,M},
                  beta::Union{($elty), Bool},
                  C::AbstractArray{$elty,G},
                  m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M,G}
        lda = max(1,transA == 'N' ? m : ka)
        ldb = max(1,transB == 'N' ? ka : n)
        ldc = max(1,m)
        ccall((@blasfunc($gemm), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt},
            Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ptr{$elty},
            Ref{BlasInt}, Clong, Clong),
            transA, transB, m, n,
            ka, alpha, A,  lda,
            B, ldb, beta, C,
            ldc, 1, 1)
        C
    end
=#
#=
    function matmul(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty,N},B::AbstractArray{$elty,M},m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M}
        C = Array{($elty),2}(undef,m,n)
        matmul!(transA, transB, alpha, A, B, zero($elty), C,m,ka,kb,n)
        return C
    end

    function matmul(transA::AbstractChar, transB::AbstractChar, A::Union{AbstractArray{$elty,N},tens{$elty}},B::Union{AbstractArray{$elty,M},tens{$elty}},m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M}
        matmul(transA, transB, one($elty), A, B,m,ka,kb,n)
    end
=#

    function matmul(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::tens{$elty},B::tens{$elty},m::Integer,ka::Integer,kb::Integer,n::Integer)
      C = Array{($elty),1}(undef,m*n)
      matmul!(transA, transB, alpha, A.T, B.T, zero($elty), C,m,ka,kb,n)
      return C
    end
    #=
    function matmul(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty,N},B::AbstractArray{$elty,M},beta::($elty),C::AbstractArray{$elty,G},m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M,G}
        newC = copy(C)
        matmul!(transA, transB, alpha, A, B, beta, newC,m,ka,kb,n)
        return newC
    end
=#   
    function matmul(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::tens{$elty},B::tens{$elty},beta::($elty),C::tens{$elty},m::Integer,ka::Integer,kb::Integer,n::Integer)
        newC = copy(C.T)
        matmul!(transA, transB, alpha, A.T, B.T, beta, newC,m,ka,kb,n)
        return newC
    end
    
    function matmul(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::Diagonal{$elty},B::Diagonal{$elty},m::Integer,ka::Integer,kb::Integer,n::Integer)
        if m > 1
          newC = Diagonal([A.T[w]*B.T[w] for w = 1:length(A)])
        else
          newC = dot(A,B,Lfct=transA == 'C' ? adjoint : identity,Rfct = transB == 'C' ? adjoint : identity)
        end
        return newC
    end
  end
end


function matmul(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::TensType,B::Diagonal{W},beta::Number,Z::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer) where W <: Number
  C = dmul(A,B)
  mult!(C,alpha)
  add!(C,Z[1],beta)
  return C
end

function matmul(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::Diagonal{W},B::TensType,beta::Number,Z::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer) where W <: Number
  C = dmul(A,B)
  mult!(C,alpha)
  add!(C,Z[1],beta)
  return C
end

#            +----------------+
#>-----------|    dualnums    |-------------<
#            +----------------+

function matmul(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::Array{W,2},B::Array{W,2},beta::Number,Z::Array{W,2},Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer) where W <: dualnum
  C = dmul(A,B)
  mult!(C,alpha)
  add!(C,Z[1],beta)
  return C
end

function matmul(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::Array{W,2},B::Array{W,2},Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer) where W <: dualnum
  C = Matrix{dualnum{W}}(undef,size(A,1),size(B,2))
  for y = 1:size(A,1)
    for x = 1:size(B,2)
      C[x,y] = dualnum(W)
      for k = 1:size(A,2)
        C[x,y] += A[x,k]*B[k,y]
      end
    end
  end
  return C
end

#=
function matmul(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::diagonal{W},B::TensType,beta::Number,Z::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer) where W <: Number
  C = dmul(A,B)
  mult!(C,alpha)
  add!(C,Z[1],beta)
  return C
end

function matmul(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::TensType,B::diagonal{W},beta::Number,Z::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer) where W <: Number
  C = dmul(A,B)
  mult!(C,alpha)
  add!(C,Z[1],beta)
  return C
end

function matmul(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::diagonal{W},B::TensType,Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer) where W <: Number
  C = dmul(A,B)
  mult!(C,alpha)
  return C
end

function matmul(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::TensType,B::diagonal{W},Lsize::Integer,innersizeL::Integer,innersizeR::Integer,Rsize::Integer) where W <: Number
  C = dmul(A,B)
  mult!(C,alpha)
  return C
end
=#
