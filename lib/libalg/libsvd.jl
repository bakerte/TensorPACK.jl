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

for elty in (:Float32,:Float64,:ComplexF32,:ComplexF64)
  @eval begin

const inplaceChar = 'O'
#const defaultsvd = gesvd! #gesdd!

    function libsvd!(A::AbstractArray{$elty,N},m::Integer,n::Integer;job::Char=inplaceChar,fct::Function=gesvd!) where N
      return fct(A,m,n,job=job)
    end

    function libsvd!(A::tens{$elty},m::Integer,n::Integer;job::Char=inplaceChar,fct::Function=gesvd!)
      return fct(A,m,n,job=job)
    end

    function libsvd!(A::AbstractMatrix{$elty};job::Char=inplaceChar)
        return libsvd!(A,size(A,1),size(A,2),job=job)
    end

    function libsvd!(A::tens{$elty};job::Char=inplaceChar)
        return libsvd!(A.T,size(A,1),size(A,2),job=job)
    end

    function libsvd(A::Array{$elty,N},m::Integer,n::Integer;job::Char='S') where N #LinearAlgebra.LAPACK.gesdd!
        return libsvd!(copy(A),m,n,job=job)
    end

    function libsvd(A::tens{$elty},m::Integer,n::Integer;job::Char='S') #LinearAlgebra.LAPACK.gesdd!
      return libsvd!(copy(A.T),m,n,job=job)
    end

    function libsvd(A::AbstractMatrix{$elty};job::Char='S')
        return libsvd!(copy(A),size(A,1),size(A,2),job=job)
    end

    function libsvd(A::tens{$elty};job::Char='S')
        return libsvd!(copy(A.T),size(A,1),size(A,2),job=job)
    end
      
    #generalized
    function libsvd!(A::AbstractMatrix{$elty}, m::Integer, n::Integer, B::AbstractMatrix{$elty};jobu::AbstractChar='S', jobv::AbstractChar='S', jobq::AbstractChar='Q',fct::Function=gesvd!)
      return fct(A, m, n, B;jobu=jobu, jobv=jobv, jobq=jobq)
    end
  end
end