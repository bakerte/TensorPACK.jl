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

    #tridiagonal
    function libeigen!(dv::AbstractVector{$elty}, ev::AbstractVector{$elty},n::Integer; job::Char='V', rank::Integer=2,fct::Function=stev!)
      if n < length(dv)
        out = fct(dv[1:n], ev[1:n], n, job=job, rank=rank)
      else
        out = fct(dv, ev, n, job=job, rank=rank)
      end
      return out
    end

    function libeigen!(dv::AbstractVector{$elty}, ev::AbstractVector{$elty};job::Char='V', rank::Integer=2) #stev!
      return libeigen!(dv,ev,length(dv),job=job,rank=rank)
    end

    function libeigen(dv::AbstractVector{$elty}, ev::AbstractVector{$elty},n::Integer; job::Char='V', rank::Integer=2) #stev!
      if n < length(dv)
        out = libeigen!(dv[1:n],ev[1:n],n,job=job,rank=rank)
      else
        out = libeigen!(copy(dv),copy(ev),n,job=job,rank=rank)
      end
      return out
    end

    function libeigen(dv::AbstractVector{$elty}, ev::AbstractVector{$elty};job::Char='V', rank::Integer=2) #stev!
      return libeigen!(copy(dv),copy(ev),length(dv),job=job,rank=rank)
    end

    #symmetric/Hermitian
    function libeigen!(A::Union{AbstractMatrix{$elty},Vector{$elty}},n::Integer;job::Char='V',uplo::Char='U',fct::Function=syev!)
      return fct(A,n,job=job,uplo=uplo)
    end

    function libeigen!(A::AbstractMatrix{$elty};job::Char='V',uplo::Char='U') #syev!
      return libeigen!(A,size(A,1),job=job,uplo=uplo)
    end

    function libeigen!(A::tens{$elty};job::Char='V',uplo::Char='U') #syev!
      return libeigen!(A.T,size(A,1),job=job,uplo=uplo)
    end

    function libeigen!(A::tens{$elty},n::Integer;job::Char='V',uplo::Char='U') #syev!
      return libeigen!(A.T,n,job=job,uplo=uplo)
    end

    function libeigen(A::AbstractMatrix{$elty};job::Char='V',uplo::Char='U') #syev!
      return libeigen!(copy(A),size(A,1),job=job,uplo=uplo)
    end

    function libeigen(A::Union{AbstractMatrix{$elty},Vector{$elty}},a::Integer;job::Char='V',uplo::Char='U') #syev!
      return libeigen!(copy(A),a,job=job,uplo=uplo)
    end

    function libeigen(A::tens{$elty};job::Char='V',uplo::Char='U') #syev!
      return libeigen!(copy(A.T),size(A,1),job=job,uplo=uplo)
    end

    function libeigen(A::tens{$elty},n::Integer;job::Char='V',uplo::Char='U') #syev!
      return libeigen!(copy(A.T),n,job=job,uplo=uplo)
    end


#    const defaulteigen = ggev!

    #generalized
    function libeigen!(A::AbstractMatrix{$elty}, n::Integer, B::AbstractMatrix{$elty};jobvl::AbstractChar='V', jobvr::AbstractChar='V',fct::Function=ggev!)
      return fct(A,n,B,jobvl=jobvl,jobvr=jobvr)
    end
  end
end

