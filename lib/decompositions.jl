#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker and M.P. Thompson (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

#       +---------------------------------------+
#       |                                       |
#>------+            SVD of a tensor            +---------<
#       |                                       |
#       +---------------------------------------+
"""
    U,D,V = libsvd(X)

Chooses the best svd function for tensor decomposition with the standard three tensor output for the SVD
"""
function libsvd end

"""
    U,D,V = libsvd!(X)

Chooses the best svd function for tensor decomposition with the standard three tensor output for the SVD but overwrites input matrix
"""
function libsvd! end

"""
    D,U = libeigen(AA[,B])

Chooses the best eigenvalue decomposition function for tensor `AA` with the standard three tensor output. Can include an overlap matrix for generalized eigenvalue decompositions `B`
"""
function libeigen end

"""
    Q,R = libqr(X[,decomposer=LinearAlgebra.qr])

Decomposes `X` with a QR decomposition.
"""
function libqr end

"""
  L,Q = liblq(X[,decomposer=LinearAlgebra.lq])

Decomposes `X` with a LQ decomposition.
"""
function liblq end

"""
  defzero = 1E-12

default value of zero used in truncating decompositions
"""
const defzero = 1E-12


"""
  newm,sizeD,truncerr,sumD = truncate(D,m,minm,mag,cutoff,effZero,nozeros,power,keepdeg)

Determines the maximum bond dimension to keep for SVD and eigenvalue decompositions based on:
  + The input diagonal matrix `D` (given as a vector here)
  + The maximum bond dimension `m`
  + The minimal bond dimension `minm`
  + The magnitude of the original tensor `mag`
  + The cutoff value `cutoff`
  + `effZero` contains the effective value of zero (default: 1E-16)
  + `nozeros` will remove all values that are approximately zero
  + `power` is the power to which the values of `D` are raised to
  + `keepdeg` is the boolean parameter whether to keep degenerate values or not

The outputs are the size of the new bond (`newm`), size of the input `D` tensor (`sizeD`), truncation error (`truncerr`), and magnitude of the original tensor (`sumD` which is skipped if the input is not 0.)

All parameters can be set in `svd` or `eigen` or similar.
"""
function truncate(D::Array{W,1}...;m::Integer=0,minm::Integer=2,mag::Float64=0.,cutoff::Real=0.,effZero::Real=defzero,nozeros::Bool=true,power::Number=2,keepdeg::Bool=true,rev::Bool=false) where W <: Number

  nQNs = length(D)
  sizeD = 0
  @inbounds @simd for q = 1:nQNs
    sizeD += length(D[q])
  end

  if isapprox(mag,0.)
    sumD = 0
    for q = 1:nQNs
      @inbounds @simd for w = 1:length(D[q])
        sumD += abs(D[q][w])^power
      end
    end
  else
    sumD = mag
  end
  sumD = sumD^(1/power)

  if m == 0 && cutoff == 0.
    m_intervals = [1:length(D[q]) for q = 1:nQNs]
    truncerr = 0.
  else

    bigD = Array{typeof(1.0),1}(undef,sizeD)

    counter = 0
    for q = 1:nQNs
      @inbounds @simd for w = 1:length(D[q])
        counter += 1
        bigD[counter] = abs(D[q][w])^power
      end
    end
    
    checksorted = true
    g = 0
    while checksorted && g < length(bigD)-1
      g += 1
      checksorted &= bigD[g] >= bigD[g+1]
    end

    if checksorted
      order = 1:length(bigD)
      ordered_bigD = bigD
    else
      order = sortperm(bigD,rev = true)
      ordered_bigD = bigD[order]
    end

    pstart,pstop,incr = sizeD,1,-1

    p = pstart

    truncerr = 0.

    if cutoff > 0.
      modcutoff = sumD*cutoff
      @inbounds truncadd = ordered_bigD[p]
      @inbounds while p > pstop && ((truncerr + truncadd < modcutoff) || (nozeros && ordered_bigD[p] < effZero))
        truncerr += truncadd
        p -= 1
        truncadd = ordered_bigD[p]
      end
      if keepdeg
        while p < sizeD && isapprox(ordered_bigD[p],ordered_bigD[p+1])
          p += 1
        end
      end
    end

    thism = m == 0 ? max(min(p,sizeD),minm) : min(max(m,minm),p,sizeD)
    @inbounds @simd for w = p:incr:thism-incr
      truncerr += ordered_bigD[w]
    end

    counts = Array{intType,1}(undef,nQNs) #length(D))
    if nQNs == 1
      counts[1] = thism
    elseif thism < sizeD
      #eliminate the small entries...
      startval = 0
      qranges = Array{UnitRange{intType},1}(undef,nQNs)
      @inbounds @simd for q = 1:nQNs
        qranges[q] = startval+1:startval+length(D[q])
        startval += length(D[q]) #qranges[q][end]
      end


      unfoundval = Array{Bool,1}(undef,nQNs)
      @inbounds @simd for q = 1:nQNs
        unfoundval[q] = true
      end

      @inbounds @simd for q = 1:nQNs
        counts[q] = 0
      end

      @inbounds for z = 1:thism
        r = 1
        while !(order[z] in qranges[r])
          r += 1
        end
        counts[r] += 1
      end

    else
      @inbounds @simd for q = 1:length(D)
        counts[q] = length(D[q])
      end
    end

    if rev
      m_intervals = [length(D[q])-counts[q]+1:length(D[q]) for q = 1:nQNs]
    else
      m_intervals = [1:counts[q] for q = 1:nQNs]
    end

  end

  return m_intervals,sizeD,truncerr,sumD
end

"""
    U,D,V,truncerr,newmag = svd(AA[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=false])

SVD routine with truncation; accepts Qtensors

# Arguments
- `cutoff::Float64`: the maximum sum of singular values squared that can be discarded
- `m::Int64`: maximum bond dimension (does not truncate if 0)
- `mag::Int64`: magnitude of the tensor (calculated here if 0)
- `nozeros::Bool`: toggles whether singular values that are zero are discarded (false by default)

# Warning:

We recommend not defining `using LinearAlgebra` to avoid conflicts.  Instead, define
```
import LinearAlgebra
```
and define functions as `LinearAlgebra.svd` to use functions from that package.

"""
function svd(AA::Array{W,G};cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,a::Integer = size(AA,1),b::Integer=size(AA,2),
              minm::Integer=2,nozeros::Bool=true,power::Number=2,effZero::Real=defzero,keepdeg::Bool=false,inplace::Bool=false,
              decomposer::Function=libsvd) where {W <: Number, G}
    U,D,V,truncerr,sumD = svd(tens(AA),power=power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,effZero=effZero,keepdeg=keepdeg,a=a,b=b,decomposer=decomposer)

    return makeArray(U),D,makeArray(V),truncerr,sumD
end
export svd

function svd(AA::tens{W};power::Number=2,cutoff::Float64 = 0.,
          m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=true,
          effZero::Number=defzero,keepdeg::Bool=false,decomposer::Function=libsvd,
          a::Integer = size(AA,1),b::Integer=size(AA,2),inplace::Bool=false) where W <: Number

  U,D,Vt = decomposer(AA.T,a,b)

  m_intervals,sizeD,truncerr,sumD = truncate(D,m=m,minm=minm,mag=mag,cutoff=cutoff,effZero=effZero,nozeros=nozeros,power=power,keepdeg=keepdeg)

  interval = m_intervals[1]
  thism = length(interval)

  if thism < minm #&& m > minm
    maxm = max(m,minm)

    Utrunc = Array{eltype(U),1}(undef,a*maxm)
    @inbounds @simd for z = 1:length(U)
      Utrunc[z] = U[z]
    end
    @inbounds @simd for z = length(U)+1:length(Utrunc)
      Utrunc[z] = 0
    end

    Dtrunc = Array{eltype(D),1}(undef,maxm)
    @inbounds @simd for z = 1:size(D,1)
      Dtrunc[z] = D[z]
    end
    @inbounds @simd for z = size(D,1)+1:maxm
      Dtrunc[z] = 0
    end

    Vtrunc = Array{eltype(Vt),1}(undef,maxm*b)
    for y = 1:b
      thisind = sizeD*(y-1)
      thisotherind = maxm*(y-1)
      @inbounds @simd for x = 1:sizeD
        Vtrunc[x + thisotherind] = Vt[x + thisind]
      end
      @inbounds @simd for x = sizeD+1:maxm
        Vtrunc[x + thisotherind] = 0
      end
    end

    Utrunc = tens([a,length(Dtrunc)],Utrunc)
    Vtrunc = tens([length(Dtrunc),b],Vtrunc)

  elseif thism < sizeD
    Vtrunc = tens([length(D),b],Vt)
    Vtrunc = Vtrunc[interval,:]

    Dtrunc = D
    for w = length(D):-1:length(interval)+1
      pop!(Dtrunc)
      for x = 1:a
        pop!(U)
      end
    end
    Utrunc = tens([a,length(interval)],U)
#      Utrunc = U
  else
    Utrunc,Dtrunc,Vtrunc = tens([a,length(D)],U),D,tens([length(D),b],Vt)
  end

  return Utrunc,Diagonal(Dtrunc),Vtrunc,truncerr,sumD
end

function svd!(AA::densTensType;power::Number=2,cutoff::Float64 = 0.,
          m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=false,
          effZero::Number=defzero,keepdeg::Bool=false,decomposer::Function=libsvd!,
          a::Integer = size(AA,1),b::Integer=size(AA,2))
  return svd(AA,power=power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,
                  effZero=effZero,keepdeg=keepdeg,a=a,b=b,decomposer=decomposer)
end

"""
  rAA,Lsizes,Rsizes = getorder(AA,vec)

Obtains the present state of an input tensor `AA` grouped into two groups `vec` (i.e., [[1,2],[3,4,5]] or any order). The output is the reshaped tensor and sizes of each dimension for rows and columns.
"""
function getorder(AA::densTensType,vecA::Array{Array{W,1},1}) where W <: Integer

  order = Array{W,1}(undef,sum(a->length(vecA[a]),1:length(vecA)))
  counter = 0
  for b = 1:length(vecA)
    @inbounds @simd for j = 1:length(vecA[b])
      counter += 1
      order[counter] = vecA[b][j]
    end
  end

  a = 1
  @inbounds @simd for w = 1:length(vecA[1])
    a *= size(AA,vecA[1][w])
  end
  b = 1
  @inbounds @simd for w = 1:length(vecA[2])
    b *= size(AA,vecA[2][w])
  end

  if issorted(order)
    AB = AA
  else
    AB = permutedims(AA,order)
  end
  rAA = typeof(AA) <: denstens || ndims(AB) == 2 ? AB : reshape(AB,a,b)

  Lsizes = [size(AA,vecA[1][i]) for i = 1:length(vecA[1])] 
  Rsizes = [size(AA,vecA[2][i]) for i = 1:length(vecA[2])] 
  return rAA,Lsizes,Rsizes,a,b
end

function getorder(AA::qarray,vecA::Array{Array{W,1},1}) where W <: Integer
  order = Array{W,1}(undef,sum(a->length(vecA[a]),1:length(vecA)))
  counter = 0
  for b = 1:length(vecA)
    @inbounds @simd for j = 1:length(vecA[b])
      counter += 1
      order[counter] = vecA[b][j]
    end
  end

  if issorted(order)
    rAA = reshape(AA,vecA)
  else
    AB = permutedims(AA,order)
    Lvec = [i for i = 1:length(vecA[1])]
    Rvec = [i + length(vecA[1]) for i = 1:length(vecA[2])]
    rAA = reshape(AB,[Lvec,Rvec]) #should always be rank-2 here
  end

  a = 1
  @inbounds @simd for w = 1:length(vecA[1])
    a *= size(AA,vecA[1][w])
  end
  b = 1
  @inbounds @simd for w = 1:length(vecA[2])
    b *= size(AA,vecA[2][w])
  end

  Lsizes = [size(AA,vecA[1][i]) for i = 1:length(vecA[1])]#ntuple(i->size(AA,vecA[1][i]),length(vecA[1]))
  Rsizes = [size(AA,vecA[2][i]) for i = 1:length(vecA[2])]#ntuple(i->size(AA,vecA[2][i]),length(vecA[2]))
  return rAA,Lsizes,Rsizes,a,b
end

function findsize(AA::TensType,vec::Array{W,1}) where W <: Number
  a = 1
  @inbounds @simd for w = 1:length(vec)
    a *= size(AA,vec[w])
  end
  return a
end

"""
    U,D,V,truncerr,mag = svd(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

Reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2 with the elements representing the grouped indices for the left and right sets of the SVD for use in unreshaping later
"""
function svd(AA::TensType,vecA::Array{Array{W,1},1};a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2]),
            cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=false,
            power::Number=2,keepdeg::Bool=false,decomposer::Function=libsvd) where {W <: Integer}
  AB,Lsizes,Rsizes,a,b = getorder(AA,vecA)

  U,D,V,truncerr,newmag = svd(AB,a=a,b=b,power = power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg=keepdeg,decomposer=decomposer)

  outU = unreshape!(U,Lsizes...,size(D,1))
  outV = unreshape!(V,size(D,2),Rsizes...)
  return outU,D,outV,truncerr,newmag
end

function svd!(AA::TensType,vecA::Array{Array{W,1},1};a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2]),
            cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=false,
            power::Number=2,keepdeg::Bool=false) where {W <: Integer}
  return svd(AA,vecA,a=a,b=b,power = power,cutoff=cutoff,m=m,mag=mag,minm=minm,
                nozeros=nozeros,keepdeg=keepdeg,decomposer=libsvd!)
end
export svd!

"""
  svdvals(A)

Returns singular values of input tensor `A`
"""
@inline function svdvals(A::AbstractVecOrMat)
  return LinearAlgebra.svdvals(A)
end
export svdvals

"""
  svdvals!(A)

Returns singular values of input tensor `A`, modifying the tensor
"""
function svdvals(A::denstens)
  U,B,V = libsvd(A,job='N') 
  return B
end
export svdvals!

"""
    D,U = eigen(AA[,B,cutoff=,m=,mag=,minm=,nozeros=])

Eigensolver routine with truncation (output to `D` and `U` just as `LinearAlgebra`'s function); accepts Qtensors; arguments similar to `svd`; can perform generalized eigenvalue problem with `B` overlap matrix

See also: [`svd`](@ref)
"""
function eigen(AA::Array{W,G},B::Array{W,R}...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,a::Integer=size(AA,1),b::Integer=size(AA,2),minm::Integer=2,nozeros::Bool=false,power::Number=1,effZero::Real=defzero,keepdeg::Bool=false,transpose::Bool=false,decomposer::Function=libeigen,rev::Bool=true) where {W <: Number, G, R}

  Dsq,U = decomposer(AA,a,B...)

  U = reshape!(U,a,b)

  m_intervals,sizeD,truncerr,sumD = truncate(Dsq,m=m,minm=minm,mag=mag,cutoff=cutoff,effZero=effZero,nozeros=nozeros,power=power,keepdeg=keepdeg,rev=rev)

  Dtrunc,Utrunc = Dsq,U

  interval = m_intervals[1]
  thism = length(interval)
  if thism < minm && m > minm
    maxm = max(m,minm)

    Utrunc = Array{W,2}(undef,a,maxm)
    @inbounds @simd for z = 1:length(U)
      Utrunc[z] = U[z]
    end
    @inbounds @simd for z = length(U)+1:length(Utrunc)
      Utrunc[z] = 0
    end

    Dtrunc = Array{eltype(Dsq),1}(undef,maxm)
    @inbounds @simd for z = 1:size(D,1)
      Dtrunc[z] = D[z]
    end
    @inbounds @simd for z = size(D,1)+1:maxm
      Dtrunc[z] = 0
    end

  elseif thism < sizeD
    Utrunc = U[:,interval]
    Dtrunc = Dsq[interval]
  end

  if transpose
    Utrunc = permutedims(Utrunc,[2,1])
    if eltype(Utrunc) <: Complex
      conj!(Utrunc)
    end
  end

  return Diagonal(Dtrunc),Utrunc,truncerr,sumD
end
export eigen

function eigen(AA::tens{W},B::tens{W}...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,a::Integer=size(AA,1),b::Integer=size(AA,2),minm::Integer=2,nozeros::Bool=false,power::Number=1,effZero::Real=defzero,keepdeg::Bool=false,transpose::Bool=false,decomposer::Function=libeigen,rev::Bool=true) where {W <: Number}

  Dsq,U,truncerr,sumD = eigen(AA.T,B...,cutoff=cutoff,m=m,mag=mag,a=a,b=b,minm=minm,nozeros=nozeros,power=power,effZero=effZero,keepdeg=keepdeg,transpose=transpose,decomposer=decomposer,rev=rev)

  if transpose
    tensU = reshape!(tens(U),size(Dsq,1),b)
  else
    tensU = reshape!(tens(U),a,size(Dsq,1))
  end

  return Dsq,tensU,truncerr,sumD
end

function eigen!(AA::Union{Array{W,2},tens{W}},B::Array{W,2}...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,
                minm::Integer=2,nozeros::Bool=false,power::Number=1,effZero::Float64=defzero,keepdeg::Bool=false,a::Integer=size(AA,1),b::Integer=size(AA,2),
                transpose::Bool=false,decomposer::Function=libeigen!,rev::Bool=true) where {W <: Number}
  return eigen(AA,B...,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,power=power,effZero=effZero,a=a,b=b,keepdeg=keepdeg,transpose=transpose,decomposer=decomposer,rev=rev)
end
export eigen!

"""
    D,U,truncerr,mag = eigen(AA,vecA[,B,cutoff=,m=,mag=,minm=,nozeros=])

reshapes `AA` for `eigen` and then unreshapes U matrix on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
"""
function eigen(AA::TensType,vecA::Array{Array{W,1},1},
                B::TensType...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,
                minm::Integer=2,nozeros::Bool=false,power::Number=1,effZero::Float64=defzero,keepdeg::Bool=false,a::Integer=size(AA,1),b::Integer=size(AA,2),
                transpose::Bool=false,decomposer::Function=libeigen,rev::Bool=true) where {W <: Number}
  AB,Lsizes,Rsizes,a,b = getorder(AA,vecA)
  D,U,truncerr,newmag = eigen(AB,B...,a=a,b=b,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg=keepdeg,transpose=transpose,decomposer=decomposer,rev=rev)
  if transpose
    outU = unreshape!(U,size(D,1),Rsizes...)
  else
    outU = unreshape!(U,Lsizes...,size(D,1))
  end
  return D,outU,truncerr,newmag
end


function eigen!(AA::TensType,vecA::Array{Array{W,1},1},B::TensType...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,
  minm::Integer=2,nozeros::Bool=false,power::Number=1,effZero::Float64=defzero,keepdeg::Bool=false,a::Integer=size(AA,1),b::Integer=size(AA,2),
  transpose::Bool=false,decomposer::Function=libeigen!,rev::Bool=true) where {W <: Number}
  return eigen(AA,vecA,B...,cutoff=cutoff,m=m,mag=mag,a=a,b=b,minm=minm,nozeros=nozeros,power=power,effZero=effZero,keepdeg=keepdeg,transpose=transpose,decomposer=decomposer,rev=rev)
end

function eigen(AA::LinearAlgebra.SymTridiagonal{W, Vector{W}}) where W <: Real
  D,U = LinearAlgebra.eigen(AA)
  return Diagonal(D),U
end

import .LinearAlgebra.eigvals
"""
  vec = eigvals(A)

Eigenvalues of input `A` (allowing julia's arrays and `denstens` types) output to a vector `vec`

See also: [`svd`](@ref)
"""
function eigvals(A::Union{Array{W,2},tens{W}}) where W <: Number
  D,U = libeigen(A,job='N')
  return D
end
export eigvals

function eigvals!(A::Union{Array{W,2},tens{W}}) where W <: Number
  D,U = libeigen!(A,job='N')
  return D
end
export eigvals!

function eigvals(alpha::Array{W,1},beta::Array{W,1},n::intType) where W <: Number
  D,U = libeigen(alpha,beta,n,job='N')
  return D
end

function eigvals!(alpha::Array{W,1},beta::Array{W,1},n::intType) where W <: Number
  D,U = libeigen(alpha,beta,n,job='N')
  return D
end

function eigvals(alpha::Array{W,1},beta::Array{W,1}) where W <: Number
  D,U = libeigen(alpha,beta,job='N')
  return D
end

function eigvals!(alpha::Array{W,1},beta::Array{W,1}) where W <: Number
  D,U = libeigen(alpha,beta,job='N')
  return D
end

"""
    Q,R,0.,1. = qr(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

Reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
"""
function qr(AA::TensType,vecA::Array{Array{W,1},1};decomposer::Function=qr,
              a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2])) where W <: Integer
  AB,Lsizes,Rsizes,a,b = getorder(AA,vecA)
  Qmat,Rmat = decomposer(AB,a=a,b=b)

  innerdim = size(Qmat,2)

  outU = unreshape!(Qmat,Lsizes...,innerdim)
  outV = unreshape!(Rmat,innerdim,Rsizes...)
  return outU,outV
end

"""
    Q,R,0.,1. = qr!(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

Reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2. Overwrites some input data from `AA`
"""
function qr!(AA::TensType,vecA::Array{Array{W,1},1};decomposer::Function=qr!,
              a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2])) where W <: Integer
  return qr(AA,vecA,decomposer=decomposer,a=a,b=b)
end


function qr(AA::densTensType;decomposer::Function=libqr,a::Integer=size(AA,1),b::Integer=size(AA,2))
  return decomposer(AA,a,b)
end

function qr!(AA::densTensType;decomposer::Function=libqr!,a::Integer=size(AA,1),b::Integer=size(AA,2))
  return qr(AA,decomposer=decomposer,a=a,b=b)
end
export qr,qr!

"""
    L,Q,0.,1. = lq(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

Reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2
"""
function lq(AA::TensType,vecA::Array{Array{W,1},1};a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2])) where W <: Integer
  return qr(AA,vecA,decomposer=lq,a=a,b=b)
end

"""
    L,Q,0.,1. = lq!(AA,vecA[,cutoff=,m=,mag=,minm=,nozeros=])

Reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2. Overwrites some input data from `AA`
"""
function lq!(AA::TensType,vecA::Array{Array{W,1},1};a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2])) where W <: Integer
  return qr(AA,vecA,decomposer=lq!,a=a,b=b)
end

function lq(AA::densTensType;a::Integer=size(AA,1),b::Integer=size(AA,2))
  return qr(AA,decomposer=liblq,a=a,b=b)
end

function lq!(AA::densTensType;a::Integer=size(AA,1),b::Integer=size(AA,2))
  return qr(AA,decomposer=liblq!,a=a,b=b)
end
export lq,lq!

"""
    LTens,RTens,D,truncerr,newmag = polar(A,group=[,*])

*-options from `svd`

Performs a polar decomposition on tensor `A` with grouping `group` (default: [[1,2],[3]] for an MPS); if `left` (default), this returns U*D*U',U'*V else functoin returns U*V',V*D*V' from an `svd`

See also: [`svd`](@ref)
"""
function polar(AA::TensType,group::Array{Array{W,1},1};
                right::Bool=true,cutoff::Float64 = 0.,m::Integer = 0,mag::Float64 = 0.,merge::Bool=true,
                minm::Integer=2,nozeros::Bool=false,keepdeg::Bool=false) where W <: Integer

  U,D,V,truncerr,newmag = svd(AA,group,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg=keepdeg)

  ndimU = ndims(U)
  ndimV = ndims(V)
  #polar decomposition
  if right
    rightTensor = ccontract(V,1,D*V,1)
    leftTensor = contract(U,ndimU,V,1)

    leftTensor = reshape!(leftTensor,[[[i] for i = 1:ndimU-1]...,[i+ndimU-1 for i = 1:ndimV-1]],merge=merge)
    rightTensor = reshape!(rightTensor,[[i for i = 1:ndimV-1],[[i+ndimV-1] for i = 1:ndimV-1]...],merge=merge)
  else
    leftTensor = contractc(U*D,ndimU,U,ndimU)
    rightTensor = contract(U,ndimU,V,1)

    leftTensor = reshape!(leftTensor,[[[i] for i = 1:ndimU-1]...,[i+ndimU-1 for i = 1:ndimU-1]],merge=merge)
    rightTensor = reshape!(rightTensor,[[i for i = 1:ndimU-1],[[i+ndimU-1] for i = 1:ndimV-1]...],merge=merge)
  end

  return leftTensor,rightTensor,D,truncerr,newmag
end
export polar
