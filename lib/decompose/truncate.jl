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
    newm,sizeD,truncerr,sumD = truncate(D[,m=0,minm=2,mag=0.,cutoff=0.,effZero=defzero,nozeros=true,power=2,keepdeg=true,rev=true])

Determines the maximum bond dimension to keep for SVD and eigenvalue decompositions based on:
+ `D`: Input eigenvalues (vector) [can be multiple inputs]
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `nozeros`: removes all zero values from the decomposition
+ `power`: power of the eigenvalues to perform truncation (default: L-1 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `rev`: reverses direction of eigenvalues for truncation if `true` (greatest to least)

The outputs are the size of the new bond (`newm`), size of the input `D` tensor (`sizeD`), truncation error (`truncerr`), and magnitude of the original tensor (`sumD` which is skipped if the input is not 0.)

All parameters can be set in `svd` or `eigen` or similar

See also: [`svd`](@ref) [`eigen`](@ref) [`defzero`](@ref)
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

  if m == 0 && cutoff == 0. && !nozeros
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

    if sizeD <= 0
      error("input a tensor into svd of small enough weight that all of it was truncated")
    end

    if nozeros
      while ordered_bigD[pstart] < effZero
        pstart -= 1
      end
    end

    p = pstart

    truncerr = 0.

    if cutoff > 0. #|| nozeros
      modcutoff = sumD*cutoff
      @inbounds truncadd = ordered_bigD[p]
      @inbounds while p > pstop && (truncerr + truncadd < modcutoff) #|| (nozeros && ordered_bigD[p] < effZero))
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