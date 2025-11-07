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
    D,U,truncerr,mag = eigen(A[,B,cutoff=0.,m=0,mag=0.,a=size(A,1),b=size(A,2),minm=2,leftflux=false,nozeros=false,power=1,effZero=defzero,keepdeg=false,transpose=false,decomposer=libeigen,rev=false])

Eigensolver routine with truncation (output to `D` and `U` just as `LinearAlgebra`'s function) for Julia `Arrays`; arguments similar to `svd`; can perform generalized eigenvalue problem with `B` overlap matrix

#Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `B`: overlap matrix for generalized eigenvalue decompositions
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `nozeros`: removes all zero values from the decomposition
+ `power`: power of the eigenvalues to perform truncation (default: L-1 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `transpose`: `true` returns U' instead of U (with appropriate change in quantum number symmetries)
+ `decomposer`: library function to perform decomposition
+ `rev`: reverses direction of eigenvalues for truncation if `true` (greatest to least)

#Outputs:
+ `D`: A diagonal matrix containing eigenvalues
+ `U`: A unitary matrix (U*D*U' is A)
+ `truncerr`: total truncation error (L-1 norm)
+ `mag`: magnitude of the output tensor

See also: [`svd`](@ref) [`libeigen`](@ref)
"""
function eigen(AA::Union{Array{W,G},Memory{W}},B::Union{Array{W,R},Memory{W}}...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,a::Integer=size(AA,1),b::Integer=size(AA,2),minm::Integer=2,nozeros::Bool=false,leftflux::Bool=false,power::Number=1,effZero::Real=defzero,keepdeg::Bool=false,transpose::Bool=false,decomposer::Function=libeigen,rev::Bool=false) where {W <: Number, G, R}

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
    @inbounds @simd for z = 1:size(Dsq,1)
      Dtrunc[z] = Dsq[z]
    end
    @inbounds @simd for z = size(Dsq,1)+1:maxm
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

"""
    D,U,truncerr,mag = eigen(A[,B,cutoff=0.,m=0,mag=0.,a=size(A,1),b=size(A,2),minm=2,leftflux=false,nozeros=false,power=1,effZero=defzero,keepdeg=false,transpose=false,decomposer=libeigen,rev=true])

Eigensolver routine with truncation (output to `D` and `U` just as `LinearAlgebra`'s function) for `tens`; arguments similar to `svd`; can perform generalized eigenvalue problem with `B` overlap matrix

#Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `B`: overlap matrix for generalized eigenvalue decompositions
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `nozeros`: removes all zero values from the decomposition
+ `power`: power of the eigenvalues to perform truncation (default: L-1 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `transpose`: `true` returns U' instead of U (with appropriate change in quantum number symmetries)
+ `decomposer`: library function to perform decomposition
+ `rev`: reverses direction of eigenvalues for truncation if `true` (greatest to least)

#Outputs:
+ `D`: A diagonal matrix containing eigenvalues
+ `U`: A unitary matrix (U*D*U' is A)
+ `truncerr`: total truncation error (L-1 norm)
+ `mag`: magnitude of the output tensor

See also: [`svd`](@ref)
"""
function eigen(AA::tens{W},B::tens{W}...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,a::Integer=size(AA,1),b::Integer=size(AA,2),minm::Integer=2,nozeros::Bool=false,leftflux::Bool=false,power::Number=1,effZero::Real=defzero,keepdeg::Bool=false,transpose::Bool=false,decomposer::Function=libeigen,rev::Bool=false) where {W <: Number}

  Dsq,U,truncerr,sumD = eigen(AA.T,B...,cutoff=cutoff,m=m,mag=mag,a=a,b=b,minm=minm,nozeros=nozeros,power=power,effZero=effZero,keepdeg=keepdeg,transpose=transpose,decomposer=decomposer,rev=rev)

  if transpose
    tensU = reshape!(tens(U),size(Dsq,1),b)
  else
    tensU = reshape!(tens(U),a,size(Dsq,1))
  end

  return Dsq,tensU,truncerr,sumD
end

"""
    D,U,truncerr,mag = eigen!(A[,B,cutoff=0.,m=0,mag=0.,a=size(A,1),b=size(A,2),minm=2,leftflux=false,nozeros=false,power=1,effZero=defzero,keepdeg=false,transpose=false,decomposer=libeigen!,rev=true])

Eigensolver routine with truncation (output to `D` and `U` just as `LinearAlgebra`'s function) for Julia `Arrays` or `tens` that performs some arguments in place; arguments similar to `svd`; can perform generalized eigenvalue problem with `B` overlap matrix

#Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `B`: overlap matrix for generalized eigenvalue decompositions
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `nozeros`: removes all zero values from the decomposition
+ `power`: power of the eigenvalues to perform truncation (default: L-1 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `transpose`: `true` returns U' instead of U (with appropriate change in quantum number symmetries)
+ `decomposer`: library function to perform decomposition
+ `rev`: reverses direction of eigenvalues for truncation if `true` (greatest to least)

#Outputs:
+ `D`: A diagonal matrix containing eigenvalues
+ `U`: A unitary matrix (U*D*U' is A)
+ `truncerr`: total truncation error (L-1 norm)
+ `mag`: magnitude of the output tensor

See also: [`svd`](@ref)
"""
function eigen!(AA::Union{Array{W,2},tens{W}},B::Array{W,2}...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,
                minm::Integer=2,nozeros::Bool=false,power::Number=1,leftflux::Bool=false,effZero::Float64=defzero,keepdeg::Bool=false,a::Integer=size(AA,1),b::Integer=size(AA,2),
                transpose::Bool=false,decomposer::Function=libeigen!,rev::Bool=false) where {W <: Number}
  return eigen(AA,B...,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,power=power,effZero=effZero,a=a,b=b,keepdeg=keepdeg,transpose=transpose,decomposer=decomposer,rev=rev)
end

"""
    D,U,truncerr,mag = eigen(A,vecA[,B,cutoff=0.,m=0,mag=0.,a=size(A,1),b=size(A,2),minm=2,leftflux=false,nozeros=false,power=1,effZero=defzero,keepdeg=false,transpose=false,decomposer=libeigen,rev=true])

reshapes `AA` for `eigen` and then unreshapes U matrix on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2; accepts `Array` or `tens`

  #Inputs:
+ `A`: Any `TensType` in the library
+ `vecA`: vector containing index groups

#Optional named inputs:
+ `B`: overlap matrix for generalized eigenvalue decompositions
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `nozeros`: removes all zero values from the decomposition
+ `power`: power of the eigenvalues to perform truncation (default: L-1 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `transpose`: `true` returns U' instead of U (with appropriate change in quantum number symmetries)
+ `decomposer`: library function to perform decomposition
+ `rev`: reverses direction of eigenvalues for truncation if `true` (greatest to least)

#Outputs:
+ `D`: A diagonal matrix containing eigenvalues
+ `U`: A unitary matrix (U*D*U' is A)
+ `truncerr`: total truncation error (L-1 norm)
+ `mag`: magnitude of the output tensor
"""
function eigen(AA::TensType,vecA::Array{Array{W,1},1},
                B::TensType...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,
                minm::Integer=2,nozeros::Bool=false,power::Number=1,leftflux::Bool=false,effZero::Float64=defzero,keepdeg::Bool=false,a::Integer=size(AA,1),b::Integer=size(AA,2),
                transpose::Bool=false,decomposer::Function=libeigen,rev::Bool=false) where {W <: Number}
  AB,Lsizes,Rsizes,a,b = getorder(AA,vecA)
  D,U,truncerr,newmag = eigen(AB,B...,a=a,b=b,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg=keepdeg,transpose=transpose,decomposer=decomposer,rev=rev)
  if transpose
    outU = unreshape!(U,size(D,1),Rsizes...)
  else
    outU = unreshape!(U,Lsizes...,size(D,1))
  end
  return D,outU,truncerr,newmag
end

"""
    D,U,truncerr,mag = eigen(A,vecA[,B,cutoff=0.,m=0,mag=0.,a=size(A,1),b=size(A,2),minm=2,leftflux=false,nozeros=false,power=1,effZero=defzero,keepdeg=false,transpose=false,decomposer=libeigen,rev=true])

reshapes `AA` for `eigen` and then unreshapes U matrix on return with some in-place operations; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2; accepts `Array` or `tens`

  Note: tensor network methods often rely on orthogonality throughout the computation, hence the generalized eigenvalue decomposition is not implemented here

  #Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `B`: overlap matrix for generalized eigenvalue decompositions
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `nozeros`: removes all zero values from the decomposition

#Outputs:
+ `D`: A diagonal matrix containing eigenvalues
+ `U`: A unitary matrix (U*D*U' is A)
+ `truncerr`: total truncation error (L-1 norm)
+ `mag`: magnitude of the output tensor
"""
function eigen!(AA::TensType,vecA::Array{Array{W,1},1},B::TensType...;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=false,leftflux::Bool=false,power::Number=1,effZero::Float64=defzero,keepdeg::Bool=false,a::Integer=size(AA,1),b::Integer=size(AA,2),
  transpose::Bool=false,decomposer::Function=libeigen!,rev::Bool=false) where {W <: Number}
  return eigen(AA,vecA,B...,cutoff=cutoff,m=m,mag=mag,a=a,b=b,minm=minm,nozeros=nozeros,power=power,effZero=effZero,keepdeg=keepdeg,transpose=transpose,decomposer=decomposer,rev=rev)
end

"""
   D,U = eigen(A)

`eigen` also accepts the `SymTridiagonal` and this function is an alias for the `LinearAlgebra` implementation
"""
function eigen(AA::LinearAlgebra.SymTridiagonal{W, Vector{W}}) where W <: Real
  D,U = LinearAlgebra.eigen(AA)
  return Diagonal(D),U
end




#       +------------------------+
#>------| Quantum number version |---------<
#       +------------------------+



"""
    D,U,truncerr,mag = eigen!(A[,cutoff=0.,m=0,mag=0.,a=size(A,1),b=size(A,2),minm=2,leftflux=false,nozeros=false,power=1,effZero=defzero,keepdeg=false,transpose=false,decomposer=libeigen!,rev=true])

Eigensolver routine with truncation (output to `D` and `U` just as `LinearAlgebra`'s function) for a `Qtens` tensor; arguments similar to `svd`

Note: tensor network methods often rely on orthogonality throughout the computation, hence the generalized eigenvalue decomposition is not implemented here

#Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `B`: overlap matrix for generalized eigenvalue decompositions
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `nozeros`: removes all zero values from the decomposition
+ `power`: power of the eigenvalues to perform truncation (default: L-1 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `transpose`: `true` returns U' instead of U (with appropriate change in quantum number symmetries)
+ `decomposer`: library function to perform decomposition
+ `rev`: reverses direction of eigenvalues for truncation if `true` (greatest to least)

#Outputs:
+ `D`: A diagonal matrix containing eigenvalues
+ `U`: A unitary matrix (U*D*U' is A)
+ `truncerr`: total truncation error (L-1 norm)
+ `mag`: magnitude of the output tensor

See also: [`svd`](@ref) [`libeigen`](@ref)
"""
function eigen!(QtensA::Qtens{W,Q};cutoff::Float64 = 0.,m::Integer = 0,effZero::Real=defzero,
              minm::Integer=2,nozeros::Bool=false,leftflux::Bool=false,
              power::Number=1,mag::Float64=0.,
              decomposer::Function=libeigen!,keepdeg::Bool=false,transpose::Bool=false,rev::Bool=false) where {W <: Number, Q <: Qnum}

  Tsize = QtensA.size
  Linds = Tsize[1]
  Rinds = Tsize[2]

  A = changeblock(QtensA,Linds,Rinds)
  nQN = length(A.T)

  LRinds = 1

  QNsummary = Array{Q,1}(undef,nQN)
  invQNsummary = Array{Q,1}(undef,nQN)
  @inbounds @simd for q = 1:nQN
    invQNsummary[q] = A.Qblocksum[q][LRinds]
    QNsummary[q] = inv(A.Qblocksum[q][LRinds])
  end
  
  nQN = length(A.T)
  newU = Array{Array{W,2},1}(undef,nQN)
  newD = Array{Array{W,1},1}(undef,nQN)
  for q = 1:nQN
    newD[q],newU[q] = decomposer(A.T[q])
  end

  if transpose
    for q = 1:nQN
      newU[q] = permutedims!(newU[q],[2,1])
      if eltype(newU[q]) <: Complex
        conj!(newU[q])
      end
    end
  end
#=
  println()
  println("start: ",nQN," ",norm(QtensA))
  for q = 1:nQN
    println(q," ",norm(newD[q]))
#    println(newD[q])
  end
  println()
=#
  m_intervals,sizeD,truncerr,sumD = truncate(newD...,m=m,minm=minm,mag=mag,cutoff=cutoff,effZero=effZero,nozeros=nozeros,power=power,keepdeg=keepdeg,rev=rev)

  thism = 0
  @inbounds @simd for q = 1:length(m_intervals)
    thism += length(m_intervals[q])
  end

  newqindexL = Array{intType,1}(undef,max(thism,minm))
  keepq = Array{Bool,1}(undef,nQN)
  tempD = Array{Diagonal{W},1}(undef,nQN)
  finalinds = Array{Array{intType,2},1}(undef,nQN)
  @inbounds for q = 1:nQN
    if length(m_intervals[q]) != 0
      keepq[q] = m_intervals[q][1] != 0 && m_intervals[q][1] <= m_intervals[q][end]

      if keepq[q]
        if transpose
          newU[q] = newU[q][m_intervals[q],:]
        else
          newU[q] = newU[q][:,m_intervals[q]]
        end

        tempD[q] = Diagonal(newD[q][m_intervals[q]])

        offset = 0
        @inbounds @simd for w = 1:q-1
          offset += length(m_intervals[w])
        end

        tempvec = Array{intType,2}(undef,1,length(m_intervals[q]))
        @inbounds @simd for i = 1:length(m_intervals[q])
          tempvec[i] = i + offset - 1
        end
        finalinds[q] = tempvec
        
        @inbounds @simd for i = 1:length(finalinds[q])
          newqindexL[i + offset] = q
        end
      end
    else
      keepq[q] = false
    end
  end

  if length(newqindexL) > thism

    findzeroQN = 1
    zeroQN = Q()
    while findzeroQN < nQN && QNsummary[findzeroQN] != zeroQN
      findzeroQN += 1
    end
    if findzeroQN > nQN
      push!(QNsummary,zeroQN)
      push!(invQNsummary,zeroQN)
    end

    lastind = 0
    @inbounds for q = 1:nQN
      lastind += isassigned(finalinds,q) ? length(finalinds[q]) : 0
    end

    @inbounds @simd for w = lastind+1:length(newqindexL)
      newqindexL[w] = findzeroQN
    end
  end

  newqindexLsum = QNsummary
  newqindexRsum = invQNsummary
  newqindexR = newqindexL

  if sum(keepq) < nQN
    outU = newU[keepq]
    outD = tempD[keepq]
  else
    outU = newU
    outD = tempD
  end

  if transpose
    U = makeV(nQN,keepq,outU,A,finalinds,newqindexR,newqindexRsum,!leftflux,Rinds)
  else
    U = makeU(nQN,keepq,outU,A,finalinds,newqindexL,newqindexLsum,leftflux,Linds)
  end
  D = makeD(nQN,keepq,outD,A,finalinds,newqindexL,newqindexR,newqindexRsum,newqindexLsum)

  return D,U,truncerr,sumD
end

"""
    D,U,truncerr,mag = eigen(A[,cutoff=0.,m=0,mag=0.,a=size(A,1),b=size(A,2),minm=2,leftflux=false,nozeros=false,power=1,effZero=defzero,keepdeg=false,transpose=false,decomposer=libeigen,rev=false])

Eigensolver routine with truncation (output to `D` and `U` just as `LinearAlgebra`'s function) for a `Qtens` tensor; arguments similar to `svd`

#Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `B`: overlap matrix for generalized eigenvalue decompositions
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `nozeros`: removes all zero values from the decomposition
+ `power`: power of the eigenvalues to perform truncation (default: L-1 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `transpose`: `true` returns U' instead of U (with appropriate change in quantum number symmetries)
+ `decomposer`: library function to perform decomposition
+ `rev`: reverses direction of eigenvalues for truncation if `true` (greatest to least)

#Outputs:
+ `D`: A diagonal matrix containing eigenvalues
+ `U`: A unitary matrix (U*D*U' is A)
+ `truncerr`: total truncation error (L-1 norm)
+ `mag`: magnitude of the output tensor

See also: [`svd`](@ref) [`libeigen`](@ref)
"""
function eigen(QtensA::Qtens{W,Q};cutoff::Float64 = 0.,m::Integer = 0,a::Integer=size(QtensA,1),b::Integer=size(QtensA,2),
                minm::Integer=2,nozeros::Bool=false,
                power::Number=1,leftflux::Bool=false,mag::Float64=0.,effZero::Real=defzero,
                decomposer::Function=libeigen,keepdeg::Bool=false,transpose::Bool=false,rev::Bool=false) where {W <: Number, Q <: Qnum}
  return eigen!(QtensA,cutoff=cutoff,m=m,minm=minm,nozeros=nozeros,power=power,effZero=effZero,
                leftflux=leftflux,mag=mag,decomposer=libeigen,keepdeg=keepdeg,transpose=transpose,rev=rev)
end

"""
    D,U,truncerr,newmag = eigen(A,order[,mag=,cutoff=,m=,name=,leftadd=])

Generates eigenvalue decomposition of named tensor `A` according to `order`; same output as regular eigenvalue but with named tensors

# Naming created index:
+ the index to the left of `D` is `vcat(name,leftadd)`

  #Inputs:
+ `A`: Any `TensType` in the library
+ `order`: vector containing index groups given as `String`

#Optional named inputs:
+ `name`: base name of the newly introduced tensor index
+ `leftadd`: string addition for the left index to distinguish from the right
+ `rightadd`: string addition for the right index to distinguish from the left
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `nozeros`: removes all zero values from the decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `power`: power of the eigenvalues to perform truncation (default: L-1 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `transpose`: `true` returns U' instead of U (with appropriate change in quantum number symmetries)
+ `decomposer`: library function to perform decomposition
+ `rev`: reverses direction of eigenvalues for truncation if `true` (greatest to least)

#Outputs:
+ `D`: A diagonal matrix containing eigenvalues
+ `U`: A unitary matrix (U*D*U' is A)
+ `truncerr`: total truncation error (L-1 norm)
+ `mag`: magnitude of the output tensor
"""
function eigen(AA::nametens,order::Array{Array{B,1},1};mag::Number = 0.,cutoff::Number = 0.,
                m::Integer = 0,name::String="eigind",leftadd::String="L",rightadd::String="R",nozeros::Bool=false,
                power::Number=1,leftflux::Bool=false,effZero::Real=defzero,
                decomposer::Function=libeigen,keepdeg::Bool=false,transpose::Bool=false,rev::Bool=false) where B <: String

  D,U,truncerr,newmag = eigen(AA.N,order,mag=mag,cutoff=cutoff,m=m,minm=minm,nozeros=nozeros,power=power,effZero=effZero,leftflux=leftflux,decomposer=libeigen,keepdeg=keepdeg,transpose=transpose,rev=rev)

  leftname = name * leftadd
  rightname = name * rightadd

  TNobjD = nametens(D,[leftname,rightname])
  if transpose
    right = matchnames(AA,order[2])
    TNobjU = nametens(U,vcat([rightname],AA.names[right])) #U'
  else
    left = matchnames(AA,order[1])
    TNobjU = nametens(U,vcat(AA.names[left],[leftname])) #U
  end
  return TNobjD,TNobjU,truncerr,newmag
end

