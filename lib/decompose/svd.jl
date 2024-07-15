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
    U,D,V,truncerr,newmag = svd(A[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=true,a=size(A,1),b=size(A,2),power=2,leftflux=false,effZero=defzero,keepdeg=false,inplace=false,decomposer=libsvd])

SVD routine with truncation for Julia `Arrays`

#Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `nozeros`: removes all zero values from the decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `power`: power of the eigenvalues to perform truncation (default: L-2 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `decomposer`: library function to perform decomposition
+ `inplace`: option to run an in-place SVD (may not be as stable)

#Outputs:
+ `U`: A unitary matrix
+ `D`: A diagonal matrix containing eigenvalues
+ 'V': A unitary matrix (U*D*V is A)
+ `truncerr`: total truncation error (L-2 norm)
+ `mag`: magnitude of the output tensor
"""
function svd(AA::AbstractArray;cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,a::Integer = size(AA,1),b::Integer=size(AA,2),leftflux::Bool=false,
              minm::Integer=2,nozeros::Bool=true,power::Number=2,effZero::Real=defzero,keepdeg::Bool=false,inplace::Bool=false,
              decomposer::Function=libsvd) #where {W <: Number, G}
    U,D,V,truncerr,sumD = svd(tens(AA),power=power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,effZero=effZero,keepdeg=keepdeg,a=a,b=b,decomposer=decomposer)

    return Array(U),D,Array(V),truncerr,sumD
end
export svd

"""
    U,D,V,truncerr,newmag = svd(A[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=true,a=size(A,1),b=size(A,2),power=2,effZero=defzero,keepdeg=false,inplace=false,decomposer=libsvd,leftflux=false])

SVD routine with truncation for Julia `denstens`

#Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `nozeros`: removes all zero values from the decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `power`: power of the eigenvalues to perform truncation (default: L-2 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `decomposer`: library function to perform decomposition
+ `inplace`: option to run an in-place SVD (may not be as stable)

#Outputs:
+ `U`: A unitary matrix
+ `D`: A diagonal matrix containing eigenvalues
+ 'V': A unitary matrix (U*D*V is A)
+ `truncerr`: total truncation error (L-2 norm)
+ `mag`: magnitude of the output tensor
"""
function svd(AA::denstens;power::Number=2,cutoff::Float64 = 0.,leftflux::Bool=false,
          m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=true,
          effZero::Number=defzero,keepdeg::Bool=false,decomposer::Function=libsvd,
          a::Integer = size(AA,1),b::Integer=size(AA,2),inplace::Bool=false)

  U,D,Vt = decomposer(AA.T,a,b)

  m_intervals,sizeD,truncerr,sumD = truncate(D,m=m,minm=minm,mag=mag,cutoff=cutoff,effZero=effZero,nozeros=nozeros,power=power,keepdeg=keepdeg)

  interval = m_intervals[1]
  thism = length(interval)

  if thism < minm #&& m > minm
    maxm = max(1,minm)

    Utrunc = Array{eltype(U),1}(undef,a*maxm)
    @inbounds @simd for z = 1:a*thism
      Utrunc[z] = U[z]
    end
    @inbounds @simd for z = a*thism+1:length(Utrunc)
      Utrunc[z] = 0
    end

    Dtrunc = Array{eltype(D),1}(undef,maxm)
    @inbounds @simd for z = 1:thism
      Dtrunc[z] = D[z]
    end
    @inbounds @simd for z = thism+1:length(Dtrunc)
      Dtrunc[z] = 0
    end

    Vtrunc = Array{eltype(Vt),1}(undef,maxm*b)
    for y = 1:b
      thisind = length(D)*(y-1)
      thisotherind = maxm*(y-1)
      @inbounds @simd for x = 1:thism
        Vtrunc[x + thisotherind] = Vt[x + thisind]
      end
      @inbounds @simd for x = thism+1:maxm
        Vtrunc[x + thisotherind] = 0
      end
    end

    Utrunc = tens((a,length(Dtrunc)),Utrunc)
    Vtrunc = tens((length(Dtrunc),b),Vtrunc)

  elseif thism < sizeD
    Vtrunc = tens((length(D),b),Vt)
    Vtrunc = Vtrunc[interval,:]

#    Dtrunc = D[1:length(interval)]
#    U = U[:,interval]
    Dtrunc = D
    for w = length(D):-1:length(interval)+1
      pop!(Dtrunc)
      for x = 1:a
        pop!(U)
      end
    end
    Utrunc = tens((a,length(interval)),U)
  else
    Utrunc,Dtrunc,Vtrunc = tens((a,length(D)),U),D,tens((length(D),b),Vt)
  end

  return Utrunc,Diagonal(Dtrunc),Vtrunc,truncerr,sumD
end

"""
    U,D,V,truncerr,newmag = svd!(A[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=true,a=size(A,1),b=size(A,2),power=2,effZero=defzero,keepdeg=false,inplace=true,decomposer=libsvd!,leftflux=false])

in-place SVD routine with truncation for Julia `Arrays` or `denstens`

#Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `nozeros`: removes all zero values from the decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `power`: power of the eigenvalues to perform truncation (default: L-2 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `decomposer`: library function to perform decomposition
+ `inplace`: option to run an in-place SVD (may not be as stable)

#Outputs:
+ `U`: A unitary matrix
+ `D`: A diagonal matrix containing eigenvalues
+ 'V': A unitary matrix (U*D*V is A)
+ `truncerr`: total truncation error (L-2 norm)
+ `mag`: magnitude of the output tensor
"""
function svd!(AA::densTensType;power::Number=2,cutoff::Float64 = 0.,
          m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=true,leftflux::Bool=false,
          effZero::Number=defzero,keepdeg::Bool=false,decomposer::Function=libsvd!,inplace::Bool=true,
          a::Integer = size(AA,1),b::Integer=size(AA,2))
  return svd(AA,power=power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,
                  effZero=effZero,inplace=inplace,keepdeg=keepdeg,a=a,b=b,decomposer=decomposer)
end

"""
    rA,Lsizes,Rsizes = getorder(A,vec)

Obtains the present state of an input tensor `A` grouped into two groups `vec` (i.e., [[1,2],[3,4,5]] or any order). The output is the reshaped tensor and sizes of each dimension for rows and columns; works for `densTensType`

#Outputs:
+ `rA`: reshaped input tensor into a matrix
+ `Lsizes`: left size of the reshaped tensor
+ 'Rsizes': right size of the reshaped tensor
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

"""
    rA,Lsizes,Rsizes = getorder(A,vec)

Obtains the present state of an input tensor `A` grouped into two groups `vec` (i.e., [[1,2],[3,4,5]] or any order). The output is the reshaped tensor and sizes of each dimension for rows and columns; works for `qarray`

#Outputs:
+ `rA`: reshaped input tensor into a matrix
+ `Lsizes`: left size of the reshaped tensor
+ 'Rsizes': right size of the reshaped tensor
"""
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

"""
   findsize(A,vec)

Helper function to find `a` and `b` sizes for input tensor (size of both groups of indices) used for SVD; outputs a single integer
"""
function findsize(AA::TensType,vec::Array{W,1}) where W <: Number
  a = 1
  @inbounds @simd for w = 1:length(vec)
    a *= size(AA,vec[w])
  end
  return a
end

"""
    U,D,V,truncerr,mag = svd(AA,vecA[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=true,power=2,effZero=defzero,keepdeg=false,inplace=true,decomposer=libsvd,leftflux=false,a=findsize(AA,vecA[1]),b=findsize(AA,vecA[2])])

Reshapes `AA` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2 with the elements representing the grouped indices for the left and right sets of the SVD for use in unreshaping later

#Inputs:
+ `A`: Any `TensType` in the library
+ `vecA`: a vector containing two vectors signifying the two groups for the SVD

#Optional named inputs:
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `nozeros`: removes all zero values from the decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `power`: power of the eigenvalues to perform truncation (default: L-2 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `decomposer`: library function to perform decomposition
+ `inplace`: option to run an in-place SVD (may not be as stable)

#Outputs:
+ `U`: A unitary matrix
+ `D`: A diagonal matrix containing eigenvalues
+ 'V': A unitary matrix (U*D*V is A)
+ `truncerr`: total truncation error (L-2 norm)
+ `mag`: magnitude of the output tensor
"""
function svd(AA::TensType,vecA::Array{Array{W,1},1};a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2]),
            cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=true,leftflux::Bool=false,
            power::Number=2,effZero::Number=defzero,
            keepdeg::Bool=false,decomposer::Function=libsvd) where {W <: Integer}
  AB,Lsizes,Rsizes,a,b = getorder(AA,vecA)

  U,D,V,truncerr,newmag = svd(AB,a=a,b=b,power = power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg=keepdeg,decomposer=decomposer,effZero=effZero)

  outU = unreshape!(U,Lsizes...,size(D,1))
  outV = unreshape!(V,size(D,2),Rsizes...)
  return outU,D,outV,truncerr,newmag
end

"""
    U,D,V,truncerr,mag = svd!(AA,vecA[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=true,power=2,effZero=defzero,keepdeg=false,inplace=true,decomposer=libsvd,leftflux=false,a=findsize(AA,vecA[1]),b=findsize(AA,vecA[2])])

Reshapes `AA` for `svd!` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2 with the elements representing the grouped indices for the left and right sets of the SVD for use in unreshaping later; in-place operations

#Inputs:
+ `A`: Any `TensType` in the library
+ `vecA`: a vector containing two vectors signifying the two groups for the SVD

#Optional named inputs:
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `nozeros`: removes all zero values from the decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `power`: power of the eigenvalues to perform truncation (default: L-2 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `decomposer`: library function to perform decomposition
+ `inplace`: option to run an in-place SVD (may not be as stable)

#Outputs:
+ `U`: A unitary matrix
+ `D`: A diagonal matrix containing eigenvalues
+ 'V': A unitary matrix (U*D*V is A)
+ `truncerr`: total truncation error (L-2 norm)
+ `mag`: magnitude of the output tensor
"""
function svd!(AA::TensType,vecA::Array{Array{W,1},1};a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2]),
            cutoff::Float64 = 0.,m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=true,
            power::Number=2,keepdeg::Bool=false) where {W <: Integer}
  return svd(AA,vecA,a=a,b=b,power = power,cutoff=cutoff,m=m,mag=mag,minm=minm,
                nozeros=nozeros,keepdeg=keepdeg,decomposer=libsvd!)
end
export svd!

#       +------------------------+
#>------| Quantum number version |---------<
#       +------------------------+


function makeU(nQN::Integer,keepq::Array{Bool,1},outU::Array{Array{W,2},1},QtensA::Qtens{W,Q},
                finalinds::Array{Array{P,2},1},newqindexL::Array{P,1},newqindexLsum::Array{Q,1},
                leftflux::Bool,Linds::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}

  finalnQN = sum(keepq)
  finalUinds = Array{NTuple{2,Array{P,2}},1}(undef,finalnQN)
  newQblocksum = Array{NTuple{2,Q},1}(undef,finalnQN)
  counter = 0
  for q = 1:nQN
    if keepq[q]
      counter += 1
      left = QtensA.ind[q][1]
      right = finalinds[q]
      finalUinds[counter] = (left,right)
      
      newQblocksum[counter] = (QtensA.Qblocksum[q][1],newqindexLsum[q])
    end
  end
  finalUQnumMat = Array{Array{intType,1},1}(undef,length(Linds)+1)
  @inbounds @simd for q = 1:length(Linds)
    finalUQnumMat[q] = QtensA.QnumMat[Linds[q]]
  end
  finalUQnumMat[end] = newqindexL

  Uflux = leftflux ? QtensA.flux : Q()

  leftinds = [i for i = 1:length(Linds)]
  rightinds = [length(Linds) + 1]
  newUQsize = [leftinds,rightinds]
  newUblocks = (leftinds,rightinds)
  finalUQnumSum = Array{Array{Q,1},1}(undef,length(Linds)+1)
  @inbounds for q = 1:length(Linds)
    finalUQnumSum[q] = QtensA.QnumSum[Linds[q]]
  end
  finalUQnumSum[end] = newqindexLsum

  return Qtens{W,Q}(newUQsize,outU,finalUinds,newUblocks,newQblocksum,finalUQnumMat,finalUQnumSum,Uflux)
end

function makeV(nQN::Integer,keepq::Array{Bool,1},outV::Array{Array{W,2},1},QtensA::Qtens{W,Q},
                finalinds::Array{Array{P,2},1},newqindexR::Array{P,1},newqindexRsum::Array{Q,1},
                leftflux::Bool,Rinds::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer}
       
  finalnQN = sum(keepq)
  finalVinds = Array{NTuple{2,Array{P,2}},1}(undef,finalnQN)
  newQblocksum = Array{NTuple{2,Q},1}(undef,finalnQN)
  counter = 0
  @inbounds for q = 1:nQN
    if keepq[q]
      counter += 1
      left = finalinds[q]
      right = QtensA.ind[q][2]
      finalVinds[counter] = (left,right)

      newQblocksum[counter] = (newqindexRsum[q],QtensA.Qblocksum[q][2])
    end
  end
  finalVQnumMat = Array{Array{intType,1},1}(undef,length(Rinds)+1)
  finalVQnumMat[1] = newqindexR
  for q = 1:length(Rinds)
    finalVQnumMat[q+1] = QtensA.QnumMat[Rinds[q]]
  end


  Vflux = !leftflux ? QtensA.flux : Q()

  leftinds = [1]
  rightinds = [i+1 for i = 1:length(Rinds)]
  newVQsize = [leftinds,rightinds]
  newVblocks = (leftinds,rightinds)

  finalVQnumSum = Array{Array{Q,1},1}(undef,length(Rinds)+1)
  finalVQnumSum[1] = newqindexRsum
  @inbounds for q = 1:length(Rinds)
    finalVQnumSum[q+1] = QtensA.QnumSum[Rinds[q]]
  end
  return Qtens{W,Q}(newVQsize,outV,finalVinds,newVblocks,newQblocksum,finalVQnumMat,finalVQnumSum,Vflux)
end

function makeD(nQN::Integer,keepq::Array{Bool,1},outD::Array{Diagonal{W},1},QtensA::Qtens{W,Q},
                finalinds::Array{Array{P,2},1},newqindexL::Array{P,1},newqindexR::Array{P,1},
                newqindexRsum::Array{Q,1},newqindexLsum::Array{Q,1}) where {W <: Number, Q <: Qnum, P <: Integer}

  finalnQN = sum(keepq)
  finalDinds = Array{NTuple{2,Array{P,2}},1}(undef,finalnQN)
  newQblocksum = Array{NTuple{2,Q},1}(undef,finalnQN)
  counter = 0
  @inbounds @simd for q = 1:nQN
    if keepq[q]
      counter += 1
      left = finalinds[q]
      right = finalinds[q]
      finalDinds[counter] = (left,right)

      newQblocksum[counter] = (newqindexRsum[q],newqindexLsum[q])
    end
  end
  finalDQnumMat = [newqindexR,newqindexL]

  Dflux = Q()
  leftinds = [1]
  rightinds = [2]
  newDQsize = [leftinds,rightinds]
  newDblocks = (leftinds,rightinds)
  finalDQnumSum = [newqindexRsum,newqindexLsum]

  return Qtens{W,Q}(newDQsize,outD,finalDinds,newDblocks,newQblocksum,finalDQnumMat,finalDQnumSum,Dflux)
end

#=
@inline function threeterm(arr::Array{Array{W,2},1};decomposer::Function=safesvd) where W <: Number
  nQN = length(arr)
  newU = Array{Array{W,2},1}(undef,nQN)
  newD = Array{Array{W,1},1}(undef,nQN)
  newV = Array{Array{W,2},1}(undef,nQN)
  for q = 1:nQN
    newU[q],newD[q],newV[q] = decomposer(arr[q])
  end
  return newU,newD,newV
end
=#
"""
    U,D,V,truncerr,newmag = svd(A[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=true,a=size(A,1),b=size(A,2),power=2,leftflux=false,effZero=defzero,keepdeg=false,inplace=false,decomposer=libsvd])

SVD routine with truncation for Julia `Arrays`

#Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `nozeros`: removes all zero values from the decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `power`: power of the eigenvalues to perform truncation (default: L-2 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `decomposer`: library function to perform decomposition
+ `inplace`: option to run an in-place SVD (may not be as stable)

#Outputs:
+ `U`: A unitary matrix
+ `D`: A diagonal matrix containing eigenvalues
+ 'V': A unitary matrix (U*D*V is A)
+ `truncerr`: total truncation error (L-2 norm)
+ `mag`: magnitude of the output tensor
"""
function svd(QtensA::Qtens{W,Q};a::Integer=size(QtensA,1),b::Integer=size(QtensA,2),
              cutoff::Float64 = 0.,m::Integer = 0,minm::Integer=2,nozeros::Bool=true,power::Number=2,leftflux::Bool=false,mag::Float64=0.,
              effZero::Real=defzero,keepdeg::Bool=false,decomposer::Function=libsvd) where {W <: Number, Q <: Qnum}

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
  newV = Array{Array{W,2},1}(undef,nQN)
  @inbounds for q = 1:nQN
    U,newD[q],Vt = decomposer(A.T[q])
    a,b = size(A.T[q])
    sizeD = length(newD[q])
    newU[q] = reshape!(U,a,sizeD)
    newV[q] = reshape!(Vt,sizeD,b)
  end

  m_intervals,sizeD,truncerr,sumD = truncate(newD...,m=m,minm=minm,mag=mag,cutoff=cutoff,effZero=effZero,nozeros=nozeros,power=power,keepdeg=keepdeg)

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
        newU[q] = newU[q][:,m_intervals[q]]

        tempD[q] = Diagonal(newD[q][m_intervals[q]])
        newV[q] = newV[q][m_intervals[q],:]

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
    @inbounds @simd for q = 1:nQN
      lastind += length(m_intervals[q])
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
    outV = newV[keepq]
  else
    outU = newU
    outD = tempD
    outV = newV
  end

  U = makeU(nQN,keepq,outU,A,finalinds,newqindexL,newqindexLsum,leftflux,Linds)
  D = makeD(nQN,keepq,outD,A,finalinds,newqindexL,newqindexR,newqindexRsum,newqindexLsum)
  V = makeV(nQN,keepq,outV,A,finalinds,newqindexR,newqindexRsum,leftflux,Rinds)

  return U,D,V,truncerr,sumD
end

"""
    U,D,V,truncerr,newmag = svd!(A[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=true,a=size(A,1),b=size(A,2),power=2,leftflux=false,effZero=defzero,keepdeg=false,inplace=false,decomposer=libsvd])

in-place SVD routine with truncation for Julia `Arrays`

#Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `nozeros`: removes all zero values from the decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `power`: power of the eigenvalues to perform truncation (default: L-2 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `decomposer`: library function to perform decomposition
+ `inplace`: option to run an in-place SVD (may not be as stable)

#Outputs:
+ `U`: A unitary matrix
+ `D`: A diagonal matrix containing eigenvalues
+ 'V': A unitary matrix (U*D*V is A)
+ `truncerr`: total truncation error (L-2 norm)
+ `mag`: magnitude of the output tensor
"""
function svd!(QtensA::Qtens{W,Q};a::Integer=size(QtensA,1),b::Integer=size(QtensA,2),
              cutoff::Float64 = 0.,m::Integer = 0,minm::Integer=2,nozeros::Bool=true,power::Number=2,leftflux::Bool=false,mag::Float64=0.,
              effZero::Real=defzero,keepdeg::Bool=false) where {W <: Number, Q <: Qnum}
  return svd(QtensA,a=a,b=b,cutoff=cutoff,m=m,minm=minm,nozeros=nozeros,power=power,
                    leftflux=leftflux,mag=mag,effZero=effZero,keepdeg=keepdeg,decomposer=libsvd!)
end

"""
    U,D,V,truncerr,newmag = svd(A,order[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=true,a=size(A,1),b=size(A,2),power=2,leftflux=false,effZero=defzero,keepdeg=false,inplace=false,decomposer=libsvd,name="svdind",leftadd="L",rightadd="R"])

Generates svd decomposition of named tensor `A` according to `order`; same output as regular eigenvalue but with named tensors

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
+ `U`: A unitary matrix
+ `D`: A diagonal matrix containing eigenvalues
+ 'V': A unitary matrix (U*D*V is A)
+ `truncerr`: total truncation error (L-2 norm)
+ `mag`: magnitude of the output tensor
"""
function svd(AA::nametens,order::Array{Array{B,1},1};cutoff::Float64 = 0.,m::Integer = 0,minm::Integer=2,nozeros::Bool=true,power::Number=2,leftflux::Bool=false,mag::Float64=0.,
  effZero::Real=defzero,keepdeg::Bool=false,decomposer::Function=libsvd,name::String="svdind",leftadd::String="L",rightadd::String="R") where B <: String

  left = matchnames(AA,order[1])
  right = matchnames(AA,order[2])

  neworder = Array{intType,1}[left,right]
  leftname = name * leftadd
  rightname = name * rightadd

  U,D,V,truncerr,newmag = svd(AA.N,neworder,power=power,mag=mag,cutoff=cutoff,m=m,nozeros=nozeros,minm=minm,leftflux=leftflux,keepdeg=keepdeg,decomposer=decomposer)

  TNobjU = nametens(U,vcat(AA.names[left],[leftname]))
  TNobjD = nametens(D,[leftname,rightname])
  TNobjV = nametens(V,vcat([rightname],AA.names[right]))

  return TNobjU,TNobjD,TNobjV,truncerr,newmag
end

"""
    U,D,V,truncerr,newmag = svd(A,order[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=true,a=size(A,1),b=size(A,2),power=2,leftflux=false,effZero=defzero,keepdeg=false,inplace=false,decomposer=libsvd,name="svdind",leftadd="L",rightadd="R"])

Generates svd decomposition of directed tensor `A` according to `order`; same output as regular eigenvalue but with named tensors

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
+ `U`: A unitary matrix
+ `D`: A diagonal matrix containing eigenvalues
+ 'V': A unitary matrix (U*D*V is A)
+ `truncerr`: total truncation error (L-2 norm)
+ `mag`: magnitude of the output tensor
"""
function svd(AA::directedtens,order::Array{Array{B,1},1};cutoff::Float64 = 0.,m::Integer = 0,minm::Integer=2,nozeros::Bool=true,power::Number=2,leftflux::Bool=false,mag::Float64=0.,
  effZero::Real=defzero,keepdeg::Bool=false,decomposer::Function=libsvd,name::String="svdind",leftadd::String="L",rightadd::String="R") where B <: String

  left = matchnames(AA,order[1])
  right = matchnames(AA,order[2])

  neworder = Array{intType,1}[left,right]
  leftname = name * leftadd
  rightname = name * rightadd

  U,D,V,truncerr,newmag = svd(AA.N,neworder,power=power,mag=mag,cutoff=cutoff,m=m,minm=minm,nozeros=nozeros,leftflux=leftflux,keepdeg=keepdeg,decomposer=decomposer)

  TNobjU = directedtens(U,vcat(AA.names[left],[leftname]),false,Int8[AA.arrows[left]...,1])
  TNobjD = directedtens(D,[leftname,rightname],false,Int8[-1,1])
  TNobjV = directedtens(V,vcat([rightname],AA.names[right]),false,Int8[-1,AA.arrows[right]...])

  return TNobjU,TNobjD,TNobjV,truncerr,newmag
end

"""
    UD,DV,truncerr,mag = symsvd(A,order[,mag=,cutoff=,m=,name=,leftadd=,rightadd=])

Takes `svd` of `A` according to `order` and returns U*sqrt(D),sqrt(D)*V

See also: [`svd`](@ref)
"""
function symsvd(AA::TNobj,order::Array{Array{B,1},1};cutoff::Float64 = 0.,m::Integer = 0,minm::Integer=2,nozeros::Bool=true,power::Number=2,leftflux::Bool=false,mag::Float64=0.,
  effZero::Real=defzero,keepdeg::Bool=false,name::String="svdind",
                leftadd::String="L",rightadd::String="R") where B <: String

  U,D,V,truncerr,mag = svd(AA,order,power=power,mag=mag,cutoff=cutoff,m=m,minm=minm,nozeros=nozeros,leftflux=leftflux,keepdeg=keepdeg,decomposer=decomposer)
  S1 = sqrt!(D)
  return U*S1,S1*V,truncerr,mag
end
export symsvd




#=
"""
  C = hadamardprod(A,B)

Performs a Hadamard operation on two input `denstens` `A` and `B` to produce `C` (which effectively multiplies all elements in order by each other to make the new matrix)
"""
function hadamardprod(F::tens{W},S::tens{R}) where {W <: Number, R <: Number}
  return tens(S.size,[F[w]*S[w] for w = 1:length(F)])
end
=#
"""
  C = hadamardprod(A,B)

Performs a Hadamard operation on` `A` (`denstens` or `Array`) and `B` (`denstens` or `Array`) to produce `C` (which effectively multiplies all elements in order by each other to make the new matrix)
"""
function hadamardprod(F::Union{AbstractArray,denstens},S::Union{AbstractArray,denstens}) #where {W <: Number, R <: Number}
  return tens(size(S),[F[w]*S[w] for w = 1:length(F)])
end
#=
function hadamardprod(F::tens{W},S::tens{R}) where {W <: Number, R <: Number}
  return tens(size(S),[F[w]*S[w] for w = 1:length(F)])
end

function hadamardprod(F::AbstractArray{W},S::tens{R}) where {W <: Number, R <: Number}
  return tens(size(S),[F[w]*S[w] for w = 1:length(F)])
end

function hadamardprod(F::tens{W},S::AbstractArray{R}) where {W <: Number, R <: Number}
  return tens([size(S,w) for w = 1:ndims(S)],[F[w]*S[w] for w = 1:length(F)])
end
=#
#=
function hadamardprod(F::Array{W,N},S::Array{R,N}) where {N, W <: Number, R <: Number}
  return F .* S
end
=#


"""
  C = hadamardprod(A,B)

Performs a Hadamard operation on two `qarray`s `A` and `B` to produce `C` (which effectively multiplies all elements in order by each other to make the new matrix)
"""
function hadamardprod(F::Qtens{W,Q},S::Qtens{R,Q}) where {W <: Number, R <: Number, Q <: Qnum}
  newF = changeblock(F,S.currblock)

  Aretind,notAretind = (2,1)
  Bretind,notBretind = (1,2)
  commonblocks = matchblocks((false,false),newF,S,ind=(Aretind,Bretind))

  theseQns = matchnames
  for q = 1:length(theseQns)
    for w = 1:length(newF.T[commonblocks[theseQns[q][1]]])
      newF.T[commonblocks[q][1]][w] *=  S.T[commonblocks[q][2]][w]
    end
  end

  return newF #error("HELP!")
end

"""
  C = hadamardprod(A,B)

Performs a Hadamard operation on two `qarray`s `A` and `B` to produce `C` (which effectively multiplies all elements in order by each other to make the new matrix)
"""
function makeF(s::Diagonal)
  F = Float64[i == j ? 0 : 1/(s[j]^2-s[i]^2) for i = 1:length(s), j = 1:length(s)]
  return F
end

"""
    U,D,V,truncerr,newmag = svd(A[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=true,a=size(A,1),b=size(A,2),power=2,leftflux=false,effZero=defzero,keepdeg=false,inplace=false,decomposer=libsvd])

SVD routine with truncation for a `dtens`, giving the outputs with derivatives

#Inputs:
+ `A`: An input `dtens`

#Optional named inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `nozeros`: removes all zero values from the decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable
+ `power`: power of the eigenvalues to perform truncation (default: L-2 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `decomposer`: library function to perform decomposition
+ `inplace`: option to run an in-place SVD (may not be as stable)

#Outputs:
+ `U`: A unitary matrix (`dtens`)
+ `D`: A diagonal matrix containing eigenvalues (`dtens`)
+ 'V': A unitary matrix (U*D*V is A) (`dtens`)
+ `truncerr`: total truncation error (L-2 norm)
+ `mag`: magnitude of the output tensor
"""
function svd(A::dtens;power::Number=2,cutoff::Float64 = 0.,
  m::Integer = 0,mag::Float64=0.,minm::Integer=2,nozeros::Bool=true,
  effZero::Number=defzero,keepdeg::Bool=false,decomposer::Function=libsvd,
  a::Integer = size(A,1),b::Integer=size(A,2),inplace::Bool=false)

  U,D,V,truncerr,sumD = svd(A[0],power=power,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,effZero=effZero,keepdeg=keepdeg,decomposer=decomposer,a=a,b=b,inplace=inplace)

  Uconvec = (1,) #ntuple(w->w,ndims(U)-1)
  Vconvec = (2,) #ntuple(w->w+1,ndims(V)-1)
  Aconvec = (2,) #ntuple(w->w+ndims(A)-length(Vconvec),length(Vconvec))

F = makeF(D)
invD = invmat(D)


rA = reshape(A[1],a,b)



  #dU
  Cterm = contract(rA,Aconvec,V,Vconvec)
  Lterm = ccontract(U,Uconvec,Cterm,Uconvec)

  Rterm = conj(Lterm)
  Rterm = permutedims!(Rterm,[2,1])

  G = hadamardprod(Lterm*D + D*Rterm,F)
#  println(norm(G-checkhadamardprod(Lterm*D + D*Rterm,F)))

  dU = contract(U,ndims(U),G,1)

  dAVS = contractc(rA,Aconvec,invD*V,Vconvec)
  UdAVS = Lterm * invD #contract(U,Uconvec,dAVS,Uconvec)
  UUdAVS = ccontract(U,ndims(U),UdAVS,1,dAVS,alpha=-1)
  dU += UUdAVS

  #dV
  G = hadamardprod(D*Lterm + Rterm*D,F)
#  println(norm(G-checkhadamardprod(D*Lterm + Rterm*D,F)))

  dV = contract(G,ndims(G),V,1,alpha=-1)

  SUdA = ccontract(U*invD,Uconvec,rA,Uconvec)
#  otherVconvec = ntuple(w->w,length(Vconvec))
  SUdAV = contractc(SUdA,Vconvec,V,Vconvec)
  SUdAVV = contract(SUdAV,ndims(SUdAV),V,1,SUdA,alpha=-1)
  dV += SUdAVV#contract(SUdAV,ndims(SUdAV),V,1) #SUdA

  dD = Diagonal([Lterm[w,w] for w = 1:size(Lterm,1)])

  return dtens(U,dU),dtens(D,dD),dtens(V,dV),truncerr,sumD
  #can reassemble dA = dU*dD*dVt...in principle
end
