#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

"""
  Module: Krylov

Expanding operators in a Krylov subspace
"""

function makeHpsi(Lenv::TensType,Renv::TensType,psi::TensType,invec::NTuple{G,R}) where {G, R <: TensType} #;convec=ntuple(w->w,ndims(psi)-1))
#  psi = invec[1]
  A = invec[1]
  return A*psi
end

function krylov(invec::R...;Lenv::TensType=[0],Renv::TensType=[0],maxiter::intType = prod(w->size(invec[2],w),1:ndims(invec[2])-1),updatefct::Function=makeHpsi,reorth::Bool=false,effZero::Real=defzero,alpha::Array{S,1}=Array{Float64,1}(undef,maxiter),beta::Array{S,1}=Array{Float64,1}(undef,maxiter),psisave::Array{R,1}=Array{typeof(invec[1]),1}(undef,maxiter),start::intType=0,cvg::Bool=false,numE::intType=cvg ? maxiter : 0,saveE::Array{S,1}=[1000. for w = 1:numE],goal::S=1E-12) where {R <: TensType, S <: Number}
  #specifying ::R for the type of the invec is best done if the TensType is not substituted for R...however, if using Julia types, then the function throws a bunch of extra allocations for the generality here. Tested for Vector and Matrix input. The code here is the most general and least costly version available and is enabled by the denstens type

  psi = invec[1]
  keepvec = Base.tail(invec)

  psi = div!(psi,norm(psi))

  K = eltype(psi)
  n = start
  betatest = true
  while n < maxiter && betatest
    n += 1

    psisave[n] = psi

    Hpsi = updatefct(Lenv,Renv,psi,keepvec)
    alphaval = dot(psi,Hpsi)
    if n > length(alpha)
      push!(alpha,alphaval)
    else
      alpha[n] = alphaval
    end

    if n < maxiter
      if n == 1
        coeffs = (1.,-alpha[n])
        psi = tensorcombination!(coeffs,Hpsi,psi)
      else
        coeffs = (1.,-alpha[n],-beta[n-1])
        psi = tensorcombination!(coeffs,Hpsi,psi,psisave[n-1])

        if reorth
          overlap = dot(psi,psisave[n])
          psi = tensorcombination!((K(1),-overlap),psi,psisave[n])
        end
      end

      betaval = norm(psi)
      if n > length(beta)
        push!(beta,betaval)
      else
        beta[n] = betaval
      end
      psi = div!(psi,betaval)

      betatest = abs(betaval) > effZero

      if cvg && betatest && n >= numE
        D = eigvals(alpha,beta,n)

        ncvg = 0
        w = 0
        while w < numE && abs(D[w+1]-saveE[w+1]) < goal
          w += 1
          ncvg += 1
        end

        if ncvg == numE
          betatest = false
        end
        for w = 1:numE
          saveE[w] = D[w]
        end
      end
#    elseif reorth
#      overlap = dot(psi,psisave[n])
#      psisave[n] = tensorcombination!((K(1),-overlap),psi,psisave[n])
    end
  end

  return n
end
export krylov

function lanczos(invec::R...;Lenv::TensType=[0],Renv::TensType=[0],maxiter::intType = size(invec[2],1),updatefct::Function=makeHpsi,reorth::Bool=false,start::intType=0,effZero::Real=defzero,retnum::intType=1,alpha::Array{S,1}=Array{Float64,1}(undef,maxiter),beta::Array{S,1}=Array{Float64,1}(undef,maxiter),psisave::Array{W,1}=Array{typeof(invec[1]),1}(undef,maxiter),cvg::Bool=false,numE::intType=cvg ? maxiter : 0, saveE::Array{S,1}=Array{Float64,1}(undef,numE)) where {R <: TensType, S <: Number, W <: TensType}

  p = krylov(invec...,Lenv=Lenv,Renv=Renv,maxiter=maxiter,updatefct=updatefct,reorth=reorth,effZero=effZero,alpha=alpha,beta=beta,psisave=psisave,cvg=cvg,numE=numE,saveE=saveE,start=start)

  D,U = libeigen(alpha,beta,p)

  if p < length(psisave)
    psisave = psisave[1:p]
  end

  if retnum == 0
    retpsi = psisave
  else
    retsize = min(p, retnum)
    retpsi = Array{eltype(psisave),1}(undef, retsize)

    typepsi = eltype(psisave[1])
    sametype = eltype(U) == eltype(psisave[1])
    if retsize == 1
      if sametype
        coeffs = ntuple(k->LinearAlgebra.adjoint(U[k,1]),p)
      else
        coeffs = ntuple(k->LinearAlgebra.adjoint(convert(typepsi,U[k,1])),p)
      end
      retpsi[1] = tensorcombination!(coeffs,psisave...)
    else
      @inbounds for i = 1:retsize
        if sametype
          coeffs = ntuple(k->LinearAlgebra.adjoint(U[k,i]),p)
        else
          coeffs = ntuple(k->LinearAlgebra.adjoint(convert(typepsi,U[k,i])),p)
        end
        retpsi[i] = tensorcombination(coeffs,psisave...)
      end
    end
  end

  return retpsi,D
end
export lanczos

function lanczos(psi::AbstractArray,invec::AbstractArray...;Lenv::TensType=[0],Renv::TensType=[0],maxiter::intType = prod(w->size(invec[1],w),1:ndims(invec[1])-1),retnum::intType=1,updatefct::Function=makeHpsi,reorth::Bool=false,effZero::Real=defzero,alpha::Array{S,1}=Array{Float64,1}(undef,maxiter),beta::Array{S,1}=Array{Float64,1}(undef,maxiter),psisave::Array{R,1}=Array{typeof(psi),1}(undef,maxiter),start::intType=0,cvg::Bool=false,numE::intType=cvg ? maxiter : 0,saveE::Array{S,1}=Array{Float64,1}(undef,numE)) where {R <: TensType, S <: Number}
  changevec = ntuple(w->tens(invec[w]),length(invec))
  if eltype(psisave) <: denstens
    newpsisave = psisave
  else
    newpsisave = Array{tens{eltype(psi)},1}(undef,length(psisave))
  end
  retpsi,D = lanczos(tens(psi),changevec...,Lenv=Lenv,Renv=Renv,maxiter=maxiter,retnum=retnum,updatefct=updatefct,reorth=reorth,effZero=effZero,alpha=alpha,beta=beta,psisave=newpsisave,start=start,numE=numE,saveE=saveE)

  true_retpsi = Array{typeof(psi),1}(undef,length(retpsi))

  for w = 1:length(true_retpsi)
    psisave[w] = makeArray(newpsisave[w])
    true_retpsi[w] = makeArray(retpsi[w])
  end

  return true_retpsi,D
end
