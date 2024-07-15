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
  Module: Krylov

Expanding operators in a Krylov subspace
"""

function makeHpsi(Lenv::TensType,Renv::TensType,psi::TensType,invec::R) where {R <: TensType} #;convec=ntuple(w->w,ndims(psi)-1))
  return makeHpsi(Lenv,Renv,psi,(invec,))
end

function makeHpsi(Lenv::TensType,Renv::TensType,psi::TensType,invec::NTuple{G,R}) where {G, R <: TensType} #;convec=ntuple(w->w,ndims(psi)-1))
#  psi = invec[1]
  A = invec[1]
  return A*psi
end

function krylov(invec::R...;Lenv::TensType=[0],Renv::TensType=[0],maxiter::intType = prod(w->size(invec[2],w),1:ndims(invec[2])-1),updatefct::Function=makeHpsi,reorth::Bool=false,effZero::Real=defzero,alpha::Array{S,1}=Array{Float64,1}(undef,maxiter),beta::Array{S,1}=Array{Float64,1}(undef,maxiter),psisave::TensType=Array{typeof(invec[1]),1}(undef,maxiter),start::intType=0,cvg::Bool=false,numE::intType=cvg ? 1 : 0,saveE::Array{S,1}=[1000. for w = 1:numE],goal::S=1E-12,large::Bool=false,eigvecs::Bool=true) where {R <: TensType, S <: Number}
  #specifying ::R for the type of the invec is best done if the TensType is not substituted for R...however, if using Julia types, then the function throws a bunch of extra allocations for the generality here. Tested for Vector and Matrix input. The code here is the most general and least costly version available and is enabled by the denstens type

  psi = invec[1]
  keepvec = Base.tail(invec)

  normpsi = norm(psi)
  if isapprox(normpsi,0)
    error("Input vector for lanczos has a norm of zero. Can not complete Lanczos iterations")
  end

  psi = div!(psi,normpsi)
  prevpsi = psi

  convec = ntuple(w->w,ndims(psi))
  mindim = prod(w->size(psi,w),convec)
  if reorth
    retbundlepsi = reshape(psi,size(psi)...,1)
  end

  K = eltype(psi)
  n = start
  betatest = true

  while (n < maxiter || maxiter == 0) && n < mindim && betatest
    n += 1

#    if eigvecs
      if n > length(psisave)
        push!(psisave,psi)
      else
        psisave[n] = psi
      end
#    end

    Hpsi = updatefct(Lenv,Renv,psi,keepvec)

    alphaval = real(dot(psi,Hpsi))
    if n > length(alpha)
      push!(alpha,alphaval)
    else
      alpha[n] = alphaval
    end

    if (n < maxiter || maxiter == 0)
      if n == 1
        coeffs = (K(1),K(-alphaval))
        Hpsi = tensorcombination!(coeffs,Hpsi,psi)
      else
        coeffs = (K(1),K(-alphaval),K(-beta[n-1]))
        Hpsi = tensorcombination!(coeffs,Hpsi,psi,prevpsi)
        prevpsi = psi
      end

      if reorth
        s = ccontract(Hpsi,convec,retbundlepsi,convec)
        Hpsi -= ccontract(s,ndims(s),retbundlepsi,ndims(retbundlepsi))
      end

      betaval = norm(Hpsi)
      if n > length(beta)
        push!(beta,betaval)
      else
        beta[n] = betaval
      end
      psi = div!(Hpsi,betaval)

      if reorth
        rpsi = reshape(psi,size(psi)...,1)
        retbundlepsi = joinindex!(retbundlepsi,rpsi,ndims(retbundlepsi))
      end

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
    end
  end

  return n
end
export krylov

function lanczos(invec::R...;Lenv::TensType=[0],Renv::TensType=[0],maxiter::intType = size(invec[2],1),updatefct::Function=makeHpsi,reorth::Bool=false,start::intType=0,goal::W=1E-12,effZero::Real=defzero,retnum::intType=1,alpha::Array{S,1}=Array{Float64,1}(undef,maxiter),beta::Array{S,1}=Array{Float64,1}(undef,maxiter),psisave::TensType=Array{typeof(invec[1]),1}(undef,maxiter),cvg::Bool=false,numE::intType=cvg ? 1 : 0,double::Bool=false, saveE::Array{S,1}=Array{Float64,1}(undef,numE),large::Bool=false,eigvecs::Bool=true) where {W <: Number, R <: TensType, S <: Number}

  p = krylov(invec...,Lenv=Lenv,Renv=Renv,maxiter=maxiter,updatefct=updatefct,reorth=reorth,effZero=effZero,alpha=alpha,beta=beta,psisave=psisave,cvg=cvg,numE=numE,saveE=saveE,start=start,large=large,eigvecs=eigvecs,goal=goal)

  D,U = libeigen(alpha,beta,p)

#  println(eigvecs)

  if eigvecs
    if p < length(psisave)
      psisave = psisave[1:p]
    end

    if retnum == 0
      retpsi = psisave
    else
      retsize = min(p, (double ? 2 : 1 ) * retnum)
      retpsi = Array{eltype(psisave),1}(undef, (double ? 2 : 1 ) * retsize)

      retrange = length(retpsi) #cld(length(retpsi),2)

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
        @inbounds for i = 1:retrange
          if sametype
            coeffs = ntuple(k->LinearAlgebra.adjoint(U[k,i]),p)
          else
            coeffs = ntuple(k->LinearAlgebra.adjoint(convert(typepsi,U[k,i])),p)
          end
          retpsi[i] = tensorcombination(coeffs,psisave...)
        end

        if double

          retrange_double = fld(length(retpsi),2)
#          println("doubling")

          counter = 0
          @inbounds for i = retrange_double:-1:1
            if sametype
              coeffs = ntuple(k->LinearAlgebra.adjoint(U[k,end-(i-1)]),p)
            else
              coeffs = ntuple(k->LinearAlgebra.adjoint(convert(typepsi,U[k,end-(i-1)])),p)
            end
            counter += 1
#            println(retrange+counter)
            retpsi[retrange+counter] = tensorcombination(coeffs,psisave...)
          end
        end

      end
    end
  else
    if p < length(psisave)
      psisave = psisave[1:p]
    end
    retpsi = psisave
#    retpsi = typeof(invec[1]) <: denstens ? tens(U) : U
  end



#  println(maxiter," ",retnum," ",p," ",)
#  println(D)
#  println(retpsi)

#  println("Lanczos result:")
#  println("alpha = ",alpha[1:p])
#  println("beta = ",beta[1:p])
#  println()

  return D,retpsi
end
export lanczos

function lanczos(psi::AbstractArray,invec::AbstractArray...;Lenv::TensType=[0],Renv::TensType=[0],maxiter::intType = prod(w->size(invec[1],w),1:ndims(invec[1])-1),retnum::intType=1,goal::W=1E-12,updatefct::Function=makeHpsi,reorth::Bool=false,effZero::Real=defzero,alpha::Array{S,1}=Array{Float64,1}(undef,maxiter),beta::Array{S,1}=Array{Float64,1}(undef,maxiter),psisave::TensType=Array{typeof(psi),1}(undef,maxiter),start::intType=0,cvg::Bool=false,double::Bool=false,numE::intType=cvg ? 1 : 0,saveE::Array{S,1}=Array{Float64,1}(undef,numE),large::Bool=false) where {S <: Number, W <: Number}
  changevec = ntuple(w->tens(invec[w]),length(invec))
  if eltype(psisave) <: denstens
    newpsisave = psisave
  else
    newpsisave = Array{tens{eltype(psi)},1}(undef,length(psisave))
  end
  retpsi,D = lanczos(tens(psi),changevec...,Lenv=Lenv,Renv=Renv,maxiter=maxiter,retnum=retnum,updatefct=updatefct,reorth=reorth,effZero=effZero,alpha=alpha,beta=beta,psisave=newpsisave,start=start,numE=numE,saveE=saveE,large=large)

  true_retpsi = Array{typeof(psi),1}(undef,length(retpsi))

  for w = 1:length(true_retpsi)
    psisave[w] = Array(newpsisave[w])
    true_retpsi[w] = Array(retpsi[w])
  end

  return D,true_retpsi,alpha,beta
end
