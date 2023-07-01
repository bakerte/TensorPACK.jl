function getsubmat(A,i,j)
  m = cld(size(A,1),4)
  xint = (i-1)*m
  yint = (j-1)*m
  return A[xint+1:xint+m,yint+1:yint+m]
end

function combinemats!(C::Array{W,2},coeff::NTuple{R,W},A::NTuple{R,Array{W,2}}) where {W <: Number, R}
  @inbounds for w = 1:length(C)
    C[w] = W(0)
    @inbounds @simd for x = 1:R
      C[w] += coeff[x]*A[x][w]
    end
  end
  return C
end

function combinemats!(C::Array{W,2},xint::UnitRange{R},yint::UnitRange{R},coeff::NTuple{G,W},A::NTuple{G,Array{W,2}}) where {W <: Number, R <: Integer, G}
  for j = 1:size(A[1],2)
    @inbounds for i = 1:size(A[1],1)
      C[xint[i],yint[j]] = W(0)
      #out = W(0)
      @inbounds @simd for x = 1:G
        C[xint[i],yint[j]] += coeff[x]*A[x][i,j]
      end
      #C[xint[i],yint[j]] = out
    end
  end
  return C
end

function strassen(M1,M2,M3,M4,M5,M6,M7)
  term1 = M1+M4-M5+M7
  term2 = M3+M5
  term3 = M2+M4
  term4 = M1-M2+M3+M6
  return [term1 term2; term3 term4]
end

function strassen!(M,M1,M2,M3,M4,M5,M6,M7)

  Threads.@threads for w = 1:4

    if w == 1
      combinemats!(M,1:size(M1,1),1:size(M1,1),(1.,1.,-1.,1.),(M1,M4,M5,M7))
    elseif w == 2
      combinemats!(M,1:size(M1,1),size(M1,1)+1:size(M,2),(1.,1.),(M3,M5))
    elseif w == 3
      combinemats!(M,size(M1,1)+1:size(M,1),1:size(M1,1),(1.,1.),(M2,M4))
    elseif w == 4
      combinemats!(M,size(M1,1)+1:size(M,1),size(M1,1)+1:size(M,2),(1.,-1.,1.,1.),(M1,M2,M3,M6))
    end

  end

  nothing
end

function strassen!(M,x,inMs)

  M1,M2,M3,M4,M5,M6,M7 = inMs

  for x = 1:4
    if x == 1
      combinemats!(M,1:size(M1,1),1:size(M1,1),(1.,1.,-1.,1.),(M1,M4,M5,M7))
    elseif x == 2
      combinemats!(M,1:size(M1,1),size(M1,1)+1:size(M,2),(1.,1.),(M3,M5))
    elseif x == 3
      combinemats!(M,size(M1,1)+1:size(M,1),1:size(M1,1),(1.,1.),(M2,M4))
    elseif x == 4
      combinemats!(M,size(M1,1)+1:size(M,1),size(M1,1)+1:size(M,2),(1.,-1.,1.,1.),(M1,M2,M3,M6))
    end
  end
  nothing
end

function strassen_noloop!(M,x,inMs)

  M1,M2,M3,M4,M5,M6,M7 = inMs

  if x == 1
    combinemats!(M,1:size(M1,1),1:size(M1,1),(1.,1.,-1.,1.),(M1,M4,M5,M7))
  elseif x == 2
    combinemats!(M,1:size(M1,1),size(M1,1)+1:size(M,2),(1.,1.),(M3,M5))
  elseif x == 3
    combinemats!(M,size(M1,1)+1:size(M,1),1:size(M1,1),(1.,1.),(M2,M4))
  elseif x == 4
    combinemats!(M,size(M1,1)+1:size(M,1),size(M1,1)+1:size(M,2),(1.,-1.,1.,1.),(M1,M2,M3,M6))
  end
  nothing
end

function strassen!(M,inMs)

  M1,M2,M3,M4,M5,M6,M7 = inMs

  for x = 1:4
    if x == 1
      combinemats!(M,1:size(M1,1),1:size(M1,1),(1.,1.,-1.,1.),(M1,M4,M5,M7))
    elseif x == 2
      combinemats!(M,1:size(M1,1),size(M1,1)+1:size(M,2),(1.,1.),(M3,M5))
    elseif x == 3
      combinemats!(M,size(M1,1)+1:size(M,1),1:size(M1,1),(1.,1.),(M2,M4))
    elseif x == 4
      combinemats!(M,size(M1,1)+1:size(M,1),size(M1,1)+1:size(M,2),(1.,-1.,1.,1.),(M1,M2,M3,M6))
    end
  end
  nothing
end

function bigloop(M,term1,term2,Avec,Bvec,disc)
  if disc
    Threads.@threads for w = 1:49
      numthread = Threads.threadid()

      Amat = ntuple(k->DMRJtensor.tensorfromdisc(Avec[As_2[w][k]]),length(As_2[w]))
      combinemats!(term1[numthread],As_1[w],Amat)

      Bmat = ntuple(k->DMRJtensor.tensorfromdisc(Bvec[Bs_2[w][k]]),length(Bs_2[w]))
      combinemats!(term2[numthread],Bs_1[w],Bmat)
      LinearAlgebra.mul!(M[w],term1[numthread],term2[numthread])
    end
  else  
    Threads.@threads for w = 1:49

      numthread = Threads.threadid()

      Amat = ntuple(k->Avec[As_2[w][k]],length(As_2[w]))
      combinemats!(term1[numthread],As_1[w],Amat)

      Bmat = ntuple(k->Bvec[Bs_2[w][k]],length(Bs_2[w]))
      combinemats!(term2[numthread],Bs_1[w],Bmat)
      LinearAlgebra.mul!(M[w],term1[numthread],term2[numthread])
    end
  end
end

function otherloop_save(P,Mmats)
  for y = 1:7
    Mvec = [tensorfromdisc(Mmats[y][w]) for w = 1:7]
  Threads.@threads for x = 1:4
#    y = cld(w,4)
#    x = (w-1) % 4 + 1
    strassen_noloop!(P[y],x,Mmats[y])
  end
  end
  nothing
end

function otherloop(P,Mmats)
  Threads.@threads for w = 1:7*4
    y = cld(w,4)
    x = (w-1) % 4 + 1
    strassen_noloop!(P[y],x,Mmats[y])
  end
  nothing
end

#Jake: these are the coefficients

const As_1 = [
  (1.,1.,1.,1.),
  (1.,1.,1.,1.),
  (1.,1.),
  (1.,1.),
  (1.,1.,1.,1.),
  (1.,-1.,1.,-1.),
  (-1.,1.,-1.,1.),
  (1.,1.,1.,1.),
  (1.,1.,1.,1.),
  (1.,1.),
  (1.,1.),
  (1.,1.,1.,1.),
  (1.,1.,-1.,-1.),
  (-1.,-1.,1.,1.),
  (1.,1.),
  (1.,1.),
  (1.,),
  (1.,),
  (1.,1.),
  (1.,-1.),
  (1.,-1.),
  (1.,1.),
  (1.,1.),
  (1.,),
  (1.,),
  (1.,1.),
  (1.,-1.),
  (1.,-1.),
  (1.,1.,1.,1.),
  (1.,1.,1.,1.),
  (1.,1.),
  (1.,1.),
  (1.,1.,1.,1.),
  (1.,1.,-1.,-1.),
  (-1.,-1.,1.,1.),
  (1.,1.,-1.,-1.),
  (1.,1.,-1.,-1.),
  (1.,-1.),
  (1.,-1.),
  (1.,1.,-1.,-1.),
  (1.,-1.,-1.,1.),
  (-1.,1.,1.,-1.),
  (-1.,-1.,1.,1.),
  (-1.,-1.,1.,1.),
  (1.,-1.),
  (1.,-1.),
  (-1.,-1.,1.,1.),
  (-1.,1.,1.,-1.),
  (1.,-1.,-1.,1.)]
  
const  Bs_1 = [(1.,1.,1.,1.),
  (1.,1.),
  (-1.,1.,-1.,1.),
  (1.,-1.,1.,-1.),
  (1.,1.),
  (1.,1.,1.,1.),
  (1.,1.,1.,1.),
  (1.,1.),
  (1.,),
  (1.,-1.),
  (1.,-1.),
  (1.,),
  (1.,1.),
  (1.,1.),
  (-1.,-1.,1.,1.),
  (1.,-1.),
  (1.,-1.,-1.,1.),
  (-1.,1.,1.,-1.),
  (1.,-1.),
  (-1.,-1.,1.,1.),
  (-1.,-1.,1.,1.),
  (1.,1.,-1.,-1.),
  (1.,-1.),
  (-1.,1.,1.,-1.),
  (1.,-1.,-1.,1.),
  (1.,-1.),
  (1.,1.,-1.,-1.),
  (1.,1.,-1.,-1.),
  (1.,1.),
  (1.,),
  (1.,-1.),
  (1.,-1.),
  (1.,),
  (1.,1.),
  (1.,1.),
  (1.,1.,1.,1.),
  (1.,1.),
  (-1.,-1.,1.,1.),
  (1.,1.,-1.,-1.),
  (1.,1.),
  (1.,1.,1.,1.),
  (1.,1.,1.,1.),
  (1.,1.,1.,1.),
  (1.,1.),
  (-1.,-1.,1.,1.),
  (1.,1.,-1.,-1.),
  (1.,1.),
  (1.,1.,1.,1.),
  (1.,1.,1.,1.)]


const As_2 = [
  (16,11,6,1),
  (16,15,6,5),
  (11,1),
  (16,6),
  (12,11,2,1),
  (15,11,5,1),
  (16,12,6,2),
  (16,14,11,9),
  (16,15,14,13),
  (11,9),
  (16,14),
  (12,11,10,9),
  (15,13,11,9),
  (16,14,12,10),
  (6,1),
  (6,5),
  (1,),
  (6,),
  (2,1),
  (5,1),
  (2,6),
  (16,11),
  (16,15),
  (11,),
  (16,),
  (12,11),
  (15,11),
  (12,16),
  (8,6,3,1),
  (8,7,6,5),
  (3,1),
  (8,6),
  (4,3,2,1),
  (7,5,3,1),
  (8,6,4,2),
  (14,9,6,1),
  (14,13,6,5),
  (9,1),
  (14,6),
  (10,9,2,1),
  (13,9,5,1),
  (14,10,6,2),
  (16,11,8,3),
  (16,15,8,7),
  (3,11),
  (8,16),
  (12,11,4,3),
  (15,11,7,3),
  (16,12,8,4)]
  
  const Bs_2 = [
    (16,11,6,1),
  (11,1),
  (16,12,6,2),
  (15,11,5,1),
  (16,6),
  (12,11,2,1),
  (16,15,6,5),
  (6,1),
  (1,),
  (2,6),
  (5,1),
  (6,),
  (2,1),
  (6,5),
  (16,11,8,3),
  (3,11),
  (16,12,8,4),
  (15,11,7,3),
  (8,16),
  (12,11,4,3),
  (16,15,8,7),
  (14,9,6,1),
  (9,1),
  (14,10,6,2),
  (13,9,5,1),
  (14,6),
  (10,9,2,1),
  (14,13,6,5),
  (16,11),
  (11,),
  (12,16),
  (15,11),
  (16,),
  (12,11),
  (16,15),
  (8,6,3,1),
  (3,1),
  (8,6,4,2),
  (7,5,3,1),
  (8,6),
  (4,3,2,1),
  (8,7,6,5),
  (16,14,11,9),
  (11,9),
  (16,14,12,10),
  (15,13,11,9),
  (16,14),
  (12,11,10,9),
  (16,15,14,13)]

function strassen(A::Array{W,2},B::Array{W,2};disc::Bool=false,otherdisc::Bool=false) where W <: Number

  store_threads = LinearAlgebra.BLAS.get_num_threads()
#  num_threads = 1 #4
  LinearAlgebra.BLAS.set_num_threads(1)


  if disc
    Avec = Array{String,1}(undef,16)
    Bvec = Array{String,1}(undef,16)

    Threads.@threads for w = 1:16
      i = cld(w,4)
      j = (w-1) % 4 + 1
      Avec[w] = "A_$w"
      DMRJtensor.tensor2disc(Avec[w],getsubmat(A,i,j))
    end

    Threads.@threads for w = 1:16
      i = cld(w,4)
      j = (w-1) % 4 + 1
      Bvec[w] = "B_$w"
      DMRJtensor.tensor2disc(Bvec[w],getsubmat(B,i,j))
    end
  else
    Avec = Array{Array{W,2},1}(undef,16)
    Bvec = Array{Array{W,2},1}(undef,16)

    Threads.@threads for w = 1:16
      i = cld(w,4)
      j = (w-1) % 4 + 1
      Avec[w] = getsubmat(A,i,j)
    end

    Threads.@threads for w = 1:16
      i = cld(w,4)
      j = (w-1) % 4 + 1
      Bvec[w] = getsubmat(B,i,j)
    end
  end


  m = cld(size(A,1),4)

  if otherdisc
    M = ["M_$w" for w = 1:49]

    term1 = ntuple(w->Array{W,2}(undef,m,m),Threads.nthreads())
    term2 = ntuple(w->Array{W,2}(undef,m,m),Threads.nthreads())

    bigloop(M,term1,term2,Avec,Bvec,disc)

    Mmats = [ntuple(k->M[k+7*(w-1)],7) for w = 1:7]


    m = cld(size(A,1),2)
    P = ntuple(w->Array{W,2}(undef,m,m),7)

    otherloop_save(P,Mmats)

  #  M1,M2,M3,M4,M5,M6,M7 = P

    C = Array{W,2}(undef,size(A,1),size(B,2))
    strassen!(C,1,P)

  else
    M = [Array{W,2}(undef,m,m) for w = 1:49]

    term1 = ntuple(w->Array{W,2}(undef,m,m),Threads.nthreads())
    term2 = ntuple(w->Array{W,2}(undef,m,m),Threads.nthreads())

    bigloop(M,term1,term2,Avec,Bvec,disc)

    Mmats = [ntuple(k->M[k+7*(w-1)],7) for w = 1:7]


    m = cld(size(A,1),2)
    P = ntuple(w->Array{W,2}(undef,m,m),7)

    otherloop(P,Mmats)

    C = Array{W,2}(undef,size(A,1),size(B,2))
    strassen!(C,1,P)
  end

  LinearAlgebra.BLAS.set_num_threads(store_threads)

  return C
end
export strassen
