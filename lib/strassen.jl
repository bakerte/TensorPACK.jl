
#naive algorithm as described in Strassen Reloaded

using LinearAlgebra
using BenchmarkTools

p = 10
m = 2^p

alpha = 1 #as in, alpha * A*B

A = rand(m,m)
B = rand(m,m)

checkC = A*B

@btime A*B



oneindexkey = false



function loadmat!(C,i,j,subM;oneindexed=oneindexkey)

  if oneindexed
    half = cld(size(C,1),2)
    a,b = (1+(i-1)*half),(i*half)
    intervalx = a:b
    a,b = (1+(j-1)*half),(j*half)
    intervaly = a:b
  else
    half = cld(size(C,1),2)
    a,b = (1+i*half),((i+1)*half)
    intervalx = a:b
    a,b = (1+j*half),((j+1)*half)
    intervaly = a:b
  end

  for y = 1:length(intervaly)
    for x = 1:length(intervalx)
      C[intervalx[x],intervaly[y]] = subM[x,y]
    end
  end
  nothing
end

function addmat!(C,i,j,subM;oneindexed=oneindexkey)

  if oneindexed
    half = cld(size(C,1),2)
    a,b = (1+(i-1)*half),(i*half)
    intervalx = a:b
    a,b = (1+(j-1)*half),(j*half)
    intervaly = a:b
  else
    half = cld(size(C,1),2)
    a,b = (1+i*half),((i+1)*half)
    intervalx = a:b
    a,b = (1+j*half),((j+1)*half)
    intervaly = a:b
  end

  for y = 1:length(intervaly)
    for x = 1:length(intervalx)
      C[intervalx[x],intervaly[y]] += subM[x,y]
    end
  end
  nothing
end

function submat!(C,i,j,subM;oneindexed=oneindexkey)

  if oneindexed
    half = cld(size(C,1),2)
    a,b = (1+(i-1)*half),(i*half)
    intervalx = a:b
    a,b = (1+(j-1)*half),(j*half)
    intervaly = a:b
  else
    half = cld(size(C,1),2)
    a,b = (1+i*half),((i+1)*half)
    intervalx = a:b
    a,b = (1+j*half),((j+1)*half)
    intervaly = a:b
  end
  
  for y = 1:length(intervaly)
    for x = 1:length(intervalx)
      C[intervalx[x],intervaly[y]] -= subM[x,y]
    end
  end
  nothing
end

function naive_strassen(A,B;alpha=1)
  m = size(A,2)
  half = cld(m,2)

  A00 = A[1:half,1:half]
  A10 = A[half+1:end,1:half]
  A01 = A[1:half,half+1:end]
  A11 = A[half+1:end,half+1:end]

  B00 = B[1:half,1:half]
  B10 = B[half+1:end,1:half]
  B01 = B[1:half,half+1:end]
  B11 = B[half+1:end,half+1:end]

#  M0,M1,M2,M3,M4,M5,M6 = ntuple(Array{Float64,2}(undef,m,m),7)

  #Strassen matrices (naive algorithm)
  M0 = alpha*(A00+A11)*(B00+B11)
  M1 = alpha*(A10+A11)*B00
  M2 = alpha*A00*(B01-B11)
  M3 = alpha*A11*(B10-B00)
  M4 = alpha*(A00+A01)*B11
  M5 = alpha*(A10-A00)*(B00+B01)
  M6 = alpha*(A01-A11)*(B10+B11)

  C = Array{Float64,2}(undef,m,m)

  #oder recommended in sc16.pdf with equals instead of += for untouched squares
    loadmat!(C,0,0,M0) #C[0,0] = M0
    loadmat!(C,1,0,M1) #C[1,0] = M1
    loadmat!(C,0,1,M2) #C[0,1] = M2

    addmat!(C,0,0,M3) #C[0,0] += M3
    addmat!(C,0,1,M4) #C[0,1] += M4
    loadmat!(C,1,1,M5) #C[1,1] = M5
    addmat!(C,0,0,M6) #C[0,0] += M6

    addmat!(C,1,1,M0) #C[1,1] += M0
    submat!(C,1,1,M1) #C[1,1] -= M1
    addmat!(C,1,1,M2) #C[1,1] += M2
    addmat!(C,1,0,M3) #C[1,0] += M3
    submat!(C,0,0,M4) #C[0,0] -= M4

  return C
end


C = naive_strassen(A,B)

@btime naive_strassen(A,B)


norm(C-checkC)

