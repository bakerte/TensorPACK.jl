
import Serialization
file = testpath*"dict"*file_extension
if isfile(file)
  performancevals = Serialization.deserialize(file)
else
  performancevals = Dict()
end

println("#            +-------------+")
println("#>-----------|  libalg.jl  |-----------<")
println("#            +-------------+")
global fulltest = true

import LinearAlgebra

typevec = [Float32,Float64,ComplexF32,ComplexF64]
tol = [1E-4,1E-10,1E-4,1E-11]

for g = 1:length(typevec)
  local A = rand(typevec[g],100,100)
  local A += A'
  local D,U = libeigen(A)

  local tA = tens(A)
  local tD,tU = libeigen(tA)

  local checkD,checkU = LinearAlgebra.eigen(A)

  local testval = isapprox(U*LinearAlgebra.Diagonal(D)*U',A)
  local newU = reshape(tU,100,100)
  local testval &= isapprox(newU*LinearAlgebra.Diagonal(D)*newU',A)
  global fulltest &= testfct(testval,"eigen $(typevec[g])")
end

println()

for g = 1:length(typevec)
  local A = rand(typevec[g],200,100)

  local U,D,V = TENPACK.libsvd(A)

  local U = reshape(U,200,100)
  local V = reshape(V,100,100)

  local tA = tens(A)
  local tU,tD,tV = TENPACK.libsvd(tA)

#  checkD,checkU = LinearAlgebra.eigen(A)

  local testval = isapprox(U*LinearAlgebra.Diagonal(D)*V,A)
  local newU = reshape(tU,200,100)
  local newV = reshape(tV,100,100)
  local testval &= isapprox(newU*LinearAlgebra.Diagonal(D)*newV,A)
  global fulltest &= testfct(testval,"svd $(typevec[g])")
end

println()

for g = 1:length(typevec)
  global A = rand(typevec[g],200,100)
  global B = rand(typevec[g],100,200)

  global alpha = typevec[g](3)
  global beta = typevec[g](3)

  global C = rand(typevec[g],200,200)

  global realC = alpha*A*B+beta*C

  global C = TENPACK.matmul!('N','N',alpha,A,B,beta,C,200,100,100,200)
  
  global checkC = TENPACK.matmul('N','N',alpha,A,B,200,100,100,200)

  global tA = tens(A)
  global tU,tD,tV = TENPACK.libsvd(tA)

#  checkD,checkU = LinearAlgebra.eigen(A)

  local testval = "isapprox(C,realC) && isapprox(checkC,alpha*A*B)"
  global fulltest &= testfct(testval,"matmul(!) $(typevec[g])",performancevals)
end

println()

for g = 1:length(typevec)
  global A = rand(typevec[g],100,100)
  global B = rand(typevec[g],100,100)
  for y = 1:100
    for x = y+1:100
      global B[x,y] = 0.
    end
  end

  global C = B*A

#  display(C)

  global checkC = TENPACK.trmm!(B,100,100,A,100,side='L')

#  display(checkC)

  local testval = "isapprox(C,checkC)"
  global fulltest &= testfct(testval,"trmm! $(typevec[g])",performancevals)
end

println()

for g = 1:length(typevec)
  global A = rand(typevec[g],100,200)
#  tA = tens(A)

  global checkQ,checkR = LinearAlgebra.qr(A)

  global Q,R = TENPACK.libqr!(copy(A),100,200)
  global aQ,aR = TENPACK.libqr(copy(A),100,200)

  global out = 0
  for y = 1:size(R,2)
    @inbounds @simd for x = y+1:size(R,1)
      global out += R[x,y]
    end
  end

  local testval = "isapprox(checkQ*checkR,A) && isapprox(Q*R,A) && isapprox(aQ*aR,A) && isapprox(out,0)"
  global fulltest &= testfct(testval,"libqr! $(typevec[g])",performancevals)
end

println()

for g = 1:length(typevec)
  global A = rand(typevec[g],100,200)
#  tA = tens(A)

  global checkQ,checkR = LinearAlgebra.lq(A)

  global Q,R = TENPACK.liblq!(copy(A),100,200)
  global aQ,aR = TENPACK.liblq!(copy(A),100,200)

  global out = 0
  for y = 1:size(Q,1)
    @inbounds @simd for x = y+1:size(Q,2)
      global out += Q[y,x]
    end
  end

  local testval = "isapprox(checkQ*checkR,A) && isapprox(Q*R,A) && isapprox(aQ*aR,A) && isapprox(out,0)"
  global fulltest &= testfct(testval,"liblq! $(typevec[g])",performancevals)
end

println()

A = rand(100,200)
B = TENPACK.libUpperHessenberg!(A)

global out = 0
for y = 1:size(B,2)
  @inbounds @simd for x = y+1:size(B,1)
    global out += B[x,y]
  end
end

testval = "isapprox(out,0)"
global fulltest &= testfct(testval,"libUpperHessenberg!",performancevals)

println()

A = rand(100,200)
B = TENPACK.libLowerHessenberg!(A)

global out = 0
for y = 1:size(B,1)
  @inbounds @simd for x = y+1:size(B,2)
    global out += B[y,x]
  end
end

testval = "isapprox(out,0)"
global fulltest &= testfct(testval,"libLowerHessenberg!",performancevals)


Serialization.serialize(file,performancevals)

