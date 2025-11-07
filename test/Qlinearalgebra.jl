
import Serialization
file = testpath*"dict"*file_extension
if isfile(file)
  performancevals = Serialization.deserialize(file)
else
  performancevals = Dict()
end

println("#            +---------------------+")
println("#>-----------|  Qlinearalgebra.jl  |-----------<")
println("#            +---------------------+")
fulltest = true

import LinearAlgebra

@makeQNs "test5" U1
QS = 2

chi = [test5(2),test5(0),test5(0),test5(-2)]

numdims = 10

QNs = [chi for g = 1:numdims]
S = div(numdims,2)
Arrows = vcat([false for g = 1:S],[true for g = 1:S])

d = size(chi,1)

msize = convert(Int64,d^(size(QNs,1)/2))

A = rand(QNs,Arrows)
B = rand(QNs,Arrows)

Ltup = [i for i = 1:S]
Rtup = [i+S for i = 1:S]

tA = tens(Array(A))
tB = tens(Array(B))

C = contract(A,Ltup,B,Rtup)
tC = contract(tA,Ltup,tB,Rtup)

testval = "isapprox(Array(C),Array(tC))"
fulltest &= testfct(testval,"contract on dense and Qtens",performancevals)

println()

C = ccontract(A,B)
tC = ccontract(tA,tB)

testval = "isapprox(C,tC)"
fulltest &= testfct(testval,"dot on dense and Qtens",performancevals)

println()

U,D,V = svd(A,[Ltup,Rtup],nozeros=false)

tU,tD,tV = svd(tA,[Ltup,Rtup],nozeros=false)

C = reshape(Array(tA),length(chi)^length(Ltup),length(chi)^length(Rtup))
checkU,checkD,checkV = LinearAlgebra.svd(C)

rA = U*D*V
rtA = tU*tD*tV

#testval = isapprox(Array(A),Array(tA))
testval = "isapprox(Array(rA),Array(rtA))"
#testval &= isapprox(Array(),Array(tA))
fulltest &= testfct(testval,"svd on dense and Qtens",performancevals)

println()

U,D,V = svd(A,[Ltup,Rtup])
tU,tD,tV = svd(tA,[Ltup,Rtup])

C = reshape(Array(tA),length(chi)^length(Ltup),length(chi)^length(Rtup))
checkU,checkD,checkV = LinearAlgebra.svd(C)

rA = U*D*V
rtA = tU*tD*tV

#testval = isapprox(Array(A),Array(tA))
testval = "isapprox(Array(rA),Array(rtA))"
#testval &= isapprox(Array(),Array(tA))
fulltest &= testfct(testval,"svd on dense and Qtens [nozeros=true]",performancevals)

println()

U,D,V = svd(A,m=10,[Ltup,Rtup])
tU,tD,tV = svd(tA,m=10,[Ltup,Rtup])

rA = U*D*V
rtA = tU*tD*tV

testval = "isapprox(Array(rA),Array(rtA))"
fulltest &= testfct(testval,"truncating svd on dense and Qtens",performancevals)

println()

m = 200

@makeQNs "testabc" U1
Qlabels = [[testabc(-2),testabc(0),testabc(),testabc(2)] for i = 1:8]
B = rand(Qlabels,[false,false,false,false,true,true,true,true])

A = Array(B)

U,D,V = svd(A,[[1,2,3,4],[5,6,7,8]])

testval = "isapprox(U*D*V,A)"
fulltest &= testfct(testval,"dense svd reconstruction with vector input",performancevals)

#println("dense svd: ",isapprox(U*D*V,A))
println()

C = copy(B)
for q = 1:length(C.T)
  C.T[q] += C.T[q]'
end

D,U = eigen(C,[[1,2,3,4],[5,6,7,8]])

checkD,checkU = eigen(Array(C),[[1,2,3,4],[5,6,7,8]])

checkC = contractc(checkU*checkD,5,checkU,5)

testval = "isapprox(Array(contractc(U*D,ndims(U),U,ndims(U))),checkC)"
fulltest &= testfct(testval,"dense eigen reconstruction with vector input",performancevals)


println()
#println()
#println()
#println("Quantum number version:")
#println()

U,D,V = svd(B,[[1,2,3,4],[5,6,7,8]])

testval = "isapprox(Array(U*D*V),A)"
fulltest &= testfct(testval,"sparse svd reconstruction with vector input",performancevals)

println()

D,U = eigen(C,[[1,2,3,4],[5,6,7,8]])

#println("sparse eigen: ",isapprox(Array(contractc(U*D,ndims(U),U,ndims(U))),checkC))

B = contract(U,ndims(U),D,1)
A = contractc(B,ndims(B),U,ndims(U))

testval = "isapprox(Array(C),Array(A))"
fulltest &= testfct(testval,"sparse eigen reconstruction with vector input",performancevals)


println()


@makeQNs "testxyz" U1

Qlabels = [[testxyz(-2),testxyz(0),testxyz(),testxyz(2)] for i = 1:8]

B = rand(Qlabels,[false,false,false,false,true,true,true,true])
for q = 1:length(B.T)
  B.T[q] += B.T[q]'
end

println()

C = Array(B)
rC = reshape(C,[[1,2,3,4],[5,6,7,8]])
rD,rU = eigen(rC)
D,U = eigen(C,[[1,2,3,4],[5,6,7,8]])

newC = contractc(contract(U,ndims(U),D,1),ndims(U),U,ndims(U))

testval = "norm(newC-C) < 1E-12 && norm(rC-reshape(C,[[1,2,3,4],[5,6,7,8]])) < 1E-12"
fulltest &= testfct(testval,"quantum eigen",performancevals)


qD,qU = eigen(B,[[1,2,3,4],[5,6,7,8]])

P = Array(qD)
testval = "isapprox(sort([P[i,i] for i = 1:size(P,1)]),sort([D[i,i] for i = 1:size(D,1)]))"
fulltest &= testfct(testval,"quantum eigen [comparison with dense version]",performancevals)

println()

newqC = contractc(contract(qU,ndims(qU),qD,1),ndims(qU),qU,ndims(qU))


testvalvec = [true]
for q = 1:length(newqC.T)
  testvalvec[1] &= isapprox(qU.T[q]*qD.T[q]*qU.T[q]',B.T[q])
end

testvalvec[1] &= norm(Array(newqC)-newC) < 1E-12
fulltest &= testfct(testvalvec[1],"reconstruct U*D*U' [quantum number version]")


qDt,qUt = eigen(B,[[1,2,3,4],[5,6,7,8]],transpose=true)

testvalvec = [true]
for q = 1:length(newqC.T)
  testvalvec[1] &= isapprox(qUt.T[q]'*qD.T[q]*qUt.T[q],B.T[q])
end

#newqCt = contract(contractc(qDt,1,qUt,1),1,qUt,1)
newqCt = ccontract(qUt,1,contract(qDt,2,qUt,1),1)
testvalvec[1] &= norm(Array(newqCt)-newC) < 1E-10
fulltest &= testfct(testvalvec[1],"reconstruct U'*D*U [quantum number version, transposed]")

println()

A = contract(qDt,2,qUt,1)
B = contract(qUt,1,qDt,2)

testval = "isapprox(norm(A-permutedims(B,[5,1,2,3,4])),0)"
fulltest &= testfct(testval,"order change for Qtens with diagonal matrix elements",performancevals)

println()

A = rand(QNs,Arrows)

U,D,V = svd(A,[Ltup,Rtup],nozeros=false,minm=2000)

testval = "norm(A-U*D*V) < 1E-10"
fulltest &= testfct(testval,"svd (minm=2000)",performancevals)

for q = 1:length(A.T)
  A.T[q] += A.T[q]'
end

D,U = eigen(A,[Ltup,Rtup],nozeros=false,minm=2000)

testval = "norm(A-contractc(U*D,ndims(U),U,ndims(U))) < 1E-10"
fulltest &= testfct(testval,"eigen (minm=2000)",performancevals)


Serialization.serialize(file,performancevals)

fulltest