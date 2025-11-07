
import Serialization
file = testpath*"dict"*file_extension
if isfile(file)
  performancevals = Serialization.deserialize(file)
else
  performancevals = Dict()
end

println("#            +--------------------+")
println("#>-----------|  tensornetwork.jl  |-----------<")
println("#            +--------------------+")
fulltest = true

import LinearAlgebra

@makeQNs "test9" U1
QS = 2

chi = [test9(2),test9(0),test9(0),test9(-2)]

numdims = 10

QNs = [chi for g = 1:numdims]
S = div(numdims,2)
Arrows = vcat([false for g = 1:S],[true for g = 1:S])

d = size(chi,1)

msize = convert(Int64,d^(size(QNs,1)/2))

A = rand(QNs,Arrows)
#B = rand(QNs,Arrows)

B = Array(A)

C = nametens(A,["c$i" for i = 1:ndims(A)])



testval = C.names == ["c$i" for i = 1:ndims(A)]
testval &= isapprox(Array(C.N),Array(A))
fulltest &= testfct(testval,"nametens(Qtens,array{string})")

D = nametens(B,["b$i" for i = 1:ndims(A)])

testval = D.names == ["b$i" for i = 1:ndims(B)]
testval &= isapprox(Array(D.N),Array(B))
fulltest &= testfct(testval,"nametens(tens,array{string})")

E = nametens(A,"e")

testval = E.names == ["e$i" for i = 1:ndims(A)]
testval &= isapprox(Array(E.N),Array(A))
fulltest &= testfct(testval,"nametens(tens,string)")

println()

A = rand(20,3,8,10)

B = Array(A)
C = nametens(B,["c$i" for i = 1:ndims(B)])

U,D,V = svd(C,[["c1","c2"],["c3","c4"]])

checkC = U*D*V
testval = "isapprox(norm(checkC),norm(C))"
fulltest &= testfct(testval,"svd(nametens,Array{Array})",performancevals)

println()

U,V = qr(C,[["c1","c2"],["c3","c4"]])

checkC = U*V
testval = "isapprox(norm(checkC),norm(C))"
fulltest &= testfct(testval,"qr(nametens,Array{Array})",performancevals)
#=
println()

println(C)

U,V = rq(C,[["c1","c2"],["c3","c4"]])

checkC = U*V
testval = isapprox(norm(checkC),norm(C))
fulltest &= testfct(testval,"rq(tens,Array{Array})")

println()

U,V = ql(C,[["c1","c2"],["c3","c4"]])

checkC = U*V
testval = isapprox(norm(checkC),norm(C))
fulltest &= testfct(testval,"ql(tens,Array{Array})")
=#
println()

U,V = lq(C,[["c1","c2"],["c3","c4"]])

checkC = U*V
testval = "isapprox(norm(checkC),norm(C))"
fulltest &= testfct(testval,"lq(nametens,Array{Array})",performancevals)

println()

testval = "isapprox((C*C)[1],contract(B))"
fulltest &= testfct(testval,"*(nametens{Qtens},nametens{Qtens})",performancevals)

testval = "isapprox(sum((D*D).N.T),contract(B))"
fulltest &= testfct(testval,"*(nametens,nametens)",performancevals)

println()




Serialization.serialize(file,performancevals)

fulltest