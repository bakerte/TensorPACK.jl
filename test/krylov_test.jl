
import Serialization
file = testpath*"dict"*file_extension
if isfile(file)
  performancevals = Serialization.deserialize(file)
else
  performancevals = Dict()
end

println("#            +-------------------+")
println("#>-----------|      krylov.jl    |-----------<")
println("#            +-------------------+")
fulltest = true


import LinearAlgebra

m = 10

A = rand(m,m)
A += A'
psi = rand(m)

#using BenchmarkTools
#@btime krylov(psi,A)

psi,A = tens(psi),tens(A)

D,psisave = lanczos(psi,A,maxiter=m,m=m,reorth=true)

P = [dot(psisave[:,i],psisave[:,j]) for i = 1:length(psisave),j=1:length(psisave)]
testval = isapprox(sum(w->P[w,w],1:size(P,1)),size(P,1)) && abs(sum(P) - size(P,1)) < 1E-3
#=
@btime krylov(psi,A)

D,U = libeigen(alpha,beta)
=#
checkD,checkU = LinearAlgebra.eigen(Array(A))


testval = "norm(D-Diagonal(checkD)) < 1E-8"
fulltest &= testfct(testval,"krylov(tens,tens)",performancevals)




println()

m = 50
g = 2

A = rand(m,m)
A += A'
psi = rand(m)

#using BenchmarkTools
#@btime krylov(psi,A)

psi,A = tens(psi),tens(A)

alpha = Vector{Float64}(undef,m)
beta = Vector{Float64}(undef,m)

n = krylov(psi,A,cvg=true,numE=g,reorth=true,alpha=alpha,beta=beta)
D,U = libeigen(alpha,beta,size(n,2))

checkD,checkU = LinearAlgebra.eigen(Array(A))

#display(D-checkD)

#display(D[1:g])
#display(checkD[1:g])

testval = "norm(D[1:g]-checkD[1:g]) < 1E-8"
fulltest &= testfct(testval,"krylov(tens,tens) [cvg=true]",performancevals)



Serialization.serialize(file,performancevals)


fulltest