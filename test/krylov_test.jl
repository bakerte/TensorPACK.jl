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
#@time krylov(psi,A)

psi,A = tens(psi),tens(A)

psisave,D = lanczos(psi,A,maxiter=m,reorth=true)

P = [dot(psisave[i],psisave[j]) for i = 1:length(psisave),j=1:length(psisave)]
testval = isapprox(sum(w->P[w,w],1:size(P,1)),size(P,1)) && abs(sum(P) - size(P,1)) < 1E-3
#=
@time krylov(psi,A)

D,U = libeigen(alpha,beta)
=#
checkD,checkU = LinearAlgebra.eigen(makeArray(A))

#display(D-checkD)

testval &= norm(D-checkD) < 1E-8
fulltest &= testfct(testval,"krylov(tens,tens)")

