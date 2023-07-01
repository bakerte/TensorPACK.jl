
fulltest = true

using BenchmarkTools

p = 12
m = 2^p

A = rand(m,m)
B = rand(m,m)

checkC = A*B
@btime A*B

C = strassen(A,B)#,A1,A2,A3,A4,B1,B2,B3,B4)
@btime strassen(A,B)#,A1,A2,A3,A4,B1,B2,B3,B4)

testval = norm(C-checkC) < 1E-8
fulltest &= testfct(testval,"strassen(Array,Array)")


