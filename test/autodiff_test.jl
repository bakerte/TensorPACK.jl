
import Serialization
file = testpath*"dict"*file_extension
if isfile(file)
  performancevals = Serialization.deserialize(file)
else
  performancevals = Dict()
end

#=
ndim = 4 #max(round(Int64,rand()*10),2)
Asize = (5,7,10,4) #ntuple(w->max(2,round(Int64,rand()*10)),ndim)
A = rand(Asize...)
=#

println("#            +---------------+")
println("#>-----------|  autodiff.jl  |-----------<")
println("#            +---------------+")
fulltest = true

p = 10
A = rand(p,p)
B = rand(p,p)

checkA = dualnum.(A)
checkB = dualnum.(B) #[dualnum(B[i,j],1.) for i = 1:p, j = 1:p]


checkC = checkA * checkB

function Babylonian(x;N=10)
  t = (1+x)/2
  for i = 2:N
    t = (t + x/t)/2
  end
  t
end

function test_autodiff()
  g = 49 # + 2im
  checkval = Babylonian(g)

  p = dualnum(g)
  val = Babylonian(p)

  p * p

  B = p * ones(10,10)

  B * B

  return abs((sqrt(p) - checkval).val) < 1E-8
end

testval = "test_autodiff()"
fulltest &= testfct(testval,"Babylonian test",performancevals)

println()

A = rand(20,10)
dA = dtens(A)

U,D,V = svd(dA)

testval = "norm(U[1]*D[0]*V[0] + U[0]*D[1]*V[0] + U[0]*D[0]*V[1] - ones(20,10)) < 1E-14"

fulltest &= testfct(testval,"svd (dtens)",performancevals)


Serialization.serialize(file,performancevals)
fulltest