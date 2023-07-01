
println("#            +-------------------+")
println("#>-----------|  contract_time.jl |-----------<")
println("#            +-------------------+")


#using BenchmarkTools

fulltest = true

@makeQNs "spin" U1
QS = 2

chi = [spin(2),spin(0),spin(0),spin(-2)]

numdims = [2*i for i = 1:7]

for w in numdims
  local QNs = [chi for g = 1:w]
  global S = div(w,2)
  local Arrows = vcat([false for g = 1:S],[true for g = 1:S])
  
  local d = size(chi,1)
  
  local msize = convert(Int64,d^(size(QNs,1)/2))
  
  println("#####################################")
  println("#####################################")
  println("       $msize x $msize        (two rank ",w,"s)")
  println("#####################################")
  println("#####################################")
  
  global A = rand(QNs,Arrows)
  global B = rand(QNs,Arrows)

  local Ltup = ntuple(i->i+S,S)
  local Rtup = ntuple(i->i,S)
  
  @time C = contract(A,Ltup,B,Rtup)
  @time C = contract(A,Rtup,B,Ltup)
  println()
end

