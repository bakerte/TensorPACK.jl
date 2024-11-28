
import Serialization
file = testpath*"dict"*file_extension
if isfile(file)
  performancevals = Serialization.deserialize(file)
else
  performancevals = Dict()
end


println("#            +-------------------+")
println("#>-----------|  contract_time.jl |-----------<")
println("#            +-------------------+")


#using BenchmarkTools

@makeQNs "spin" U1
QS = 2

chi = [spin(2),spin(0),spin(0),spin(-2)]

numdims = [2*i for i = 1:6]#7]


global testvec = Array{Bool,1}(undef,length(numdims))

global p = 0

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

#  if d < 8
#    local C = contract(A,Rtup,B,Ltup)
    local C = contract(A,Ltup,B,Rtup)

    local checkC = Array(C)

    dA = Array(A)
    dB = Array(B)
    println("dense time: ")
    denseC = contract(dA,Ltup,dB,Rtup)
    @time contract(dA,Ltup,dB,Rtup)

    global p += 1
    diffval = norm(checkC-denseC)

    testvec[p] = diffval < 1E-8
    print("accuracy: ")

    if testvec[p]
      printstyled("PASS",color=:green)
    else
      printstyled("FAIL ",color=:red)
    end
    println()

#  end


  
  @time contract(A,Ltup,B,Rtup)
  @time contract(A,Rtup,B,Ltup)
  println()
end

fulltest = sum(testvec) == length(testvec)

if sum(testvec) != length(testvec)
  println(testvec)
end

Serialization.serialize(file,performancevals)
