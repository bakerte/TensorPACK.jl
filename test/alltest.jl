#########################################################################
#
#           Tensor Linear Algebra Package (TENPACK)
#                          v1.0
#
#########################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.4+)
#

function stdavg(x)
  avgval = sum(x)/length(x)
  if length(x) == 1
    stdval = 0
  else #if typeof(x) <: Union{Tuple,Array}
    stdval = sqrt(sum(w->(x[w]-avgval)^2,1:length(x))/(length(x)*(length(x)-1)))
  end
  return avgval,stdval
end

function testfct(test::Bool,message::String)
#  try test
  if test
    printstyled("PASS",color=:green)
  else
    printstyled("FAIL",color=:red)
  end
  println(" "*message)
#  catch
#    error(message)
#  end
  return test
end

const timeflag = "_time"
const memflag = "_mem"

function comparrow!(xvec,y,σ;tol=2)

  print("  ")

  for w = 1:length(xvec)
    x = xvec[w]
    diffxy = abs(x - y)

    if diffxy > tol*σ
      if x < y
        printstyled("v",color=:green)
      else
        printstyled("^",color=:red)
      end
    else
      printstyled(".",color=:yellow)
    end
  end
  print(" ")
end

const SI_units = [("T",12),("G",9),("M",6),("k",3),("",0),("c",-2),("m",-3),("μ",-6),("n",-9),("p",-12),("f",-15),("a",-18)]

function findSI(x::Number)
  w = 1 #length(SI_units)
#  println()
#  println(x)
#  println()
  while w < length(SI_units) && !(SI_units[w][2] > x >= SI_units[w+1][2])
    w += 1
  end
  return SI_units[w+1][1],10^(-1.0*SI_units[w+1][2])
end

const countnum = 10

function compareval!(storedict::Dict,message::String,value::Number;tol::Number=5)

  if typeof(value) <: Integer
    printval = "$(value)"
  else
    p = Base.log(10,value)

    power = floor(intType,p)
#    val = round(value*10^(-power),digits=3)

    prefix,adjustval = findSI(power)
    printval = round(value*adjustval,digits=4)
    printval = "$(printval)$(prefix)s"
  end

  if haskey(storedict,message)
    checkval = storedict[message]
    avgval,stdval = stdavg(checkval)

    diffval = abs(value - avgval)
    
    colourbool = diffval > tol*stdval
    printstyled(printval,color = colourbool ? (:red) : (:green))
    if colourbool
      sigval = length(checkval) == 1 ? 0 : round(diffval/stdval,digits=2)
      printstyled(" (+/- ",sigval,"σ)",color= :yellow)
    end

#    comparrow_vec = [ for w = 1:length(checkval)]

#    compstring = comparrow(checkval,avgval#=value=#,stdval)
#    print(" "*compstring*" ")

    if typeof(checkval) <: Number
      outval = (checkval,value)
    else
      if typeof(checkval) <: Tuple
        if length(checkval) >= countnum
          outval = (Base.tail(checkval)...,value)
        else
          outval = (checkval...,value)
        end
      else
        outval = (value,)
      end
    end

    comparrow!(checkval,avgval,stdval,tol=tol)

  else
    printstyled(printval,color=:blue)
    outval = value
  end

  setindex!(performancevals,outval,message)
end


function testfct(evalstring::String,message::String,storedict::Dict)

  base = Meta.parse(evalstring)
  test = eval(base)

  #  try test
    if test
      printstyled("PASS",color=:green)
    else
      printstyled("FAIL",color=:red)
    end
    println(" "*message)
  #  catch
  #    error(message)
  #  end


    t1 = time()
    alloc = @allocations eval(base)
    t2 = time()



    memstring = message * memflag
    print("     mem: ")

    compareval!(storedict,memstring,alloc)

    print(" | ")

    timestring = message * timeflag
    print("time: ")
    timer = t2-t1#@time eval(base)
    compareval!(storedict,timestring,timer)
    println()

    return test
  end
#=
#define macro like in profileview
function score(fct::Function,inputs...)
  alloc = @allocations fct(inputs...)
  timer = @time fct(inputs...)
  print("memory: ")
  printstyled("$alloc",color=alloc > storedict ? (:red) : (:green))
  print(" ")
  print("time: ")
  printstyled("$timer",color=timer > storedict ? (:red) : (:green))
  println()
end
  =#
function checkall(fulltestrecord::Array{Bool,1},i::Integer,fulltest::Bool)
  fulltestrecord[i] = fulltest
  print("All tests passed? ")
  if fulltest
    printstyled(fulltest,color=:green)
  else
    printstyled(fulltest,color=:red)
  end
  println()
end

const tests = [

  "tensor_test.jl",

  "QN_test.jl",
  "Qtensor_test.jl",

  "libalg_test.jl",

  "Qlinearalgebra.jl",
  
  "contract_time.jl",
  "svdtest.jl",

  "dense_contract_test.jl",
  "dense_decomposition_test.jl",
  "krylov_test.jl",

  "autodiff_test.jl",

  "tensornetwork_test.jl"

  
]

"""
  testlib([,tests=,path=libdir*"/test/"])

Tests all functions in the files enumerated in `tests`. Default is to check all test functionality in the library. Used in nightly builds. See format in `/tests/` folder

See also: [`libdir`](@ref)
"""
function libtest(;tests::Array{String,1}=tests,dir::String=libdir,path::String=dir*"../test/")

  fulltestrecord = Array{Bool,1}(undef,length(tests))

  for i = 1:length(tests)
    @time include(path*tests[i])
    checkall(fulltestrecord,i,fulltest)
  end

  println()

  for i = 1:length(tests)
    if fulltestrecord[i]
      printstyled(fulltestrecord[i],color=:green)
    else
      printstyled(fulltestrecord[i],color=:red)
    end
    println("    ",i,"   ",tests[i])
  end

  println()

  if sum(fulltestrecord) == length(tests)
    println("All passed. Good work. We happy :^)")
  else
    println("These passed:")
    printstyled(tests[fulltestrecord],color=:green)
    println()
    println()
    println("These did not pass:")
    notfulltestrecord = [!fulltestrecord[w] for w = 1:length(tests)]
    printstyled(tests[notfulltestrecord],color=:red)
  end
end
export libtest
