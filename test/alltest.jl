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

function avg(x)
  return sum(x)/length(x)
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
    if haskey(storedict,memstring)
      colourbool = abs(alloc - avg(storedict[memstring]))/alloc > 0.1
      printstyled("$alloc",color= colourbool ? (:red) : (:green))
      setindex!(performancevals,alloc,memstring)
    else
      printstyled("$alloc",color=:blue)
      setindex!(performancevals,alloc,(memstring,))
    end

    print(" | ")

    timestring = message * timeflag
    print("time: ")
    timer = t2-t1#@time eval(base)
    if haskey(storedict,timestring)
      timechange = abs(timer - avg(storedict[timestring]))/timer
      colourbool = timechange > 0.2
      printstyled("$timer",color= colourbool ? (:red) : (:green))
      if colourbool
        pertime = round(timechange*100)
        printstyled(" ($(pertime)%)",color= :yellow)
      end
    else
      printstyled("$timer",color=:blue)
    end
    setindex!(performancevals,timer,timestring)
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
function libtest(;tests::Array{String,1}=tests,dir::String=libdir,path::String=dir*"/../test/")




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
