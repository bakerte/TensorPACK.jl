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

"""
TENPACK  (version 1.0)\n
(made for julia v1.10.4+ (July 22, 2024), see included license)

Code: https://github.com/bakerte/TensorPACK.jl

Documentation: T.E. Baker, "forthcoming"\n

Funding for this program is graciously provided by:
   + Institut quantique (Université de Sherbrooke)
   + Département de physique, Université de Sherbrooke
   + Canada First Research Excellence Fund (CFREF)
   + Institut Transdisciplinaire d'Information Quantique (INTRIQ)
   + US-UK Fulbright Commission (Bureau of Education and Cultural Affairs from the United States Department of State)
   + Department of Physics, University of York
   + Canada Research Chair in Quantum Computing for Modelling of Molecules and Materials
   + Department of Physics & Astronomy, University of Victoria
   + Department of Chemistry, University of Victoria
   + Faculty of Science, University of Victoria
   + National Science and Engineering Research Council (NSERC)

# Warning:

We recommend not defining `using LinearAlgebra` to avoid conflicts.  Instead, define
```
import LinearAlgebra
```
and define functions as `LinearAlgebra.svd` to use functions from that package.

"""
module TensorPACK
import LinearAlgebra
import Printf
import Distributed

println(" _____ _____ _   _ ______  ___  _____  _   __ ")
println("|_   _|  ___| \\ | || ___ \\/ _ \\/  __ \\| | / / ")
println("  | | | |__ |  \\| || |_/ / /_\\ \\ /  \\/| |/ /  ")
println("  | | |  __|| . ` ||  __/|  _  | |    |    \\  ")
println("  | | | |___| |\\  || |   | | | | \\__/\\| |\\  \\ ")
println("  \\_/ \\____/\\_| \\_/\\_|   \\_| |_/\\____/\\_| \\_/ ")

println("version 1.0")
println("(made for julia v1.10.4+, see included license)")
println()
println("Code: https://github.com/bakerte/TENPACK.jl")
println()
println("Please cite: T.E. Baker, \"forthcoming\"")
println("...and any other algorithms used (noted in documentation)")
println()
println("Funding for this program is graciously provided by:")
println("   + Institut quantique (Université de Sherbrooke)")
println("   + Département de physique, Université de Sherbrooke")
println("   + Canada First Research Excellence Fund (CFREF)")
println("   + Institut Transdisciplinaire d'Information Quantique (INTRIQ)")
println("   + US-UK Fulbright Commission (Bureau of Education and Cultural Affairs from the United States Department of State)")
println("   + Department of Physics, University of York")
println("   + Canada Research Chair in Quantum Computing for Modeling of Molecules and Materials")
println("   + Department of Physics & Astronomy, University of Victoria")
println("   + Department of Chemistry, University of Victoria")
println("   + Faculty of Science, University of Victoria")
println("   + National Science and Engineering Research Council (NSERC)")

const TENPACK = TensorPACK
export TENPACK


#println("Running the julia kernel with --check-bounds=no can decrease runtimes by 20%")
juliathreads = Threads.nthreads()
println("julia threads: ",juliathreads,"    (modify with 'export JULIA_NUM_THREADS=' or `julia -t #`)")
println("julia processes: ",Distributed.nprocs(),"    (modify with 'Distributed' package commands, `addprocs()` or `julia -p #`)")
LinearAlgebra.BLAS.set_num_threads(juliathreads)
#println("BLAS threads (set in DMRjulia.jl): ",juliathreads)
#println("BLAS threads: ",ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ()))
println()
println("# Warning:")
println("We recommend not defining `using LinearAlgebra` to avoid conflicts.  Instead, define")
println("```\n
import LinearAlgebra\n
```\n")
println("and define functions as `LinearAlgebra.svd` to use functions from that package.")
println()


const libdir = @__DIR__

libpath = libdir*"/../lib/"

#Linear algebra routines

files = ["types.jl","tensordef.jl","largevector.jl","libalg.jl","isapprox.jl"]
for w = 1:length(files)
   include(libpath*files[w])
end

subdir = "tensor/"
files = ["tens.jl","diagonal.jl","Array.jl","undefMat.jl","dtens.jl","directedtens.jl","network.jl"]
for w = 1:length(files)
   include(libpath*subdir*files[w])
end

subdir = "methods/"
files = ["rand.jl","zeros.jl","ones.jl","eye.jl","eltype.jl","root.jl","size.jl","length.jl","conj.jl","copy.jl","setindex.jl","getindex.jl","reshape.jl","exp.jl","invmat.jl","sqrt.jl","positionindex.jl","tensor_ranges.jl","minmax.jl","transpose.jl","adjoint.jl","directsum.jl","checktype.jl","addmultsubdiv.jl","sum.jl","norm.jl","convIn.jl","permute.jl","tensorcombination.jl","display.jl","print.jl","nametens.jl","ndims.jl","dual.jl","rename.jl","swapname.jl","joinTens.jl","dualnum.jl","swapgate.jl","det.jl"]
for w = 1:length(files)
   include(libpath*subdir*files[w])
end




subdir = "qnumber/"
include(libpath*subdir*"qnum.jl")

files = [#="orderblocks.jl",=#"QnumList.jl","matchblocks.jl","changeblock.jl","findextrablocks.jl","Qtens.jl","checkflux.jl"]
for w = 1:length(files)
   include(libpath*subdir*files[w])
end


subdir = "contract/"
files = ["dot.jl","dmul.jl","diagcontract.jl","maincontractor.jl","contract.jl","ccontract.jl","contractc.jl","ccontractc.jl","star.jl","autocontract.jl","trace.jl","checkcontract.jl"]
for w = 1:length(files)
   include(libpath*subdir*files[w])
end


subdir = "decompose/"
files = ["decompositions.jl","truncate.jl","svd.jl","svdvals.jl","eigen.jl","eigvals.jl","lq.jl","qr.jl","polar.jl","Krylov.jl","nullspace.jl"]
for w = 1:length(files)
   include(libpath*subdir*files[w])
end

const testpath = libdir*"/../test/"

include(testpath*"alltest.jl")

end

#using .TensorPACK
