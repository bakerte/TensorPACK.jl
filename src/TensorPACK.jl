#########################################################################
#
#  Tensor Linear Algebra Package (TENPACK)
#                  v0.1
#
#########################################################################
# Made by Thomas E. Baker (2023)
# See accompanying license with this program
# This code is native to the julia programming language (v1.8.5+)
#

"""
TENPACK  (version 0.2)\n
(made for julia v1.8.5+ (January 8, 2023), see included license)

Code: https://github.com/bakerte/TENPACK.jl

Documentation: T.E. Baker, "forthcoming"\n
Funding for this program is graciously provided by:
   + US-UK Fulbright Commission (Bureau of Education and Cultural Affairs from the United States Department of State)
   + Department of Physics, University of York
   + Canada Research Chair in Quantum Computing for Modeling of Molecules and Materials
   + Department of Physics & Astronomy, University of Victoria
   + Department of Chemistry, University of Victoria
   
Running the julia kernel with --check-bounds=no can decrease runtimes by 20%

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

println("version 0.1")
println("(made for julia v1.8.5+, see included license)")
println()
println("Code: https://github.com/bakerte/TENPACK.jl")
println()
println("Documentation: T.E. Baker, \"forthcoming\"")
println("Funding for this program is graciously provided by:")
println("   + US-UK Fulbright Commission (Bureau of Education and Cultural Affairs from the United States Department of State)")
println("   + Department of Physics, University of York")
println("   + Canada Research Chair in Quantum Computing for Modeling of Molecules and Materials")
println("   + Department of Physics & Astronomy, University of Victoria")
println("   + Department of Chemistry, University of Victoria")

println("Running the julia kernel with --check-bounds=no can decrease runtimes by 20%")
juliathreads = Threads.nthreads()
println("julia threads: ",juliathreads,"    (modify with 'export JULIA_NUM_THREADS=' or `julia -t #`)")
println("julia processes: ",Distributed.nprocs(),"    (modify with 'Distributed' package commands, `addprocs()` or `julia -p #`)")
LinearAlgebra.BLAS.set_num_threads(juliathreads)
#println("BLAS threads (set in DMRjulia.jl): ",juliathreads)
#println("BLAS threads: ",ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ()))
println()


const libdir = @__DIR__

libpath = libdir*"/lib/"

#Linear algebra routines

include(libpath*"tensor.jl")
include(libpath*"libalg.jl")


include(libpath*"QN.jl")
include(libpath*"Qtensor.jl")


include(libpath*"contractions.jl")
include(libpath*"strassen.jl")


include(libpath*"decompositions.jl")
include(libpath*"Krylov.jl")



include(libpath*"Qcontractions.jl")
include(libpath*"Qdecompositions.jl")


const testpath = libdir*"/test/"

include(testpath*"alltest.jl")

end

#using .TensorPACK
