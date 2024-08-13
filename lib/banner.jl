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
println("   + Canada Research Chair in Quantum Computing for Modelling of Molecules and Materials")
println("   + Department of Physics & Astronomy, University of Victoria")
println("   + Department of Chemistry, University of Victoria")
println("   + Faculty of Science, University of Victoria")
println("   + National Science and Engineering Research Council (NSERC)")

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
