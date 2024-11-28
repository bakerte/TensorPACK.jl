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

const libblastrampoline = "libblastrampoline" * (Sys.iswindows() ? "-5" : "")
#libblastrampoline_handle = C_NULL

import LinearAlgebra: BlasReal, BlasComplex, BlasFloat, BlasInt, DimensionMismatch, checksquare, axpy!
import LinearAlgebra.BLAS: @blasfunc #, libblastrampoline

