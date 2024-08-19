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

#       +---------------------------------------+
#       |                                       |
#>------+            SVD of a tensor            +---------<
#       |                                       |
#       +---------------------------------------+
"""
    U,D,V = libsvd(X)

Chooses the best svd function for tensor decomposition with the standard three tensor output for the SVD
"""
function libsvd end

"""
    U,D,V = libsvd!(X)

Chooses the best svd function for tensor decomposition with the standard three tensor output for the SVD but overwrites input matrix
"""
function libsvd! end

"""
    D,U = libeigen(A[,B])

Chooses the best eigenvalue decomposition function for tensor `A` with the standard three tensor output. Can include an overlap matrix for generalized eigenvalue decompositions `B`
"""
function libeigen end

"""
    Q,R = libqr(X[,decomposer=LinearAlgebra.qr])

Decomposes `X` with a QR decomposition.
"""
function libqr end

"""
    L,Q = liblq(X[,decomposer=LinearAlgebra.lq])

Decomposes `X` with a LQ decomposition.
"""
function liblq end

"""
    defzero = 1E-28

default value of zero used in truncating decompositions; we truncate (typically) in the square of the density matrix occupation value, so this is (1E-14)^2
"""
const defzero = 1E-28
