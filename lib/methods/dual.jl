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
    B = dual!(A)

Creates a new network `B` from any `TNnetwork` `A` with its level field incremented
"""
function dual(A::TNnetwork)
  return dual!(copy(A))
end

"""
    A = dual!(A)

Outputs any `TNnetwork` `A` with its level field incremented
"""
function dual!(A::TNnetwork)
  A.level += 1
  return A
end