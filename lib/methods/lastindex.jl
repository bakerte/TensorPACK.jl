###############################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                               v1.0
#
###############################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.0+)
#

import Base.lastindex
"""
    B = D[end]

`lastindex!` allows to get the end element of an `diagonal`
"""
function lastindex(A::diagonal)
  return length(A)
end
