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
    display(A)

fancy display format for `denstens` `A` (appears as Array)
"""
function display(A::denstens)
  display(Array(A))
  nothing
end

"""
    display(A)

fancy display format for `TNobj` `A` (appears as Array)
"""
function display(X::TNobj)
  if ndims(X) == 0
    display(X[1])
  else
    display(X.N)
  end
  nothing
end

"""
    display(A)

fancy display format for `diagonal` `A` (appears as LinearAlgebra.Diagonal)
"""
function display(A::diagonal)
  display(LinearAlgebra.Diagonal(A.T))
  nothing
end