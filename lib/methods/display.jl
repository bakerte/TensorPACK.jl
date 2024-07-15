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

import Base.display
function display(A::denstens)
  display(Array(A))
  nothing
end

import Base.display
function display(X::TNobj)
  if ndims(X) == 0
    display(X[1])
  else
    display(X.N)
  end
  nothing
end