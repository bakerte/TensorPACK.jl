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
    A = findextrablocks(B,commoninds)

Finds blocks that do not match in quantum numbers in a `qarray` `B` with an array of matching indices `commoninds`; returns the blocks not found to be used in several functions where tensors are joined together
"""
function findextrablocks(B::Qtens{W,Q},commoninds::Array{NTuple{2,intType},1}) where {W <: Number, Q <: Qnum}
  nB = length(B.T)-length(commoninds)
  Bleftover = Array{intType,1}(undef,nB)

  counter = 0
  z = 0
  while z < length(B.T) && counter < nB
    z += 1
    notfound = true
    q = 0
    @inbounds while notfound && q < length(commoninds)
      q += 1
      notfound = z != commoninds[q][2]
    end
    if notfound
      counter += 1
      Bleftover[counter] = z
    end
  end
  return Bleftover
end