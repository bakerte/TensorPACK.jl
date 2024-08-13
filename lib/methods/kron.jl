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

import LinearAlgebra

"""
CHECK THIS

"""
function kron(Ns::Integer, Mterm::Tuple{R,S}...;dim::Array{S}=[size(Mterm[(w-1) % length(Mterm) + 1][1],1) for w = 1:Ns]) where {R <: TensType, S <: Integer}
  posvec = [Mterm[w][2] for w = 1:length(Mterm)]
  final  =  1
  for i = 1:Ns
      if i in posvec
        index = findfirst(w->i==posvec[w],1:length(posvec))
        final = kron(final, Array(Mterm[index][1]))
      else
        id = Array(eye(dim[i]))
        final = LinearAlgebra.kron(final, id)
      end
  end

  return final
end

function kron(M1::TensType,i::intType,Ns::intType)
  return kron(Ns,(M1,i))
end

function kron(M1::TensType,i::intType,M2::TensType,j::intType,Ns::intType)
  return kron(Ns,(M1,i),(M2,j))
end

function kron(M1::TensType,i::intType,M2::TensType,j::intType,M3::TensType,k::intType,Ns::intType)
  return kron(Ns,(M1,i),(M2,j),(M3,k))
end

function kron(M1::TensType,i::intType,M2::TensType,j::intType,M3::TensType,k::intType,M4::TensType,l::intType,Ns::intType)
  return kron(Ns,(M1,i),(M2,j),(M3,k),(M4,l))
end


function kron(Ns::intType,M1::TensType,i::intType)
  return kron(Ns,(M1,i))
end

function kron(Ns::intType,M1::TensType,i::intType,M2::TensType,j::intType)
  return kron(Ns,(M1,i),(M2,j))
end

function kron(Ns::intType,M1::TensType,i::intType,M2::TensType,j::intType,M3::TensType,k::intType)
  return kron(Ns,(M1,i),(M2,j),(M3,k))
end

function kron(Ns::intType,M1::TensType,i::intType,M2::TensType,j::intType,M3::TensType,k::intType,M4::TensType,l::intType)
  return kron(Ns,(M1,i),(M2,j),(M3,k),(M4,l))
end

function kron(M1::TensType,num::Number)
  return M1*num
end
