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
    C = kron(Ns,M...[,dim=...])

Takes the kronecker product over `Ns` sites of any number of operators given in a tuple `M` (2 elements: operator, site to be applied) with dimensions on each site `dim`
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

"""
    C = kron(Ns,M...[,dim=...])

Takes the kronecker product over `Ns` sites of any number of operators given in a tuple `M` (2 elements: operator, site to be applied) with dimensions on each site `dim`
"""
function kron(F::Union{denstens,diagonal,qarray}, Ns::Integer,Mterm::Tuple{R,S}...;dim::Array{S}=[size(Mterm[(w-1) % length(Mterm) + 1][1],1) for w = 1:Ns]) where {R <: TensType, S <: Integer}
  posvec = [Mterm[w][2] for w = 1:length(Mterm)]
  final  =  1
  minpos = minimum(posvec)
  for i = 1:Ns
      if i in posvec
        index = findfirst(w->i==posvec[w],1:length(posvec))
        final = kron(final, Array(Mterm[index][1]))
      elseif i < minpos
        arrayF = Array(F)
        final = LinearAlgebra.kron(final, arrayF)
      else
        id = Array(eye(dim[i]))
        final = LinearAlgebra.kron(final, id)
      end
  end
  return final
end

"""
    C = kron(Op,i,Ns)

Creates the kronecker product of an input operator `Op` on site `i` of an `Ns` site system
"""
function kron(M1::TensType,i::intType,Ns::intType)
  return kron(Ns,(M1,i))
end

"""
    C = kron(F,Op,i,Ns)

Creates the kronecker product of an input operator `Op` (`denstens`) on site `i` of an `Ns` site system with a trailing operator `F`
"""
function kron(F::TensType,M1::TensType,i::intType,Ns::intType)
  return kron(F,Ns,(M1,i))
end

"""
    C = kron(F,Op,i,Ns)

Creates the kronecker product of an input operator `Op` (Array) on site `i` of an `Ns` site system with a trailing operator `F`
"""
function kron(F::Array,M1::Array,i::intType,Ns::intType)
  return kron(tens(F),Ns,(tens(M1),i))
end

#=
"""
    C = kron(Op,Ns)

Creates the kronecker product of an input operator `Op` on all sites for an `Ns` site system
"""
function kron(M1::TensType,Ns::intType)
  return sum(w->kron(Ns,(M1,w)),1:Ns)
end
=#
"""
    C = kron(Op,i,Op2,j,Ns)

Creates the kronecker product of an input operator `Op` on site `i` and `Op2` on site `j` of an `Ns` site system
"""
function kron(M1::TensType,i::intType,M2::TensType,j::intType,Ns::intType)
  return kron(Ns,(M1,i),(M2,j))
end

"""
    C = kron(Op,i,Op2,j,Op3,k,Ns)

Creates the kronecker product of an input operator `Op` on site `i`, `Op2` on site `j`, and `Op3` on site `k` of an `Ns` site system
"""
function kron(M1::TensType,i::intType,M2::TensType,j::intType,M3::TensType,k::intType,Ns::intType)
  return kron(Ns,(M1,i),(M2,j),(M3,k))
end

"""
    C = kron(Op,i,Op2,j,Op3,k,Op4,l,Ns)

Creates the kronecker product of an input operator `Op` on site `i`, `Op2` on site `j`, `Op3` on site `k`, and `Op4` on site `l` of an `Ns` site system
"""
function kron(M1::TensType,i::intType,M2::TensType,j::intType,M3::TensType,k::intType,M4::TensType,l::intType,Ns::intType)
  return kron(Ns,(M1,i),(M2,j),(M3,k),(M4,l))
end

"""
    C = kron(Ns,Op,i)

Creates the kronecker product of an input operator `Op` on site `i` of an `Ns` site system
"""
function kron(Ns::intType,M1::TensType,i::intType)
  return kron(Ns,(M1,i))
end

"""
    C = kron(Ns,Op,i,Op2,j)

Creates the kronecker product of an input operator `Op` on site `i` and `Op2` on site `j` of an `Ns` site system
"""
function kron(Ns::intType,M1::TensType,i::intType,M2::TensType,j::intType)
  return kron(Ns,(M1,i),(M2,j))
end

"""
    C = kron(Ns,Op,i,Op2,j,Op3,k)

Creates the kronecker product of an input operator `Op` on site `i`, `Op2` on site `j`, and `Op3` on site `k` of an `Ns` site system
"""
function kron(Ns::intType,M1::TensType,i::intType,M2::TensType,j::intType,M3::TensType,k::intType)
  return kron(Ns,(M1,i),(M2,j),(M3,k))
end

"""
    C = kron(Ns,Op,i,Op2,j,Op3,k,Op4,l)

Creates the kronecker product of an input operator `Op` on site `i`, `Op2` on site `j`, `Op3` on site `k`, and `Op4` on site `l` of an `Ns` site system
"""
function kron(Ns::intType,M1::TensType,i::intType,M2::TensType,j::intType,M3::TensType,k::intType,M4::TensType,l::intType)
  return kron(Ns,(M1,i),(M2,j),(M3,k),(M4,l))
end

"""
    kron(Op,a)

Multiplies operator `Op` by number `a`
"""
function kron(M1::TensType,num::Number)
  return M1*num
end

"""
    kron(a,Op)

Multiplies operator `Op` by number `a`
"""
function kron(num::Number,M1::TensType)
  return M1*num
end

"""
    kron(A,B)

Kroneckers diagonal type `A` by `TensType` `B`
"""
function kron(A::diagonal,B::TensType)
  return kron(Array(A),B)
end

"""
    kron(A,B)

Kroneckers diagonal type `B` by `TensType` `A`
"""
function kron(A::TensType,B::diagonal)
  return kron(A,Array(B))
end
