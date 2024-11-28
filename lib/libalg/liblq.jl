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

function liblq!(A::Array{W,P},a::Integer,b::Integer;m::Integer=min(a,b)) where {W <: Number, P}
  tau = Array{W,1}(undef,m)
  L = libLowerHessenberg!(A,a,b,tau=tau) #gelqf!

  sizetau = length(tau)
#  @inbounds while sizetau > 0 && !isapprox(tau[sizetau],0)
#    sizetau -= 1
#  end

  Q = libLQmake!(A,a,b,tau,sizetau) #orglq

  return L,Q
end

function liblq(A::Array{W,2},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  return liblq!(copy(A),a,b,m=m)
end

function liblq(A::tens{W},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  L,Q = liblq!(copy(A.T),a,b,m=m)
  return tens{W}([a,m],L),tens{W}([m,b],Q)
end

function liblq!(A::tens{W},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  L,Q = liblq!(A.T,a,b,m=m)
  return tens{W}([a,m],L),tens{W}([m,b],Q)
end
