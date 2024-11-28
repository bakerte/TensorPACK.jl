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


function libqr(A::Array{W,2},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  return libqr!(copy(A),a,b,m=m)
end

function libqr(A::tens{W},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  Q,R = libqr!(copy(A.T),a,b,m=m)
  return tens{W}([a,m],Q),tens{W}([m,b],R)
end

function libqr!(A::tens{W},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  Q,R = libqr!(A.T,a,b,m=m)
  return tens{W}([a,m],Q),tens{W}([m,b],R)
end


function libqr!(A::Array{W,P},a::Integer,b::Integer;m::Integer=min(a,b)) where {W <: Number, P}
  tau = Array{W,1}(undef,m)
  R = libUpperHessenberg!(A,a,b,tau=tau) #geqrf!

  sizetau = length(tau)
#  @inbounds while sizetau > 0 && !isapprox(tau[sizetau],0)
#    sizetau -= 1
#  end

  Q = libQRmake!(A,a,b,tau,sizetau) #orgqr
  return Q,R
end