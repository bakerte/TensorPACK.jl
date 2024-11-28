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

const libQRmake! = orgqr!
const libLQmake! = orglq!
const libRQmake! = orgrq!
const libQLmake! = orgql!

function libUpperHessenberg!(A::Array{W,P};a::Integer=size(A,1),b::Integer=size(A,2),tau::Array{W,1}=Array{W,1}(undef,min(a,b)),m::Integer=length(tau)) where {W <: Number, P}
  return libUpperHessenberg!(A,a,b,tau=tau,m=m)
end

function libUpperHessenberg!(A::Array{W,P},a::Integer,b::Integer,tau::Array{W,1};m::Integer=length(tau)) where {W <: Number, P}
  return libUpperHessenberg!(A,a,b,tau=tau,m=m)
end

function libUpperHessenberg!(A::Array{W,P},a::Integer,b::Integer;tau::Array{W,1}=Array{W,1}(undef,min(a,b)),m::Integer=length(tau)) where {W <: Number, P}
  G,tau = geqrf!(A,a,b,tau)

  R = P == 1 ? Array{W,1}(undef,m*b) : Array{W,2}(undef,m,b)
  for y = 1:b
    thisind = m*(y-1)
    thisotherind = a*(y-1)
    @inbounds @simd for x = 1:min(y,m)
      R[x + thisind] = A[x + thisotherind]
    end
    @inbounds @simd for x = y+1:m
      R[x + thisind] = 0
    end
  end
  return R
end
export libUpperHessenberg!




function libLowerHessenberg!(A::Array{W,P};a::Integer=size(A,1),b::Integer=size(A,2),tau::Array{W,1}=Array{W,1}(undef,min(a,b)),m::Integer=length(tau)) where {W <: Number, P}
  return libLowerHessenberg!(A,a,b,tau=tau,m=m)
end

function libLowerHessenberg!(A::Array{W,P},a::Integer,b::Integer,tau::Array{W,1};m::Integer=length(tau)) where {W <: Number, P}
  return libLowerHessenberg!(A,a,b,tau=tau,m=m)
end

function libLowerHessenberg!(A::Array{W,P},a::Integer,b::Integer;tau::Array{W,1}=Array{W,1}(undef,min(a,b)),m::Integer=length(tau)) where {W <: Number, P}
  G,tau = gelqf!(A,a,b,tau)

  L = P == 1 ? Array{W,1}(undef,a*m) : Array{W,2}(undef,a,m)
  for y = 1:m
    thisind = a*(y-1)
    @inbounds @simd for x = 1:y-1
      L[x+thisind] = 0
    end
    @inbounds @simd for x = y:a
      L[x+thisind] = A[x+thisind]
    end
  end
  return L
end
export libLowerHessenberg!
