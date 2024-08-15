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
    G = tens([type=Float64])

Initializes an empty tensor `G` with no indices
"""
function tens(;type::DataType=Float64)
  return tens{type}((intType(0),),type[])
end

"""
    G = tens(type)

Initializes an empty tensor `G` with no indices of data-type `type`
"""
function tens(type::DataType)
  return tens{type}((intType(0),),type[])
end

"""
    G = tens{W}()

Initializes an empty tensor `G` with no indices of data-type `W`
"""
function tens{T}() where T <: Number
  return tens(type=T)
end

"""
    A = tens(A)

Trivial convertion of denstens `A` to itself
"""
function tens(A::denstens)
  return A
end

"""
    A = tens(A)

Trivial convertion of denstens `A` to itself
"""
function tens(A::diagonal)
  return tens(LinearAlgebra.diagm(A.T))
end

"""
    A = tens(A)

Trivial convertion of denstens `A` to itself
"""
function tens{W}(A::diagonal{Z}) where {W <: Number, Z <: Number}
  if Z != W
    outtens = tens(LinearAlgebra.diagm(convert(Array{W,1},A.T)))
  else
    outtens = tens(A)
  end
  return outtens
end

#=
function tens(G::DataType,P::AbstractArray{W,N}) where W <: Number where N
  rP = reshape(Array(P),prod(size(P)))
  Psize = ntuple(w->size(P,w),N) #[size(P,w) for w = 1:N]
  return tens{G}(Psize,rP)
end
=#
"""
    G = tens(W,P)

Converts tensor `P` into a `denstens` and converts to type `W` for output tensor `G`

See also: [`denstens`](@ref)
"""
function tens(G::DataType,P::AbstractArray{W,N}) where W <: Number where N
  sizeP = ntuple(w->size(P,w),N)
  #sizeP = ntuple(w->size(P,w),ndims(P)) #size of P to a vector
  vecP = reshape(P,prod(sizeP))
  if G != eltype(P) #converts types if they do not match
    rP = convert(Array{G,1},vecP)
  else
    rP = vecP
  end
  return tens{G}(sizeP,rP)
end

"""
    G = tens(W,P)

Converts diagonal type tensor `P` to a `denstens` type

See also: [`denstens`](@ref)
"""
function tens(G::DataType,P::Diagonal{W}) where W <: Number
  if G != eltype(P) #converts types if they do not match
    rP = Diagonal(convert(Array{G,1},P.T))
  else
    rP = P
  end
  return tens(rP)
end
#=
"""
    G = tens(tsize,P)

Converts `denstens` `P` back into `denstens` of size `tsize` (mainly for interface in some places)

See also: [`denstens`](@ref)
"""
function tens(tsize::Tuple,P::tens{W}) where {W <: Number}

  println(tsize)
  println(size(P))
  println(length(P.T))

  return tens{W}(tsize,P.T)
end
=#
"""
    G = tens(P)

Converts tensor `P` into a `denstens` (`G`) of the same type

See also: [`denstens`](@ref)
"""
function tens(P::Array{W,N}) where {W <: Number, N}
  return tens(W,P)
end

function tens(P::AbstractArray{W,N}) where {W <: Number, N}
  return tens(eltype(P),Array(P))
end

"""
    G = tens{W}(P)

Converts array `P` into a `denstens` (`G`) and converts to type `W`

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function tens{W}(P::Array{G,N}) where {W <: Number, G <: Number} where N
  return tens(W,P)
end

"""
    G = tens{W}(P)

Converts tensor `P` into the `denstens` (`G`) and converts to type `W`

See also: [`denstens`](@ref)
"""
function tens{W}(P::tens{Z}) where {W <: Number, Z <: Number}
  if W != Z
    newtens = tens{W}(P.size,convert(Array{W,1},P.T))
  else
    newtens = P
  end
  return newtens
end

"""
    G = tens(W,P)

Converts tensor `P` into the `denstens` (`G`) and converts to type `W`

See also: [`denstens`](@ref)
"""
function tens(W::DataType,P::tens{Z}) where {Z <: Number}
  return tens{W}(P)
end









"""
    A = tens(Qt)

converts Qtensor (`Qt`) to dense array
"""
function tens(QtensA::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  truesize = basesize(QtensA)
  Lsizes = truesize[QtensA.currblock[1]]
  Rsizes = truesize[QtensA.currblock[2]]
  Ldim = prod(Lsizes)
  Rdim = prod(Rsizes)
  
  G = zeros(W,truesize)

  newpos = Array{intType,1}(undef,length(QtensA.QnumMat))  
  @inbounds for q = 1:length(QtensA.ind)
    thisTens = QtensA.T[q]
    theseinds = QtensA.ind[q]

    @inbounds for y = 1:size(thisTens,2)
      @inbounds for n = 1:length(QtensA.currblock[2])
        rr = QtensA.currblock[2][n]
        newpos[rr] = theseinds[2][n,y] + 1
      end
      for x = 1:size(thisTens,1)
        @inbounds @simd for m = 1:length(QtensA.currblock[1])
          bb = QtensA.currblock[1][m]
          newpos[bb] = theseinds[1][m,x] + 1
        end
        G[newpos...] = thisTens[x,y]
      end
    end
  end
  return tens{W}(G)
end











































#=
import Base.iterate
"""
  iterate(T)

Function to provide vectorized operations for denstens class
"""
function iterate(A::tens)
  return A.T
end
=#