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
    G,K = checkType(A,B)

converts both of `A` and `B` to `denstens` (`G` and `K`) if they are not already and the types are mixed between AbstractArrays and `denstens`
"""
function checkType(A::R,B::S) where {R <: AbstractArray, S <: denstens}
  outType = typeof(eltype(A)(1)*eltype(B)(1))
  return tens{outType}(checkType(A)),tens{outType}(B)
end

function checkType(A::S,B::R) where {R <: AbstractArray, S <: denstens}
  outType = typeof(eltype(A)(1)*eltype(B)(1))
  return tens{outType}(A),tens{outType}(checkType(B))
end

#for abstract arrays, we only care about the element type
function checkType(A::AbstractArray{R,N},B::AbstractArray{S,M}) where {R <: Number, S <: Number, N, M}
  outType = typeof(eltype(A)(1)*eltype(B)(1))
  return convert(Array{outType,N},checkType(A)),convert(Array{outType,M},checkType(B))
end

function checkType(A::AbstractArray)
  if typeof(A) <: Array
    out = A
  else
    out = Array(A)
  end
  return out
end

function checkType(A::denstens)
  return A
end

function checkType(A::tens{W},B::tens{G}) where {W <: Number, G <: Number}
  outType = typeof(eltype(A)(1)*eltype(B)(1))
  return tens{outType}(A),tens{outType}(B)
end

function checkType(A::diagonal{W},B::tens{G}) where {W <: Number, G <: Number}
  if W == G
    mA,mB = A,B
  else
    outType = typeof(eltype(A)(1)*eltype(B)(1))
    mA,mB = diagonal{outType}(A.T),tens{outType}(B)
  end
  return mA,mB
end

function checkType(A::tens{W},B::diagonal{G}) where {W <: Number, G <: Number}
  if W == G
    mA,mB = A,B
  else
    outType = typeof(eltype(A)(1)*eltype(B)(1))
    mA,mB = tens{outType}(A),diagonal{outType}(B.T)
  end
  return mA,mB
end

function checkType(A::diagonal{W},B::diagonal{G}) where {W <: Number, G <: Number}
  if W == G
    mA,mB = A,B
  else
    outType = typeof(eltype(A)(1)*eltype(B)(1))
    mA,mB = diagonal{outType}(A),diagonal{outType}(B.T)
  end
  return mA,mB
end
export checkType


"""
    G,K = checkType(A,B)

converts both of `A` and `B` to `Qtens` (`G` and `K`) if they are not already and the types are mixed between element types
"""
function checkType(A::Qtens{W,Q},B::Qtens{G,Q}) where {W <: Number, G <: Number, Q <: Qnum}
  outType = typeof(eltype(A)(1)*eltype(B)(1))
  return Qtens(outType,A),Qtens(outType,B)
end

