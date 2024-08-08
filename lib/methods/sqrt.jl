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
  G = sqrt!(M)

Takes the square root of a dense tensor (new tensor created) or Qtensor (in-place) with output `G`
"""
function sqrt!(M::TensType)
  return tensorcombination!(M,alpha=(0.5,),fct=^)
end

function sqrt!(M::diagonal{W}) where W <: Number
  @inbounds @simd for i = 1:length(M)
    M[i] = sqrt(M[i])
  end
  return M
end
export sqrt!

"""
  G = sqrt(M)

Takes the square root of a tensor with output `G`

See also: [`sqrt`](@ref)
"""
function sqrt(M::TensType)
  return tensorcombination(M,alpha=(0.5,),fct=^)
end

function sqrt(M::diagonal{W}) where W <: Number
  return sqrt!(copy(M))
end

function abs(x::Number,a::Number)
  return abs(x)
end

"""
  G = sqrtabs!(M)

Takes the square root of a dense tensor (new tensor created) or Qtensor (in-place) with output `G`
"""
function sqrtabs!(M::TensType)
  M = tensorcombination!(M,fct=abs)
  return tensorcombination!(M,alpha=(0.5,),fct=^)
end

function sqrtabs!(M::diagonal{W}) where W <: Number
  @inbounds @simd for i = 1:length(M)
    M[i] = sqrt(abs(M[i]))
  end
  return M
end
export sqrt!

"""
  G = sqrt(M)

Takes the square root of a tensor with output `G`

See also: [`sqrt`](@ref)
"""
function sqrtabs(M::TensType)
  G = tensorcombination(M,fct=abs)
  return tensorcombination!(G,alpha=(0.5,),fct=^)
end

function sqrtabs(M::diagonal{W}) where W <: Number
  return sqrt!(copy(M))
end


"""
    sqrt(A)

Takes the square root of named tensor `A`

See also: [`sqrt`](@ref)
"""
function sqrt(A::TNobj;root::Number=0.5)
  B = copy(A)
  return sqrt!(B,root=root)
end

"""
    sqrt!(A)

Takes the square root of named tensor `A`

See also: [`sqrt!`](@ref)
"""
function sqrt!(A::TNobj;root::Number=0.5)
  A.N = tensorcombination!(A.N,alpha=(root,),fct=^)#sqrt!(A.N,root=root)
  return A
end

"""
    sqrt(A)

Takes the square root of named tensor `A`

See also: [`sqrt`](@ref)
"""
function sqrtabs(A::TNobj;root::Number=0.5)
  B = copy(A)
  return sqrtabs!(B,root=root)
end

"""
    sqrt!(A)

Takes the square root of named tensor `A`

See also: [`sqrt!`](@ref)
"""
function sqrtabs!(A::TNobj;root::Number=0.5)
  A.N = tensorcombination!(A.N,fct=abs)
  A.N = tensorcombination!(A.N,alpha=(root,),fct=^)#sqrt!(A.N,root=root)
  return A
end

"""
  sqrt(x)

Square-root of a `dualnum`
"""
sqrt(x::dualnum) = x^0.5
