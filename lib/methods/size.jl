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
    G = size(A)

Outputs tuple `G` representing the size of a `denstens` `A` (identical usage to `Array` `size` call)

See also: [`tupsize`](@ref) [`vecsize`](@ref)
"""
function size(A::denstens)
  return A.size
end

"""
    G = tupsize(A)

Returns size of input tensor `A` but as a tuple instead of a vector

See also: [`size`](@ref) [`denstens`](@ref) [`Array`](@ref)
"""
function tupsize(A::Union{TensType,diagonal})
  return ntuple(b->size(A,b),ndims(A)) #(A.size...,)
end

"""
    G = vecsize(A)

Returns size of input tensor `A` but as a vector

See also: [`size`](@ref) [`denstens`](@ref) [`Array`](@ref)
"""
function vecsize(A::Union{TensType,diagonal})
  output = Array{intType,1}(undef,ndims(A))
  for w = 1:ndims(A)
    output[w] = A.size[w]
  end
  return output
end

"""
    G = size(A)

Outputs tuple `G` representing the size of a `diagonal` `A`

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function size(A::diagonal)
  output = Memory{intType}(undef,2)
  output[1] = output[2] = length(A)
  return output
end

"""
    G = size(A,i)

Gets the size of index `i` of a `denstens` `A` (identical usage to `Array` `size` call) as output `G`

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function size(A::denstens,i::Integer)::intType
  return i > ndims(A) ? 1 : A.size[i]
end

"""
    G = size(A,i)

Gets the size of index `i` of a `diagonal` `A` as output `G`

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function size(A::diagonal,i::Integer)::intType
  return i == 1 || i == 2 ? length(A) : 1
end





"""
    size(A)

Gives the size of `TNobj` `A`
"""
function size(A::TNobj)
  return size(A.N)
end

"""
    size(A,w)

Gives the size of `TNobj` `A` where `w` is the index (integer)
"""
function size(A::TNobj,w::Integer)
  return size(A.N,w)
end

"""
    size(A,w)

Gives the size of `TNobj` `A` where `w` is the index (String)
"""
function size(A::TNobj,w::String)
  condition = true
  p = 0
  while condition && p < ndims(A)
    p += 1
    condition = A.names[p] != w
  end
  return size(A.N,p)
end

"""
    size(A)

Gives the size of `dtens` `A`
"""
function size(A::dtens)
  return size(A[0])
end

"""
    size(A,w)

Gives the size of `dtens` `A` where `w` is the index (integer)
"""
function size(A::dtens,i::Integer)
  return size(A[0],i)
end

"""
    size(A[,i])

gets the size of a Qtensor (identical usage to dense `size` call)
"""
function size(A::qarray, i::intType)::intType
  return prod(w->length(A.QnumMat[w]),A.size[i])
end

"""
    size(A)

size of `qarray` `A`
"""
function size(A::qarray)
  return ntuple(w->size(A,w),ndims(A))
end
