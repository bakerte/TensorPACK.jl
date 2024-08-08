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

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function size(A::tens)
  return (A.size...,)
end

function size(A::diagonal)
  return (length(A),length(A))
end

"""
  G = size(A,i)

Gets the size of index `i` of a `denstens` `A` (identical usage to `Array` `size` call) as output `G`

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function size(A::tens,i::Integer)::intType
  return A.size[i]
end

function size(A::diagonal,i::Integer)::intType
  return i == 1 || i == 2 ? length(A) : 1
end





"""
    size(A[,w=])

Gives the size of named tensor `A` where `w` is an integer or an index label
"""
function size(A::TNobj)
  return size(A.N)
end

function size(A::TNobj,w::Integer)
  return size(A.N,w)
end

function size(A::TNobj,w::String)
  condition = true
  p = 0
  while condition && p < ndims(A)
    p += 1
    condition = A.names[p] != w
  end
  return size(A.N,p)
end

function size(A::dtens,i)
  return size(A[0],i)
end

function size(A::dtens)
  return size(A[0])
end

"""
    size(A[,i])

gets the size of a Qtensor (identical usage to dense `size` call)
"""
function size(A::qarray, i::intType)::intType
  return prod(w->length(A.QnumMat[w]),A.size[i])
end

function size(A::qarray)
  return ntuple(w->size(A,w),ndims(A))
end
