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
    C = trace!(A,iA)

Computes trace of `A` (any `TensType`) over indices `iA` by first making an identity tensor among those indices; `iA` is a vector of tuples (ex: (1,3)) or a vector of 2-element vectors ([(1,2),(3,4),(5,6)])

# Example:

```julia
julia> A = [1 0;0 -1];
julia> trace(A,[1,2])
0-dimensional Array{Float64,0}:
0.0

julia> A = ones(10,20,10,20);
julia> trace(A,[(1,3),(2,4)])
0-dimensional Array{Float64,0}:
200.0
```
"""
function trace!(A::TensType,iA::Array{NTuple{2,P},1}) where P <: Integer
  Id = eye(A,iA)
  conA = (iA[1]...,)
  for w = 2:length(iA)
    conA = (conA...,iA[w]...)
  end

  conId = ntuple(w->w,2*length(iA))

  return contract(A,conA,Id,conId)
end

"""
    C = trace!(A,iA)

Computes trace of `A` over indices `iA` by first making an identity tensor among those indices; `iA` is a vector of numbers (ex: [1,3]) or a vector of 2-element vectors ([[1,2],[3,4],[5,6]])

# Example:

```julia
julia> A = [1 0;0 -1];
julia> trace(A,[1,2])
0-dimensional Array{Float64,0}:
0.0

julia> A = ones(10,20,10,20);
julia> trace(A,[[1,3],[2,4]])
0-dimensional Array{Float64,0}:
200.0
```
"""
function trace!(A::TensType,iA::Array{Array{P,1},1}) where P <: Integer
  return trace!(A,[convIn(iA[w]) for w = 1:length(iA)])
end

"""
    C = trace!(A,iA)

Computes trace of `A` over indices `iA` by first making an identity tensor among those indices; `iA` is a vector of numbers (ex: [1,3]) or a vector of 2-element vectors ([1,2],[3,4],[5,6])

# Example:

```julia
julia> A = [1 0;0 -1];
julia> trace(A,[1,2])
0-dimensional Array{Float64,0}:
0.0

julia> A = ones(10,20,10,20);
julia> trace(A,[1,3],[2,4])
0-dimensional Array{Float64,0}:
200.0
```
"""
function trace!(A::TensType,iA::Array{P,1}...) where P <: Integer
  return trace!(A,[convIn(iA[w]) for w = 1:length(iA)])
end

"""
    C = trace!(A)

Find trace of a matrix `A` (any `TensType` but sensical if only two indices)
"""
function trace!(A::TensType)
  return sum(w->A[w,w],1:size(A,1))
end
export trace!

"""
    C = trace(A,iA)

Computes trace of `A` (copying `A`) over indices `iA` by first making an identity tensor among those indices; `iA` is a vector of numbers or tuples (ex: [1,3] or (1,3)) or a vector of 2-element vectors ([[1,2],[3,4],[5,6]])

# Example:

```julia
julia> A = [1 0;0 -1];
julia> trace(A,[1,2])
0-dimensional Array{Float64,0}:
0.0

julia> A = ones(10,20,10,20);
julia> trace(A,[[1,3],[2,4]])
0-dimensional Array{Float64,0}:
200.0
```
"""
function trace(A::TensType,iA::R...) where R <: Union{Array{P,1},Array{Array{P,1},1},Array{NTuple{2,P},1}} where P <: Integer
  return trace!(copy(A),iA...)
end
export trace



"""
    C = trace(A)

Computes the trace of named tensor `A` (rank-2) over indices with 1) the same name and 2) opposite arrows (not checked)
"""
function trace(B::TNobj)
  if ndims(B) == 2
    return trace(B.N,[1,2])
  else
    error("undefined trace of tensor with rank > 2")
  end
end

"""
    C = trace(A,inds)

Computes the trace of named tensor `A` with specified `inds` (integers, symbols, or strings--ex: [[1,2],[3,4],[5,6]])
"""
function trace(A::nametens,inds::Array{W,1}) where W <: String
  if W <: Integer
    error("Why are you putting integers to trace a nametens? Should it not be strings? Can extract unnamed tensor with A.N or similar")
#    return trace(A.N,inds)
  else

    vect = Array{intType,1}(undef,2)
    for a = 1:length(vect)
      c = 1
      while inds[a] != A.names[c]
        c += 1
      end
      vect[a] = c
    end

    B = trace(A.N,vect)

    newnames = Array{eltype(A.names),1}(undef,length(A.names)-length(vect))
    w = 0
    counter = 0
    while w < length(A.names) && counter < length(newnames)
#    for w = 1:length(newnames)
      w += 1
      if !(w in vect)
        counter += 1
        newnames[counter] = A.names[w]
      end
    end
#    newnames = setdiff(A.names,inds)

#    println(newnames)

    return nametens(B,newnames)
  end
end
#=
function trace(A::directedtens,inds::Array{Array{W,1},1}) where W <: Union{Any,Integer}
  B = trace(A.T,inds)
  newinds = vcat(inds...)
  leftoverinds = setdiff([i for i = 1:ndims(A)],newinds)
  newarrows = A.arrows[leftoverinds]
  return directedtens(B,newarrows,A.conj)
end
=#
"""
    C = trace(A,inds)

Computes the trace of named tensor `A` with specified `inds` (integers, symbols, or strings--ex: [1,2])
"""
function trace(A::TNobj,inds::Array{W,1}) where W <: Union{Any,Integer}
  return trace(A,inds)
end
export trace





"""
    C = trace(A,inds)

Computes the trace of `dtens` `A` with specified `inds` (integers, symbols, or strings--ex: [1,2])
"""
function trace!(A::dtens)
  return sum(w->searchindex(A[0],w,w),1:size(A[0],1)),sum(w->searchindex(A[1],w,w),1:size(A[1],1))
end
