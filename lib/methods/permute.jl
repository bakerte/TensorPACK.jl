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
    B = permutedims!(A,[1,3,2,...])

Permute dimensions of an Array or `denstens` in-place (no equivalent, but returns value so it can be used identically to `permutedims` on a dense tensor)

See also: [`permutedims`](@ref)
"""
function permutedims!(M::tens{W}, vec::Array{P,1}) where {P <: intType, W <: Number}
  return permutedims(M,(vec...,))
end

"""
    B = permutedims!(A,[1,3,2,...])

Permute dimensions of an Array in-place (no equivalent, but returns value so it can be used identically to `permutedims` on a dense tensor)

See also: [`permutedims`](@ref)
"""
function permutedims!(M::Array{W,G}, vec::Array{P,1}) where {P <: intType, G, W <: Number}
  return permutedims(M,(vec...,))
end

"""
    B = permutedims!(A,(1,3,2,...))

Permute dimensions of an Array or `denstens` (tuple input) in-place (no equivalent, but returns value so it can be used identically to `permutedims` on a dense tensor)

See also: [`permutedims`](@ref)
"""
function permutedims!(M::tens{W}, vec::NTuple{N,intType}) where {N, W <: Number}
  return permutedims(M,vec)
end

"""
    B = permutedims!(A,(1,3,2,...))

Permute dimensions of an Array (tuple input) in-place (no equivalent, but returns value so it can be used identically to `permutedims` on a dense tensor)

See also: [`permutedims`](@ref)
"""
function permutedims!(M::Array{W,G}, vec::NTuple{N,intType}) where {N, G, W <: Number}
  return permutedims(M,vec)
end

"""
    B = permutedims!(A,[1,3,2,...])

Permute dimensions of a `diagonal` in-place (no equivalent, but returns value so it can be used identically to `permutedims` on a dense tensor)

See also: [`permutedims`](@ref)
"""
function permutedims!(M::diagonal, vec::Union{Array,Tuple,Memory})
  return M
end

"""
    B = permutedims(A,[1,3,2,...])

Permute dimensions of a `diagonal` with either vector or tuple input

See also: [`permutedims`](@ref)
"""
function permutedims(M::diagonal, vec::Union{Array,Tuple,Memory})
  return M
end

"""
  G = permutedims(A,[1,3,2,...])

Permutes dimensions of `A` (identical usage to dense `size` call) for a `denstens`

See also: [`permutedims!`](@ref)
"""
function permutedims(M::tens{W}, vec::Union{Memory{P},Array{P,1}}) where {W <: Number, P <: intType}
  return permutedims(M,(vec...,))
end

"""
  G = permutedims(A,[1,3,2,...])

Permutes dimensions of `A` (identical usage to dense `size` call) for an Array

See also: [`permutedims!`](@ref)
"""
function permutedims(M::Array{W,G}, vec::Union{Memory{P},Array{P,1}}) where {W <: Number, P <: intType, G}
  return permutedims(M,(vec...,))
end

"""
  G = permutedims(A,(1,3,2,...))

Permutes dimensions of `A` (input tuple) for an Array

See also: [`permutedims!`](@ref)
"""
function permutedims(A::Array{W,G},iA::NTuple{G,intType}) where {W <: Number, G}
  if issorted(iA)
    out = A
  else
    Asizes = ntuple(w->size(A,w),G)
    newsizes = ntuple(w->Asizes[iA[w]],G)

    out = Array{W,G}(undef,newsizes...)

    permutedims!(out,A,iA,Asizes,newsizes)
  end
  return out
end

"""
    permutedims!(B,A,iA,Asizes,newsizes)

Permute dimensions of an input Array `A` output to array `B` (any rank) with permutation rule `iA`, original sizes of the `A` tensor `Asizes` and size of output `B` tensor `Bsizes`

See also: [`permutedims`](@ref)
"""
function permutedims!(P::Union{Array{W,R},Memory{W}},A::Union{Array{W,R},Memory{W}},iA::Union{Array{intType,1},NTuple{G,intType}},Asizes::Union{Array{intType,1},NTuple{G,intType},Memory{intType}},newsizes::Union{Array{intType,1},NTuple{G,intType},Memory{intType}}) where {W <: Number, G, R}

  if length(Asizes) != length(newsizes)
    error("Non-matching rank of input tensor for permuted dimensions rule")
  end

  startind = 0
  @inbounds while startind < G && iA[startind+1] == startind+1
    startind += 1
  end

  permutevec = makepos(G)
  if startind > 0
    startindex = 1
    @inbounds @simd for w = 1:startind
      startindex *= newsizes[w]
      permutevec[w] = newsizes[w]
    end
    @inbounds @simd for w = 1:startindex
      P[w] = A[w]
    end
  else
    startindex = 0
  end

  d1 = 1
  @inbounds while iA[d1] != 1
    d1 += 1
  end
  
  notfirst = d1 > 1
  if notfirst
    factor = 1
    @inbounds @simd for h = d1-1:-1:1
      factor *= newsizes[h]
    end
  end

  i = startindex
  @inbounds while i < length(A) 
    i += 1
    position_incrementer!(permutevec,Asizes)
    newvec = ntuple(w->permutevec[iA[w]],G)

    backZ = newvec[G]
    @inbounds @simd for w = G-1:-1:d1
      backZ -= 1
      backZ *= newsizes[w]
      backZ += newvec[w]
    end

    if notfirst
      foreZ = newvec[d1-1]
      @inbounds @simd for w = d1-2:-1:1
        foreZ -= 1
        foreZ *= newsizes[w]
        foreZ += newvec[w]
      end

      @inbounds @simd for x = 0:Asizes[1]-1
        z = x + backZ
        z -= 1
        z *= factor
        z += foreZ

        P[z] = A[i+x]
      end
    else
      @inbounds @simd for x = 0:Asizes[1]-1
        z = x + backZ
        P[z] = A[i+x]
      end
    end
    permutevec[1] = Asizes[1]
    i += Asizes[1]-1
  end
  nothing
end

"""
  G = permutedims(A,(1,3,2,...))

Permutes dimensions of `A` (input tuple) for a `denstens`

See also: [`permutedims!`](@ref)
"""
function permutedims(A::tens{W},iA::NTuple{G,intType}) where {W <: Number, G}

  if issorted(iA)
    out = A
  else
    Asizes = size(A) #ntuple(w->size(A,w),G)
    newsizes = Memory{intType}(undef,length(iA))
    for w = 1:length(newsizes)
      newsizes[w] = Asizes[iA[w]]
    end
#    newsizes = [Asizes[iA[w]] for w = 1:length(iA)] #ntuple(w->Asizes[iA[w]],G)

    psize = 1
    @inbounds @simd for w = 1:G
      psize *= Asizes[w]
    end
    P = Memory{W}(undef,psize)

    permutedims!(P,A.T,iA,Asizes,newsizes)

#    vecnewsizes = newsizes #ntuple(w->newsizes[w],G)
    out = tens{W}(newsizes,P)
  end
  return out
end

"""
    permtuedims(A,order)

Permutes named tensor `A` according to `order` (ex: [[1,2],[3,4]] or [["a","b"],["c","d"]])

See also: [`permutedims!`](@ref)
"""
function permutedims(A::TNobj,order::Array{W,1}) where W <: Union{String,Integer}
  B = copy(A)
  return permutedims!(B,order)
end

#differentiate between case for integers (above but wrong code) and for labels
#get "not labels" for set diff of the labels we know and don't know
"""
    permtuedims!(A,order)

Permutes named tensor `A` according to `order` (ex: ["a","d","c","b"])

See also: [`permutedims`](@ref)
"""
function permutedims!(A::TNobj,order::Array{W,1}) where W <: String
  intorder = Array{intType,1}(undef,length(order))
  for w = 1:length(order)
    checkname = false
    x = 0
    while !checkname && x < length(A.names)
      x += 1
      checkname = order[w] == A.names[x]
    end
    if checkname
      intorder[w] = x
    else
      error("name not found on tensor that was requested to permute...make input names to permutedims match input tensor")
    end
  end
  A.names = order
  permutedims!(A.N,intorder)
  return A
end



"""
    permutedims(A,[1,3,2,...])

permutes dimensions of `A`  (identical usage to dense `size` call)

See also: [`permutedims!`](@ref)
"""
function permutedims(currQtens::Qtens{W,Q}, vec::Union{NTuple{N,P},Array{P,1}}) where {N, W <: Number, Q <: Qnum, P <: Integer}
  Qtens = copy(currQtens)
  permutedims!(Qtens, vec)
  return Qtens
end

"""
    permutedims!(A,[1,3,2,...])

permute dimensions of a Qtensor in-place (no equivalent, but returns value so it can be used identically to `permutedims` on a dense tensor)

See also: [`permutedims`](@ref)
"""
function permutedims!(currQtens::Qtens{W,Q}, vec::Union{NTuple{N,P},Array{P,1}}) where {N, W <: Number, Q <: Qnum, P <: Integer}

  Rsize = currQtens.size
  totalordersize = sum(q->length(Rsize[q]),1:length(Rsize))
  order = Array{intType,1}(undef,totalordersize)
  count = 0

  for i = 1:length(vec)
    @inbounds @simd for j = 1:length(Rsize[vec[i]])
      count += 1
      order[count] = Rsize[vec[i]][j]
    end
  end

  permorder = Array{intType,1}(undef,length(order))

  newRsize = Array{Array{intType,1},1}(undef,length(vec))
  @inbounds for q = 1:length(vec)
    newRsize[q] = Rsize[vec[q]]
  end
  counting = 0
  for k = 1:length(Rsize)
    @inbounds @simd for m = 1:length(Rsize[k])
      counting += 1
      permorder[order[counting]] = counting
    end
  end

  counter = 0
  for w = 1:length(newRsize)
    @inbounds @simd for a = 1:length(newRsize[w])
      counter += 1
      newRsize[w][a] = counter
    end
  end

  currQtens.size = newRsize
  currQtens.QnumMat = currQtens.QnumMat[[order...]]
  currQtens.QnumSum = currQtens.QnumSum[[order...]]

  first = [permorder[currQtens.currblock[1][i]] for i = 1:length(currQtens.currblock[1])]
  second = [permorder[currQtens.currblock[2][i]] for i = 1:length(currQtens.currblock[2])]
  currQtens.currblock = (first,second)

  return currQtens
end
