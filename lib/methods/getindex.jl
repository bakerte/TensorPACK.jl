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
  G = getindex(M,a...)

Selects subset or single element `G` of input tensor `M` with elements of `a` being of `genColType` (ranges or values in tensor)

For example, `A[:,3:6,2,[1,2,4,8]]`

See also: [`searchindex`](@ref) [`getindex!`](@ref) [`denstens`](@ref) [`genColType`](@ref)
"""
function getindex(M::denstens, a::genColType...)
  return getindex!(M,a...)
end

function getindex(C::tens{W}, a::G...)where W <: Number where G <: Array{Bool,1}
  M = Array(C)
  return tens{W}(M[a...])
end

"""
  G = getindex!(A,genColType...)

For example, `getindex!(A,[:,3:6,2,[1,2,4,8]])`

Finds selected elements of a `denstens` similar to julia's form for `denstens`. Performs operation in-place but outputs tensor to `G`

See also: [`searchindex`](@ref) [`getindex`](@ref) [`denstens`](@ref) [`genColType`](@ref)
"""
function getindex!(C::tens{W}, a::genColType...) where W <: Number
  i = 0
  allintegers = true
  while allintegers && i < length(a)
    i += 1
    allintegers = typeof(a[i]) <: Integer
  end
  if allintegers
    dC = searchindex(C,a...)
  else
    cols = get_denseranges(size(C),a...)
    temp = Array(C)[cols...]
    dC = tens(temp)
#    dC = auxassemble!(C,a)
  end
  return dC
end
export getindex!
#=
function auxassemble!(C::tens{W},a::NTuple{G,genColType}) where {W <: Number, G}

  notintegers = 0
  startind = 0
  @inbounds for w = 1:G
    if !(typeof(a[w]) <: Integer)
      notintegers += 1
      if startind == 0
        startind = w
      end
    end
  end

  cols,newsize = get_denseranges_sizes(size(C),a...)


  x = prod(newsize)
  newT = Array{W,1}(undef,x)


  notfirst = startind > 1
  factor = 1
  if notfirst
    @inbounds @simd for h = startind-1:-1:1
      factor *= newsize[h]
    end
  end

  reqvec = makepos(G)

  i = 0
  @inbounds while i < length(newT)
    position_incrementer!(reqvec,newsize)
  
  
    backZ = reqvec[G]
    @inbounds @simd for w = G-1:-1:startind
      backZ -= 1
      backZ *= size(C,w)
      backZ += reqvec[w]
    end
  
    if notfirst
  
      foreZ = newsize[startind-1]
      @inbounds @simd for p = startind-2:-1:1
        foreZ -= 1
        foreZ *= newsize[w]
        foreZ += newvec[w]
      end
  
      @inbounds @simd for x = 0:newsize[startind]-1
        z = x + backZ
        z -= 1
        z *= factor
        z += foreZ
  
        i += 1
        newT[i] = C.T[z]
      end
  
    else
      @inbounds @simd for x = 0:newsize[startind]-1
        z = x + backZ
        i += 1
        newT[i] = C.T[z]
      end
    end
  
    reqvec[startind] += newsize[startind]
  end

  dC = tens{W}(newsize,newT)

  return dC
end
export auxassemble!
=#

function getindex!(C::diagonal{W}, b::genColType,a::genColType) where W <: Number
  if typeof(a) <: Integer
    return a == b ? C.T[a] : W(0)
  elseif a == b
    return getindex!(C,a)
  else
    return getindex!(Array(C),b,a)
  end
end

function getindex!(C::diagonal{W}, a::genColType) where W <: Number
#  return diagonal(C.T[a...])
  i = 0
  allintegers = true
  while allintegers && i < length(a)
    i += 1
    allintegers = typeof(a[i]) <: Integer
  end
  if allintegers
    return searchindex(C,a...)
  else
    cols = get_denseranges(size(C),a...)
    return diagonal{W}(C.T[cols...])
  end
end

"""
  G = searchindex(C,a...)

Find element of `C` that corresponds to positions `a` and outputs value `G`
"""
function searchindex(C::denstens,a::Integer...)
  if length(C.T) == 0
    outnum = eltype(C)(0)
  elseif length(a) == 1
    outnum = C.T[a...]
  else
    w = pos2ind(a,size(C))
    outnum = C.T[w]
  end
  return outnum
end

function searchindex(C::AbstractArray,a::Integer...)
  return searchindex(C,a)
end

function searchindex(C::LinearAlgebra.Diagonal,a::Integer,b::Integer)
  return C[a,b]
end

function searchindex(C::Union{Array{W,N},Diagonal{W}},a::NTuple{N,intType}) where {W <: Number, N}
  return C[a...]
end

"""
    searchindex(C,a...)

Find element of `C` that corresponds to positions `a`
"""
function searchindex(C::Qtens{W,Q},a::intType...) where {Q <: Qnum, W <: Number}
  q = findqsector(C,a)
  x = scaninds(1,q,C,a)
  y = scaninds(2,q,C,a)
  
  outnum = q == 0 ? W(0) : C.T[q][x,y]
  return outnum
end

export searchindex

"""
  G = lastindex(M,i)

Same as julia's `Array` `lastindex` but for `denstens` with output `G`

See also: [`lastindex`](@ref) [`denstens`](@ref)
"""
function lastindex(M::denstens, i::Integer)
  return M.size[i]
end

function lastindex(Qts::TNnetwork)
  return length(Qts.net)
end

#get the last index of a Qtensor (ex: A[:,1:end]...defines "end")
"""
    lastindex(Qtens,i)

get the last index of a Qtensor (ex: A[:,1:end]...defines "end")
"""
function lastindex(Qtens::qarray, i::Integer)
  return Qtens.size[i]
end

#=
"""
  G = lastindex(M)

Same as julia's `Array` `lastindex` but for `denstens` with output `G` (last element)

See also: [`lastindex`](@ref) [`denstens`](@ref)
"""
function lastindex(M::denstens)
  return M.T[length(M.T)]
end
=#

"""
  A = tensorfromdisc(name[,ext=".dmrjulia"])

Reads tensor `A` from disc with the Serialization package and filename `name`*`ext`

See also: [`tensor2disc`](@ref)
"""
function tensorfromdisc(name::String;ext::String=file_extension)
  return Serialization.deserialize(name*ext)
end

function getindex(A::bigvec,i::Integer)
  return tensorfromdisc(A.V[i])
end

function getindex(Qts::TNnetwork,i::Integer)
  return Qts.net[i]
end

function getindex(Qts::TNnetwork,i::UnitRange)
  return Qts.net[i]
end

"""
  getindex(A,i)

For a `nametens` with input `dtens` `A`, returns either the values (`i=0`) or derivatives (`i=1`)

See also: [`nametens`](@ref) [`dtens`](@ref)
"""
function getindex(Qts::TNobj,i::Integer)
  if typeof(Qts.N) <: dtens
    out = nametens(Qts.N[i],Qts.names,Qts.arrows,Qts.conj)
  else
    out = Qts.N[i]
  end
  return out
end
#=
function getindex(Qts::TNobj,i::Integer)
  return getindex(Qts.N,i)
end
=#

function getindex(A::dtens,w::intType)
  return A.d[w+1]
end

function getindex!(A::dtens,w::intType)
  return getindex(A,w)
end



"""
    A[:,3:6,2,[1,2,4,8]]

Finds selected elements of a Qtensor or dense tensor;

#Note:
+ Any modification to the output of this function can make the orignal tensor invalid.
  If the orignal tensor is needed and you need to operate on the resulting tensor of this function, 
  do a copy of one of the two before hand. This will decouple them.
+ (For Qtensors): Always returns a Qtensor.  If you want one element, use the searchindex function below

See also: [`searchindex`](@ref)
"""
function getindex(C::qarray, a::genColType...)
  return getindex!(C, a...)
end

function getindex!(A::Qtens{W,Q}, a::genColType...) where {Q <: Qnum, W <: Number}
  if isinteger(a...)
    return searchindex(A,a...)
  end

  condition = true
  for p = 1:length(a)
    condition = condition && (typeof(a[p]) <: Colon)
    condition = condition && (typeof(a[p]) <: UnitRange && length(a[p]) == size(A,p))
  end
  if condition
    return A
  end

  isjoinedindices = 0
  @inbounds for w = 1:length(A.size)
    isjoinedindices += length(A.size[w]) > 1
  end

  if isjoinedindices > 0
    C = mergereshape(A)
  else
    C = A
  end
  
  unitranges = get_ranges(size(C),a...)

  keepinds = Array{Bool,1}(undef,length(unitranges))
  @inbounds @simd for w = 1:length(keepinds)
    keepinds[w] = !(typeof(a[w]) <: Integer)
  end

  newdim = sum(keepinds)
  newQnumMat = Array{Array{intType,1},1}(undef,newdim)
  newQnumSum = Array{Array{Q,1},1}(undef,newdim)
  newsize = Array{intType,1}(undef,newdim)
  counter = 0
  @inbounds for i = 1:length(keepinds)
    if keepinds[i]
      counter += 1
      newQnumMat[counter] = C.QnumMat[i][a[i]]
      newQnumSum[counter] = C.QnumSum[i]
      newsize[counter] = length(unitranges[i])
    end
  end
  tup_newsize = Array{Array{intType,1},1}(undef,length(newQnumMat))
  @inbounds for i = 1:length(newQnumMat)
    tup_newsize[i] = [i]
  end

  if length(keepinds) != sum(keepinds)
    newflux = -C.flux
    @inbounds for k = 1:length(keepinds)
      if !keepinds[k]
        newflux += getQnum(k,a[k],C)
      end
    end
    newflux = -newflux
  else
    newflux = C.flux
  end
  

  Linds = C.currblock[1]
  keep_one = keepinds[Linds]
  Lkeepinds = Linds[keep_one]

  Rinds = C.currblock[2]
  keep_two = keepinds[Rinds]
  Rkeepinds = Rinds[keep_two]

  keepers,loadT,loadind_one,loadind_two = innerjoinloop(C,Linds,unitranges,Rinds,keep_one,keep_two,Lkeepinds,Rkeepinds)

  keptindices = vcat(A.size[keepinds]...)
  convertInds = Array{intType,1}(undef,length(A.QnumMat))
  count = 0
  @inbounds @simd for i = 1:length(keptindices)
    count += 1
    convertInds[keptindices[i]] = count
  end

  newcurrblock = (Lkeepinds,Rkeepinds)

  for w = 1:2
    @inbounds @simd for r = 1:length(newcurrblock[w])
      g = newcurrblock[w][r]
      newcurrblock[w][r] = convertInds[g]
    end
  end

  newT = loadT[keepers]

  nkeeps = sum(keepers)
  newinds = Array{NTuple{2,Array{intType,2}},1}(undef,nkeeps)

  newQsum = Array{NTuple{2,Q},1}(undef,nkeeps)
  counter = 0
  @inbounds for q = 1:length(loadind_one)
    if keepers[q]
      counter += 1
      newinds[counter] = (loadind_one[q],loadind_two[q])


      startQNs = Array{Q,1}(undef,2) # [Q(),Q()]
      for w = 1:2
        thisQN = Q()
        @inbounds for a = 1:length(newcurrblock[w])
          index = newcurrblock[w][a]
          dimval = newinds[counter][w][a,1] + 1
          thisQN += getQnum(index,dimval,newQnumMat,newQnumSum)
        end
        startQNs[w] = thisQN
      end
      newQsum[counter] = (startQNs...,)    
    end
  end

  return Qtens{W,Q}(tup_newsize, newT, newinds, newcurrblock, newQsum, newQnumMat, newQnumSum, newflux)
end
export getindex!

function isinteger(a::genColType...)
  isinteger = true
  w = 0
  @inbounds while isinteger && w < length(a)
    w += 1
    isinteger = typeof(a[w]) <: Integer
  end
  return isinteger
end
export isinteger

function get_ranges(sizes::NTuple{G,intType},a::genColType...) where G
  unitranges = Array{genColType,1}(undef,length(a))
  @inbounds for i = 1:length(a)
    if typeof(a[i]) <: Colon
      unitranges[i] = 0:sizes[i]-1#[w for w = 1:sizes[i]]
    elseif typeof(a[i]) <: Array{intType,1} || typeof(a[i]) <: Tuple{intType,Vararg{intType}}
      unitranges[i] = [a[i][w]-1 for w = 1:length(a[i])]
    elseif typeof(a[i]) <: UnitRange{intType}
      unitranges[i] = UnitRange(a[i][1]-1,a[i][end]-1)
    elseif typeof(a[i]) <: StepRange{intType}
      unitranges[i] = UnitRange(a[i][1]-1,a[i][end]-1)
    elseif typeof(a[i]) <: Integer
      unitranges[i] = a[i]-1
    end
  end
  return unitranges
end

function innerjoinloop(C::Qtens{W,Q},Linds::Array{P,1},unitranges::Array{G,1},Rinds::Array{P,1},
                        keep_one::Array{B,1},keep_two::Array{B,1},Lkeepinds::Array{P,1},Rkeepinds::Array{P,1}) where {W <: Number, Q <: Qnum, P <: Integer, B <: Bool, G <: genColType}

  keepers = [false for i = 1:length(C.T)]
  loadT = Array{Array{W,2},1}(undef,length(C.T))
  loadind_one = Array{Array{intType,2},1}(undef,length(C.T))
  loadind_two = Array{Array{intType,2},1}(undef,length(C.T))

  @inbounds for q = 1:length(C.T)

    keeprows = evaluate_keep(C,q,Linds,unitranges,1)

    if sum(keeprows) > 0
      keepcols = evaluate_keep(C,q,Rinds,unitranges,2)
      if sum(keepcols) > 0

        keepers[q] = true

        loadT[q] = C.T[q][keeprows,keepcols]

        loadind_one[q] = truncate_replace_inds(C,q,1,Lkeepinds,keep_one,unitranges,keeprows)
        loadind_two[q] = truncate_replace_inds(C,q,2,Rkeepinds,keep_two,unitranges,keepcols)
      end
    end

  end
  return keepers,loadT,loadind_one,loadind_two
end


function truncate_replace_inds(C::qarray,q::Integer,rowcol::Integer,Lkeepinds::Array{P,1},
                                keepbool::Array{Bool,1},kept_unitranges::Array{K,1},keeprows::Array{Bool,1}) where {P <: Integer, K <: genColType}

  thisindmat = C.ind[q][rowcol][keepbool,keeprows]
  offset = (rowcol-1)*length(C.currblock[1])

  @inbounds for a = 1:size(thisindmat,1)
    theseranges = kept_unitranges[Lkeepinds[a]]
    @inbounds for x = 1:size(thisindmat,2)
      thisval = thisindmat[a,x]

      newval = findfirst(w -> w == thisval,theseranges)[1]
      thisindmat[a,x] = newval-1
    end
  end
  return thisindmat
end


function evaluate_keep(C::qarray,q::Integer,Linds::Array{P,1},ap::Array{K,1},rowcol::Integer) where {P <: Integer, K <: genColType}
  thisindmat = C.ind[q][rowcol]
  keeprows = Array{Bool,1}(undef,size(thisindmat,2))
  rowindexes = size(thisindmat,1)
  @inbounds for x = 1:size(thisindmat,2)
    condition = true
    index = 0
    @inbounds while condition && index < rowindexes
      index += 1
      condition = thisindmat[index,x] in ap[Linds[index]]
    end
    keeprows[x] = condition
  end
  return keeprows
end


function getindex(A::dualnum{W},w::intType) where W <: Number
  if w == 0
    out = A.val
  elseif w == 1
    out = A.grad
  else
    out = W(0)
  end
  return out
end
