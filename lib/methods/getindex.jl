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

Selects subset or single element `G` of input tensor `M` (`denstens` or `Diagonal`) with elements of `a` being of `genColType` (ranges or values in tensor)

For example, `A[:,3:6,2,[1,2,4,8]]`

See also: [`searchindex`](@ref) [`getindex!`](@ref) [`denstens`](@ref) [`genColType`](@ref)
"""
function getindex(M::Union{denstens,diagonal}, a::genColType...)
  return getindex!(M,a...)
end

"""
  G = getindex(M,a...)

Selects subset or single element `G` of input tensor `M` (`denstens`) with elements of `a` being of arrays of booleans, keeping the true values

For example, `A[[true,false,true]]`

See also: [`searchindex`](@ref) [`getindex!`](@ref) [`denstens`](@ref) [`genColType`](@ref)
"""
function getindex(C::tens{W}, a::G...)where W <: Number where G <: Array{Bool,1}
  M = Array(C)
  return tens{W}(M[a...])
end










"""
    copyloop(x,p,forefactor,prefactor,backfactor,w,newT,A,copy_forefactor,copy_prefactor,copy_backfactor)

Helper function for function `tens_getindex` which obtains indices from tensors for input `x` (a vector or tuple of integers) which corresponds to the current position of the largest index in the tensor, `p` an integer corresopnding to the length of the input `x`, `forefactor` is the factor of all the sizes and positions multiplied together before the selected index, `backfactor` is the multiplication of all of the sizes of the indices and positions after the selected index, `prefactor` is the sizes before the indexes before the selected index, `w` is a counter to record where in the copied tensor we are, `newT` is the Array to copy elements to, `A` is the input `denstens`, `copy_forefactor` same as `forefactor` but for the copied tensor, `copy_prefactor` is the same as `prefactor` but for the copied tensor, and `copy_backfactor` is the same as the `backfactor` but for the copied tensor
"""
function copyloop(x::Union{Vector,Tuple{intType,Vararg{intType}}},p::Integer,forefactor::Integer,prefactor::Integer,backfactor::Integer,w::Integer,newT::Array{W,1},A::tens{W},copy_forefactor::Integer,copy_prefactor::Integer,copy_backfactor::Integer) where W <: Number
  @inbounds @simd for r = 1:p
    w = copy_forefactor + copy_prefactor * (r-1 + copy_backfactor) #1
    y = forefactor + prefactor * (x[r]-1 + backfactor) #pos2ind(pos,sizeA)
    newT[w] = A.T[y]
  end
  return w
end

"""
    copyloop(x,p,forefactor,prefactor,backfactor,w,newT,A,copy_forefactor,copy_prefactor,copy_backfactor)

Helper function for function `tens_getindex` which obtains indices from tensors for input `x` (a UnitRange, integer, Colon, or StepRange) which corresponds to the current position of the largest index in the tensor, `p` an integer corresopnding to the length of the input `x`, `forefactor` is the factor of all the sizes and positions multiplied together before the selected index, `backfactor` is the multiplication of all of the sizes of the indices and positions after the selected index, `prefactor` is the sizes before the indexes before the selected index, `w` is a counter to record where in the copied tensor we are, `newT` is the Array to copy elements to, `A` is the input `denstens`, `copy_forefactor` same as `forefactor` but for the copied tensor, `copy_prefactor` is the same as `prefactor` but for the copied tensor, and `copy_backfactor` is the same as the `backfactor` but for the copied tensor
"""
function copyloop(x::Union{UnitRange{intType},intType,Colon,StepRange{intType}},p::Integer,forefactor::Integer,prefactor::Integer,backfactor::Integer,w::Integer,newT::Array{W,1},A::tens{W},copy_forefactor::Integer,copy_prefactor::Integer,copy_backfactor::Integer) where W <: Number
  @inbounds @simd for r = 0:p-1
    w = copy_forefactor + copy_prefactor * (r + copy_backfactor) #1
    y = forefactor + prefactor * (r + backfactor)
    newT[w] = A.T[y]
  end
  return w
end

"""
    simple_copyloop(x,p,forefactor,prefactor,backfactor,w,newT,A,copy_forefactor,copy_prefactor,copy_backfactor)

Helper function for function `tens_getindex` which obtains indices from tensors for input `x` (a vector or tuple of integers) which corresponds to the current position of the largest index in the tensor, `p` an integer corresopnding to the length of the input `x`, `forefactor` is the factor of all the sizes and positions multiplied together before the selected index, `backfactor` is the multiplication of all of the sizes of the indices and positions after the selected index, `prefactor` is the sizes before the indexes before the selected index, `w` is a counter to record where in the copied tensor we are, `newT` is the Array to copy elements to, `A` is the input `denstens`
"""
function simple_copyloop(x::Union{Vector,Tuple{intType,Vararg{intType}}},p::Integer,backfactor::Integer,w::Integer,newT::Array{W,1},A::tens{W}) where W <: Number
  @inbounds @simd for r = 1:p
    w += 1
    y = x[r] + backfactor #pos2ind(pos,sizeA)
    newT[w] = A.T[y]
  end
  return w
end

"""
    simple_copyloop(x,p,forefactor,prefactor,backfactor,w,newT,A,copy_forefactor,copy_prefactor,copy_backfactor)

Helper function for function `tens_getindex` which obtains indices from tensors for input `x` (a UnitRange, integer, Colon, or StepRange) which corresponds to the current position of the largest index in the tensor, `p` an integer corresopnding to the length of the input `x`, `forefactor` is the factor of all the sizes and positions multiplied together before the selected index, `backfactor` is the multiplication of all of the sizes of the indices and positions after the selected index, `prefactor` is the sizes before the indexes before the selected index, `w` is a counter to record where in the copied tensor we are, `newT` is the Array to copy elements to, `A` is the input `denstens`
"""
function simple_copyloop(x::Union{UnitRange{intType},intType,Colon,StepRange{intType}},p::Integer,backfactor::Integer,w::Integer,newT::Array{W,1},A::tens{W}) where W <: Number
  @inbounds @simd for r = 1:p
    w += 1
    y = r + backfactor #pos2ind(pos,sizeA)
    newT[w] = A.T[y]
  end
  return w
end

"""
    tens_getindex(A,a...)

A modified version of the `getindex` function that throws less allocations for `denstens` at the cost of more verbose code. Can be faster in some cases (speed checks as of v1.10.4)
"""
function tens_getindex(A::tens{W}, a::genColType...) where W <: Number

  G = length(a)

  minmaxvals = Array{NTuple{2,intType},1}(undef,G) #+1

  newsizes = Array{intType,1}(undef,G) #+1
  for w = 1:G
    if typeof(a[w]) <: Colon
      minmaxvals[w] = (1,size(A,w))
      newsizes[w] = size(A,w)
    elseif typeof(a[w]) <: Vector
      minmaxvals[w] = (1,length(a[w]))
      newsizes[w] = length(a[w])
    else
      minmaxvals[w] = (minimum(a[w]),maximum(a[w]))
      newsizes[w] = length(a[w])
    end
  end



  copy_lengths = newsizes
  maxlength = maximum(copy_lengths)

  d1 = 1  
  while d1 <= length(a) && copy_lengths[d1] != maxlength
    d1 += 1
  end


  totsize = 1
  for w = 1:G
    totsize *= newsizes[w]
  end
  newT = Array{W,1}(undef,totsize)

  count_vectors = 0
  for w = 1:length(a)
    count_vectors += typeof(a[w]) <: Vector || typeof(a[w]) <: Tuple
  end

  if count_vectors > 0
    whichVectors = Array{intType,1}(undef,count_vectors)
    counter = 0
    w = 0
    while w < length(a) && counter < count_vectors
      w += 1
      if typeof(a[w]) <: Vector || typeof(a[w]) <: Tuple
        counter += 1
        whichVectors[counter] = w
      end
    end
  end
  
  sizeA = tupsize(A)

  pos = Array{intType,1}(undef,G)
  for w = 2:length(pos)
    pos[w] = minmaxvals[w][1]
  end
  pos[1] = 0

  if count_vectors > 0
    legacypos = Array{intType,1}(undef,G)
  else
    legacypos = pos
  end

  if d1 > 1
    prefactor = 1
    @inbounds @simd for h = d1-1:-1:1
      prefactor *= sizeA[h]
    end

    copy_prefactor = 1
    @inbounds @simd for h = d1-1:-1:1
      copy_prefactor *= newsizes[h]
    end
  end

  w = 0
  while w < length(newT)
    position_incrementer!(pos,minmaxvals)
    pos[d1] = 1

    if count_vectors > 0
      @inbounds @simd for w = 1:length(legacypos)
        legacypos[w] = pos[w]
      end

      @inbounds @simd for p = 1:count_vectors
        x = whichVectors[p]
        legacypos[x] = a[x][pos[x]]
      end
    end

    backfactor = 0
    @inbounds @simd for g = length(legacypos):-1:d1+1
      backfactor *= sizeA[g]
      backfactor += legacypos[g]
      backfactor -= 1
    end
    backfactor *= sizeA[d1]

    if d1 > 1



      forefactor = legacypos[d1-1]
      @inbounds @simd for w = d1-2:-1:1
        forefactor -= 1
        forefactor *= sizeA[w]
        forefactor += legacypos[w]
      end



      copy_backfactor = 0
      @inbounds @simd for g = length(pos):-1:d1+1
        copy_backfactor *= newsizes[g]
        copy_backfactor += pos[g]
        copy_backfactor -= 1
      end
      copy_backfactor *= newsizes[d1]

      copy_forefactor = pos[d1-1]
      @inbounds @simd for w = d1-2:-1:1
        copy_forefactor -= 1
        copy_forefactor *= newsizes[w]
        copy_forefactor += pos[w]
      end

      w = copyloop(a[d1],newsizes[d1],forefactor,prefactor,backfactor,w,newT,A,copy_forefactor,copy_prefactor,copy_backfactor)
    else
        w = simple_copyloop(a[d1],newsizes[d1],backfactor,w,newT,A)
    end
    pos[d1] = newsizes[d1]

  end
  return tens(newsizes,newT)
end





"""
  G = getindex!(A,a...)

For example, `getindex!(A,[:,3:6,2,[1,2,4,8]])`

Finds selected elements of a `denstens` according to rule `a` of a `genColType`. Performs operation in-place but outputs tensor to `G`

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

    dC = tens_getindex(C,a...)
#    cols = get_denseranges(size(C),a...)
#    temp = Array(C)[cols...]
#    dC = tens{W}(temp)
    isintegers = Array{Bool,1}(undef,ndims(dC))
    for w = 1:length(isintegers)
      isintegers[w] = !(typeof(a[w]) <: Integer)
    end
    if sum(isintegers) > 0
      dC.size = dC.size[isintegers]
    end
  end
  return dC
end

"""
  G = getindex!(A,a,b)

For example, `getindex!(A,[:,3:6,2,[1,2,4,8]])`

Finds selected elements of a `Diagonal` according to integer positions `a` and `b` . Performs operation in-place but outputs tensor to `G`

See also: [`searchindex`](@ref) [`getindex`](@ref) [`denstens`](@ref) [`genColType`](@ref)
"""
function getindex!(C::Diagonal{W}, b::genColType,a::genColType) where W <: Number
  if typeof(a) <: Integer
    return a == b ? C.T[a] : W(0)
  elseif a == b
    return getindex!(C,a)
  else
    return getindex!(Array(C),b,a)
  end
end

"""
  G = getindex!(A,a)

For example, `getindex!(A,[:,3:6,2,[1,2,4,8]])`

Finds selected elements of a `Diagonal` according to integer positions `a` (searches along diagonal) . Performs operation in-place but outputs tensor to `G`

See also: [`searchindex`](@ref) [`getindex`](@ref) [`denstens`](@ref) [`genColType`](@ref)
"""
function getindex!(C::Diagonal{W}, a::genColType) where W <: Number
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
    return Diagonal{W}(C.T[cols...])
  end
end

"""
  G = searchindex(C,a...)

Find element of `C` (`denstens` or `Diagonal`) that corresponds to position `a` and outputs value `G`
"""
function searchindex(C::Union{denstens,diagonal},a::Integer...)
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

"""
  G = searchindex(C,a...)

Find element of `C` (AbstractArray) that corresponds to positions `a` and outputs value `G`
"""
function searchindex(C::AbstractArray,a::Integer...)
  return searchindex(C,a)
end

"""
  G = searchindex(C,a,b)

Find element of `C` (LinearAlgebra.Diagonal) that corresponds to positions `a` and `b` and outputs value `G`
"""
function searchindex(C::LinearAlgebra.Diagonal,a::Integer,b::Integer)
  return C[a,b]
end

"""
  G = searchindex(C,a,b)

Find element of `C` (Array) that corresponds to positions `a` and `b` and outputs value `G`
"""
function searchindex(C::Array{W,N}#=Union{Array{W,N},Diagonal{W}=#,a::NTuple{N,intType}) where {W <: Number, N}
  return C[a...]
end

"""
    searchindex(C,a...)

Find element of `C` (`qarray`) that corresponds to positions `a`
"""
function searchindex(C::Qtens{W,Q},a::intType...) where {Q <: Qnum, W <: Number}
  q = findqsector(C,a)
  x = scaninds(1,q,C,a)
  y = scaninds(2,q,C,a)
  
  outnum = q == 0 ? W(0) : C.T[q][x,y]
  return outnum
end


"""
  G = lastindex(M,i)

Same as julia's `Array` `lastindex` but for `denstens` with output `G`

See also: [`lastindex`](@ref) [`denstens`](@ref)
"""
function lastindex(M::denstens, i::Integer)
  return M.size[i]
end

"""
  G = lastindex(M,i)

Same as julia's `Array` `lastindex` but for `TNnetwork` with output `G` = `M`[`i`]

See also: [`lastindex`](@ref) [`TNnetwork`](@ref)
"""
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

"""
    B = getindex(A,i)

grabs element `i` of `bigvec` `A` as output `B`
"""
function getindex(A::bigvec,i::Integer)
  return tensorfromdisc(A.V[i])
end

"""
    B = getindex(A,i)

grabs element `i` of `TNnetwork` `A` as output `B`
"""
function getindex(Qts::TNnetwork,i::Integer)
  return Qts.net[i]
end

"""
    B = getindex(A,r)

grabs elements `r` given as a UnitRange of `TNnetwork` `A` as output `B`
"""
function getindex(Qts::TNnetwork,i::UnitRange)
  return Qts.net[i]
end

"""
    B = getindex(A,i)

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

"""
    B = getindex(A,i)

grabs element `i` of `dtens` `A` as output `B`

Note: `dtens` is 0-indexed so inputting i = 0 will give `B` = `A`[1]
"""
function getindex(A::dtens,w::intType)
  return A.d[w+1]
end

"""
    B = getindex!(A,i)

grabs element `i` of `dtens` `A` as output `B`

Note: `dtens` is 0-indexed so inputting i = 0 will give `B` = `A`[1]
"""
function getindex!(A::dtens,w::intType)
  return getindex(A,w)
end



"""
    getindex(C,a...)


Example: A[:,3:6,2,[1,2,4,8]]

Finds selected elements of a `qarray`

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

"""
    getindex!(C,a...)


Example: A[:,3:6,2,[1,2,4,8]]

Finds selected elements of a `qarray`

#Note:
+ Any modification to the output of this function can make the orignal tensor invalid.
  If the orignal tensor is needed and you need to operate on the resulting tensor of this function, 
  do a copy of one of the two before hand. This will decouple them.
+ (For Qtensors): Always returns a Qtensor.  If you want one element, use the searchindex function below

See also: [`searchindex`](@ref)
"""
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

"""
    isinteger(a...)

Function boundary for checking if input tuple of `genColType` is actually all integers
"""
function isinteger(a::genColType...)
  isinteger = true
  w = 0
  @inbounds while isinteger && w < length(a)
    w += 1
    isinteger = typeof(a[w]) <: Integer
  end
  return isinteger
end

"""
    B = get_ranges(sizes,a...)

Converts various input `genColType`s from `a` to get-able objects on output `B`

Implementation notes: Could be optimized out but this is quite low impact
"""
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

"""
    innerjoinloop(C,Linds,unitranges,Rinds,keep_one,keep_two,Lkeepinds,Rkeepinds)

Function boundary for the `joinindex!` function. Takes an input `qarray` `C` and `Linds` the values to keep on an index, `unitranges` are the value on the index to keep, `Rinds` are the other side of the matrix-equivalent of `C` to keep, `keep_one` is the elements to keep on side 1, `keep_two` is the elements to keep on side 2, `Lkeepinds` is the actualy indices to keep on the left, `Rkeepinds` is the actualy indices to keep on the right

Some versions of Julia demand that this function be input in order to ensure low allocation throws
"""
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

"""
    truncate_replace_inds(C,q,rowcol,Lkeepinds,keepbool,kept_unitranges,keeprows)

On evaluation of a `getindex` function, we obtain the rows and columns that are kept from an input `qarray` `C`, the requested block `q`, `rowcol` is an integer of the .ind field that we want to examine, `Lkeepinds` holds the unit ranges for the kept values in the .ind field, `keepbool` is an integer value corresponding to the .ind field, `keeprows` are the rows of the .ind field to be kept
"""
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

"""
    evaluate_keep(C,q,Linds,ap,rowcol)

Evaluates which indices to keep on a `qarray` when getting indices for an input `qarray` `C` with quantum number block `q`, `Linds` corresponding to .ind field to search, `ap` containing a conversion rule for `Linds` to see kept indices, and `rowcol` containing the kept .ind values
"""
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

"""
    B = getindex(A,w)

Obtain elements from a `dualnum` with `w`=0 being the value and `w`=1 being the derivative of `A`
"""
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
