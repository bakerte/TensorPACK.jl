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
    set_copyloop(x,p,forefactor,prefactor,backfactor,w,newT,A,copy_forefactor,copy_prefactor,copy_backfactor)

Helper function for function `tens_getindex` which obtains indices from tensors for input `x` (a vector or tuple of integers) which corresponds to the current position of the largest index in the tensor, `p` an integer corresopnding to the length of the input `x`, `forefactor` is the factor of all the sizes and positions multiplied together before the selected index, `backfactor` is the multiplication of all of the sizes of the indices and positions after the selected index, `prefactor` is the sizes before the indexes before the selected index, `w` is a counter to record where in the copied tensor we are, `newT` is the Array to copy elements to, `A` is the input `denstens`, `copy_forefactor` same as `forefactor` but for the copied tensor, `copy_prefactor` is the same as `prefactor` but for the copied tensor, and `copy_backfactor` is the same as the `backfactor` but for the copied tensor
"""
function set_copyloop!(x::Union{UnitRange{intType},StepRange{intType},Vector,Tuple{intType,Vararg{intType}}},p::Integer,w::Integer,newT::Union{Memory{W},tens{W},Array{W,P}},A::Union{Array{W,P},tens{W}},copy_forefactor::Integer,copy_prefactor::Integer,copy_backfactor::Integer) where {W <: Number, P}
  @inbounds @simd for r = 1:p
    y = copy_forefactor + copy_prefactor * (x[r]-1 + copy_backfactor) #pos2ind(pos,sizeA)
    newT[y] = A[w + r]
  end
  return p
end

"""
    set_copyloop(x,p,forefactor,prefactor,backfactor,w,newT,A,copy_forefactor,copy_prefactor,copy_backfactor)

Helper function for function `tens_getindex` which obtains indices from tensors for input `x` (a UnitRange, integer, Colon, or StepRange) which corresponds to the current position of the largest index in the tensor, `p` an integer corresopnding to the length of the input `x`, `forefactor` is the factor of all the sizes and positions multiplied together before the selected index, `backfactor` is the multiplication of all of the sizes of the indices and positions after the selected index, `prefactor` is the sizes before the indexes before the selected index, `w` is a counter to record where in the copied tensor we are, `newT` is the Array to copy elements to, `A` is the input `denstens`, `copy_forefactor` same as `forefactor` but for the copied tensor, `copy_prefactor` is the same as `prefactor` but for the copied tensor, and `copy_backfactor` is the same as the `backfactor` but for the copied tensor
"""
function set_copyloop!(x::Union{intType,Colon},p::Integer,w::Integer,newT::Union{Memory{W},tens{W},Array{W,P}},A::Union{Array{W,P},tens{W}},copy_forefactor::Integer,copy_prefactor::Integer,copy_backfactor::Integer) where {W <: Number, P}
  @inbounds @simd for r = 0:p-1
#    println(r," ",copy_forefactor," ",copy_prefactor," ",copy_backfactor)
    y = copy_forefactor + copy_prefactor * (r + copy_backfactor)
#    println(y," ",w," ",r)
    newT[y] = A[w + r + 1]
  end
  return p
end

"""
    set_simple_copyloop(x,p,forefactor,prefactor,backfactor,w,newT,A,copy_forefactor,copy_prefactor,copy_backfactor)

Helper function for function `tens_getindex` which obtains indices from tensors for input `x` (a vector or tuple of integers) which corresponds to the current position of the largest index in the tensor, `p` an integer corresopnding to the length of the input `x`, `forefactor` is the factor of all the sizes and positions multiplied together before the selected index, `backfactor` is the multiplication of all of the sizes of the indices and positions after the selected index, `prefactor` is the sizes before the indexes before the selected index, `w` is a counter to record where in the copied tensor we are, `newT` is the Array to copy elements to, `A` is the input `denstens`
"""
function set_simple_copyloop!(x::Union{UnitRange{intType},StepRange{intType},Vector,Tuple{intType,Vararg{intType}}},p::Integer,backfactor::Integer,w::Integer,newT::Union{Memory{W},tens{W},Array{W,S}},A::Union{Array{W,R},Memory{W},tens{W}}) where {R, S, W <: Number}
  @inbounds @simd for r = 1:p
    w += 1
    y = x[r] + backfactor #pos2ind(pos,sizeA)
    newT[y] = A[w]
  end
  return p
end

"""
    set_simple_copyloop(x,p,forefactor,prefactor,backfactor,w,newT,A,copy_forefactor,copy_prefactor,copy_backfactor)

Helper function for function `tens_getindex` which obtains indices from tensors for input `x` (a UnitRange, integer, Colon, or StepRange) which corresponds to the current position of the largest index in the tensor, `p` an integer corresopnding to the length of the input `x`, `forefactor` is the factor of all the sizes and positions multiplied together before the selected index, `backfactor` is the multiplication of all of the sizes of the indices and positions after the selected index, `prefactor` is the sizes before the indexes before the selected index, `w` is a counter to record where in the copied tensor we are, `newT` is the Array to copy elements to, `A` is the input `denstens`
"""
function set_simple_copyloop!(x::Union{intType,Colon},p::Integer,backfactor::Integer,w::Integer,newT::Union{Memory{W},tens{W},Array{W,S}},A::Union{Array{W,R},Memory{W},tens{W}}) where {R, S, W <: Number}
  @inbounds @simd for r = 1:p
    w += 1
    y = r + backfactor #pos2ind(pos,sizeA)
    newT[y] = A[w]
  end
  return p
end



"""
    setindex!(B,A,a...)

Takes elements from `A` (`denstens`) and puts them into `B` (`denstens`) along elements given by `a` (`genColType`)
"""
function setindex!(B::Union{Array{W,R},tens{W}},A::tens{W},a::genColType...) where {R, W <: Number}

  G = length(a)

  minmaxvals = Array{NTuple{2,intType},1}(undef,G)

  for w = 1:G
    if typeof(a[w]) <: Colon
      minmaxvals[w] = (1,size(A,w))
    elseif typeof(a[w]) <: Vector
      minmaxvals[w] = (1,length(a[w]))
    else
      minmaxvals[w] = (minimum(a[w]),maximum(a[w]))
    end
  end

  copy_lengths = [minmaxvals[w][2]-minmaxvals[w][1]+1 for w = 1:length(minmaxvals)]
  maxlength = maximum(copy_lengths)

  d1 = 1  
  while d1 <= length(a) && copy_lengths[d1] != maxlength
    d1 += 1
  end


  newT = B #typeof(B) <: denstens ? B.T : B

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
#    prefactor = 1
#    @inbounds @simd for h = d1-1:-1:1
#      prefactor *= sizeA[h]
#    end

    copy_prefactor = 1
    @inbounds @simd for h = d1-1:-1:1
      copy_prefactor *= B.size[h]
    end
  end

  savepos = Array{intType,1}(undef,length(minmaxvals))

  w = 0
  while w < length(A)
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

    copy_backfactor = 0
    @inbounds @simd for g = length(legacypos):-1:d1+1
      copy_backfactor *= B.size[g]
      copy_backfactor += legacypos[g]
      copy_backfactor -= 1
    end
    copy_backfactor *= B.size[d1]

    if d1 > 1

      copy_forefactor = legacypos[d1-1]
      @inbounds @simd for w = d1-2:-1:1
        copy_forefactor -= 1
        copy_forefactor *= B.size[w]
        copy_forefactor += legacypos[w]
      end

      w += set_copyloop!(a[d1],copy_lengths[d1],w,newT,A,copy_forefactor,copy_prefactor,copy_backfactor)
    else
        w += set_simple_copyloop!(a[d1],copy_lengths[d1],copy_backfactor,w,newT,A)
    end

    pos[d1] = minmaxvals[d1][2]

  end
  nothing
end

function setindex!(B::tens{W},A::Array{W,R},a::genColType...) where {R, W <: Number}
  return setindex!(B,tens(A),a...)
end

#=
"""
    setindex!(B,A,a...)

Takes elements from `A` (`denstens`) and puts them into `B` (Array) along elements given by `a` (`genColType`)
"""
function setindex!(B::Array{W,S},A::tens{W},a::genColType...) where {R, S, W <: Number}
end
=#


#=
"""
    setindex!(B,A,a...)

Takes elements from `A` (Array) and puts them into `B` (`denstens`) along elements given by `a` (`genColType`)
"""
function setindex!(B::tens{W},A::Array{W,N},a::genColType...) where {W <: Number, N}
  G = Array(B)
  G[a...] = A
  @inbounds @simd for w = 1:length(G)
    B.T[w] = G[w]
  end
  nothing
end
=#
"""
    setindex!(B,A,a...)

Takes elements from `A` (`denstens`) and puts them into `B` (`Diagonal`) along elements given by `a` (`genColType`)
"""
function setindex!(B::tens{W},A::Diagonal{W},a::genColType...) where W <: Number
  G = Array(B)
  G[a...] = Array(A)
  @inbounds @simd for w = 1:length(G)
    B.T[w] = G[w]
  end
  nothing
end

"""
    setindex!(B,A,a...)

Takes elements from `A` (Array) and puts them into `B` (`Diagonal`) along elements given by `a` (`genColType`)
"""
function setindex!(B::Array{W,N},A::Diagonal{W},a::genColType...) where {W <: Number, N}
  G = Array(A)
  for w = 1:length(A)
    B[w] = A[w]
  end
  nothing
end










#=
"""
    setindex!(B,A,a...)

Takes elements from `A` (Array) and puts them into `B` (`denstens`) along elements given by `a` (`genColType`)
"""
function setindex!(B::Array{W,N},A::tens{W},a::genColType...) where {W <: Number, N}
  G = Array(A)
  B[a...] = G
  nothing
end
=#
#=
function setindex!(B::tens{W},A::W,a::Integer...) where W <: Number
  @inbounds index = a[end]-1
  @inbounds @simd for q = length(a)-1:-1:1
    index *= size(B,q)
    index += a[q]-1
  end
  @inbounds B.T[index+1] = A
  nothing
end
=#

"""
    setindex!(B,A,b,a...)

Takes elements from `A` (number) and puts it into `B` (`denstens`) along elements given by `b` and `a` (all integers only)
"""
function setindex!(B::tens{W},A::W,a::Integer,b::Integer...) where W <: Number
  index = 0
  @inbounds @simd for q = length(b):-1:1
    index += b[q]-1
    index *= size(B,q)
  end
  index += a-1
  @inbounds B.T[index+1] = A
  nothing
end

"""
    setindex!(B,A,b)

Takes elements from `A` (number) and puts it into `B` (`denstens`) along elements given by `b` (integer)
"""
function setindex!(B::tens{W},A::W,a::Integer) where W <: Number
  @inbounds B.T[a] = A
  nothing
end

"""
    setindex!(B,A,b)

Takes elements from `A` (number) and puts it into `B` (`Diagonal`) along elements given by `b` (integer)
"""
function setindex!(B::Diagonal{W},A::W,a::Integer) where W <: Number
  @inbounds B.T[a] = A
  nothing
end

"""
    setindex!(B,A,b,a)

Takes elements from `A` (number) and puts it into `B` (`Diagonal`) along elements given by `b` and `a` (integer)
"""
function setindex!(B::Diagonal{W},A::W,a::Integer,b::Integer) where W <: Number
  if a != b
    error("not defined for loading diagonal on off-diagonal elements...convert to denstens with eye(a,b) [two integers]")
  end
  @inbounds B.T[a] = A
  nothing
end

"""
    setindex!(Qts,newTens,i)

Places `newTens` (any type of tensor allowed) into `Qts` which is a `TNnetwork` at position `i`
"""
function setindex!(Qts::TNnetwork,newTens::allTens,i::Integer)
  return Qts.net[i] = newTens
end

"""
    setindex!(B,A,a...)

Places `dtens` `A` into another `dtens` `B` at positions given by `a` (`genColType`)
"""
function setindex!(B::dtens,A::dtens,a::genColType...)
  for w = 1:length(A.d)
    setindex!(B[w-1],A[w-1],a...)
  end
  nothing
end

"""
    setindex!(B,A,a)

Loads tensor `A` (any appropriate type, same as those in `B`) into `B` (`dtens`) at position `a` (either 0 or 1 currently, else does nothing)
"""
function setindex!(B::dtens,A::allTens,a::Integer)
#  A.d[a+1] = B
if a == 1
  A.d = (B,A[1])
elseif a == 0
  A.d = (A[0],B)
end
#  setindex!(A.d,B,a+1)
#  setindex!(A[a+1],B,)
  nothing
end

"""
    setindex!(B,A,a...)

Takes elements from `A` (`qarray`) and puts them into `B` (`qarray`) along elements given by `a` (`genColType`)
"""
function setindex!(A::Qtens{W,Q},B::Qtens{W,Q},vals::genColType...) where {W <: Number, Q <: Qnum}
  C = changeblock(B,A.currblock)
  Asize = size(A)
  unitranges = get_denseranges(Asize,vals...)

  commoninds = matchblocks((false,false),A,C,ind=(2,1),matchQN=A.flux)

  Lpos = Array{intType,1}(undef,length(C.currblock[1]))
  Rpos = Array{intType,1}(undef,length(C.currblock[2]))

  valvec = [0]

  @inbounds for q = 1:length(commoninds)
    Aqind = commoninds[q][1]
    Cqind = commoninds[q][2]
    for y = 1:size(C.ind[Cqind][2],2)
      loadpos!(Rpos,C,Cqind,2,y,unitranges)
      found2,colindex = findmatch(Rpos,A,C,Aqind,2)
      if found2
        @inbounds for x = 1:size(C.ind[Cqind][1],2)
          loadpos!(Lpos,C,Cqind,1,x,unitranges)
          found,rowindex = findmatch(Lpos,A,C,Aqind,1)
          if found
            num = C.T[Cqind][x,y]
            A.T[Aqind][rowindex,colindex] = num
          end
        end
      end

    end
  end
  nothing
end

"""
    setindex!(B,A,a...)

Takes elements from `A` (`denstens`) and puts them into `B` (Array) along elements given by `a` (`genColType`)
"""
function setindex!(A::Array{W,N},B::Qtens{W,Q},vals::genColType...) where {W <: Number, Q <: Qnum, N}
  Asize = size(A)
  unitranges = get_denseranges(Asize,vals...)

  Lpos = Array{intType,1}(undef,length(B.currblock[1]))
  Rpos = Array{intType,1}(undef,length(B.currblock[2]))
  totalpos = Array{intType,1}(undef,ndims(A))

  @inbounds for q = 1:length(B.T)
    @inbounds for y = 1:size(B.ind[q][2],2)
      loadpos!(Rpos,B,q,2,y,unitranges)
      @inbounds @simd for w = 1:length(Rpos)
        totalpos[B.currblock[2][w]] = Rpos[w]
      end
      @inbounds for x = 1:size(B.ind[q][1],2)
        loadpos!(Lpos,B,q,1,x,unitranges)
        num = B.T[q][x,y]
        @inbounds @simd for w = 1:length(Lpos)
          totalpos[B.currblock[1][w]] = Lpos[w]
        end
        A[totalpos...] = num
      end
    end
  end
  nothing
end

"""
    setindex!(B,A,a...)

Takes elements from `A` (`denstens`) and puts them into `B` (`denstens`) along elements given by `a` (`genColType`)
"""
function setindex!(A::tens{W},B::Qtens{W,Q},vals::genColType...) where {W <: Number, Q <: Qnum}
  Asize = size(A)
  unitranges = get_denseranges(Asize,vals...)

  Lpos = Array{intType,1}(undef,length(B.currblock[1]))
  Rpos = Array{intType,1}(undef,length(B.currblock[2]))
  totalpos = Array{intType,1}(undef,ndims(A))

  @inbounds for q = 1:length(B.T)
    @inbounds for y = 1:size(B.ind[q][2],2)
      loadpos!(Rpos,B,q,2,y,unitranges)
      @inbounds @simd for w = 1:length(Rpos)
        totalpos[B.currblock[2][w]] = Rpos[w]
      end
      @inbounds for x = 1:size(B.ind[q][1],2)
        loadpos!(Lpos,B,q,1,x,unitranges)
        num = B.T[q][x,y]
        @inbounds @simd for w = 1:length(Lpos)
          totalpos[B.currblock[1][w]] = Lpos[w]
        end
        z = pos2ind(totalpos,Asize)
        A.T[z] = num
      end
    end
  end
  nothing
end

"""
    setindex!(B,A,a...)

Takes elements from `A` (a number) and puts them into `B` (`qarray`) along elements given by `a` (integers)
"""
function setindex!(C::Qtens{W,Q},val::W,a::intType...) where {W <: Number, Q <: Qnum}
  if length(C.T) > 0
    q = findqsector(C,a)

    x = scaninds(1,q,C,a)
    y = scaninds(2,q,C,a)

    @inbounds C.T[q][x,y] = val
  end
  nothing
end

"""
    findmatch(Lpos,A,C,Aqind,LR)

finds a position in a `qarray` `A` and matches quantum number in `C` given an index `Aqind` and left or right number (1 or 2) `LR`; `Lpos` is a vector of integers corresponding to the element in question
"""
function findmatch(Lpos::Array{P,1},A::Qtens{W,Q},C::Qtens{W,Q},Aqind::Integer,LR::Integer) where {P <: Integer, W <: Number, Q <: Qnum}
  found = false
  rowindex = 0
  while !found && rowindex < size(A.ind[Aqind][LR],2)
    rowindex += 1
    matchinginds = true
    g = 0
    @inbounds while matchinginds && g < length(C.currblock[LR])
      g += 1
      matchinginds = A.ind[Aqind][LR][g,rowindex] == Lpos[g]
    end
    found = matchinginds
  end
  return found,rowindex
end

"""
    loadpos!(Lpos,C,Cqind,LR,x,Cqind,unitranges)

Function boundary to create .ind fields for an input `qarray` `C` on index `Cqind`, whether field 1 or 2 for .ind `LR`, an integer `x` for the position in the .ind field, and unit ranges `unitranges` (an array of `genColType`) that are loaded into `Lpos`
"""
function loadpos!(Lpos::Array{P,1},C::Qtens{W,Q},Cqind::Integer,LR::Integer,x::Integer,unitranges::Array{B,1}) where {B <: genColType, P <: Integer, W <: Number, Q <: Qnum}
  @inbounds @simd for w = 1:length(Lpos)
    index = C.currblock[LR][w]
    xpos = C.ind[Cqind][LR][w,x] + 1
    Lpos[w] = unitranges[index][xpos]
  end
  nothing
end

"""
    findqsector(C,a)

Finds quantum number of the sector for `qarray` `C` at position `a`, a tuple full of integers
"""
function findqsector(C::qarray,a::NTuple{N,intType}) where N

  LR = length(C.currblock[1]) < length(C.currblock[2]) ? 1 : 2

  smallinds = C.currblock[LR]
  if length(smallinds) == 0
    targetQN = C.flux
  else
    targetQN = C.flux
    @inbounds @simd for i = 1:length(smallinds)
      y = smallinds[i]
      targetQN += getQnum(y,a[y],C)
    end
  end

  notmatchingQNs = true
  q = 0
  @inbounds while q < length(C.T) && notmatchingQNs
    q += 1
    currQN = C.Qblocksum[q][LR]
    notmatchingQNs = targetQN != currQN
  end
  return q
end

"""
    x = scaninds(blockindex,q,C,a)

takes a `blockindex` (integer corresponding to .ind field of `qarray`) for a `qarray` `C` for an integer `q` corresponding to .ind field (`C.ind[q][blockindex]`) at index positions `a` reported as a tuple; returns an integer `x` corresponding to where in the block the element is
"""
function scaninds(blockindex::intType,q::intType,C::qarray,a::NTuple{N,intType}) where N
  x = 0
  notmatchingrow = q > 0
  @inbounds while notmatchingrow && x < size(C.ind[q][blockindex],2)
    x += 1
    r = 0
    matchvals = true
    @inbounds while matchvals && r < length(C.currblock[blockindex])
      r += 1
      matchvals = C.ind[q][blockindex][r,x] + 1 == a[C.currblock[blockindex][r]]
    end
    notmatchingrow = !matchvals
  end
  return x
end



"""
    tensor2disc(name,tensor[,ext=".dmrjulia"])

Writes `tensor` to disc with the Serialization package and filename `name`*`ext`

See also: [`tensorfromdisc`](@ref)
"""
function tensor2disc(name::String,tensor::TensType;ext::String=file_extension)
  Serialization.serialize(name*ext,tensor)
  nothing
end

"""
    setindex!(B,A,i[,ext=file_extension])

Put tensor `A` (`allTens`) into a large vector `H` (`bigvec`) at position `i`; can adjust `ext` to make a new file extension other than the default
"""
function setindex!(H::bigvec,A::allTens,i::intType;ext::String=file_extension)
  tensor2disc(H.V[i],A,ext=ext)
  nothing
end
