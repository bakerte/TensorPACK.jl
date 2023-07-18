#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v1.0
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1+)
#
#=
module tensornetwork
#using ..shuffle
using ..tensor
using ..QN
using ..Qtensor
#using ..Qtask
using ..contractions
using ..decompositions
using ..MPutil
=#
abstract type TNobj end
export TNobj

abstract type TNnetwork end
export TNnetwork

"""
    nametens{W,B}

named tensor with tensor of type `W` and type of names `B`

# Fields:
+ `N::W`: Tensor stored
+ `names::Array{B,1}`: names of all indices
"""
mutable struct nametens{W,B} <: TNobj where {W <: TensType, B <: Union{Any,String}}
  N::W
  names::Array{B,1}
end

"""
    nametens(Qt,namez)

constructor for named tensor from a tensor `Qt` and vector of index names `namez`
"""
function nametens(Qt::TensType,namez::Array{B,1};regTens::Bool=false)::TNobj where B <: Union{Any,String}
  newQt = !regTens && typeof(Qt) <: AbstractArray ? tens(Qt) : Qt
  return nametens{typeof(newQt),B}(newQt,namez)
end

"""
    nametens(Qt,string)

constructor for named tensor from a tensor `Qt` and vector of index names `string` where an integer is added onto `string` to make the full name of the index
"""
function nametens(Qt::T,namez::String;regTens::Bool=false)::TNobj where T <: TensType
  return nametens(Qt,[namez*"$i" for i = 1:basedims(Qt)],regTens=regTens)
end
export nametens



"""
  directedtens{W,B}

named tensor with named tensor of type `W` and vector of Booleans `B`. Creates a directed graph for use in MERA computations

# Fields:
+ `T::W`: named tensor stored
+ `names::Array{B,1}`: arrows of all indices

See also: [`nametens`](@ref)
"""
mutable struct directedtens{W,B} <: TNobj where {W <: nametens, B <: Bool}
  T::W
  arrows::Array{B,1}
  conj::Bool
end



"""
  directedtens(Qt,vecbools)

constructor for named tensor `Qt` and vector of directed arrows `vecbools`
"""
function directedtens(Qt::nametens{W,B},vecbools::Array{Bool,1};conj::Bool=false) where {W <: TensType, B <: Union{Any,String}}
  return directedtens(Qt,vecbools,conj)
end

"""
  directedtens(Qt,namez)

constructor for named tensor from a tensor `Qt` and vector of index names `namez`
"""
function directedtens(Qt::TensType,namez::Array{B,1},vecbools::Array{Bool,1};regTens::Bool=false,conj::Bool=false)::TNobj where B <: Union{Any,String}
  return directedtens(nametens(newQt,namez,regtens=regtens),vecbools,conj=conj)
end

"""
  directedtens(Qt,string)

constructor for named tensor from a tensor `Qt` and vector of index names `string` where an integer is added onto `string` to make the full name of the index
"""
function directedtens(Qt::T,namez::String;regTens::Bool=false)::TNobj where T <: TensType
  return directedtens(nametens(Qt,namez,regtens=regtens),vecbools,conj=conj)
end
export directedtens

"""
    network{N,W}

Generates a network of TNobjs that stores more than one named tensor

# Fields:
+ `net::NTuple{N,W}`: A network of named tensors
"""
mutable struct network{W} <: TNnetwork where W  <: TNobj
  net::Array{W,1}
end

"""
    network(Qts)

constructor to generates a network of TNobjs that stores a vector of named tensors `Qts`
"""
function network(Qts::Array{W,1}) where W  <: TNobj
  return network{W}(Qts)
end

"""
    network(Qts)

converts named tensor to a network with a single tensor element
"""
function network(Qts::W...) where W  <: TNobj #where S <: Union{Any,String}
  return network{W}([Qts[i] for i = 1:length(Qts)])
end

"""
  network(Qt,i)

converts named tensor `Qt` to a network with `i` copied elements not shallow copied
"""
function network(Qts::W,n::Integer) where W  <: TNobj #where S <: Union{Any,String}
return network{W}([copy(Qts) for i = 1:n])
end
export network

import ..Base.getindex
function getindex(Qts::TNnetwork,i::Integer)
  return Qts.net[i]
end

function getindex(Qts::nametens,i::Integer)
  return Qts.N[i]
end

function getindex(Qts::directedtens,i::Integer)
  return getindex(Qts.T,i)
end

import ..Base.setindex!
function setindex!(Qts::TNnetwork,newTens::TNobj,i::Integer)
  return Qts.net[i] = newTens
end

import ..Base.length
function length(Qts::TNnetwork)
  return length(Qts.net)
end

function contractinds(A::nametens,B::nametens;check::Bool=false)
  pairs = Array{Bool,2}(undef,length(A.names),length(B.names))
  counter = 0
  for b = 1:size(pairs,2)
    for a = 1:size(pairs,1)
      if A.names[a] == B.names[b]
        counter += 1
        pairs[a,b] = true
#        pairs[b,a] = true
      else
        pairs[a,b] = false
#        pairs[b,a] = false
      end
    end

    if check
      checkcounter = 0
      @inbounds @simd for x = 1:size(pairs,1)
        checkcounter += pairs[x,b]
      end
      if checkcounter > 1
        error("Indices not paired on contraction of named tensors (duplicate index name detected)")
      end
    end
  end

  vecA = Array{intType,1}(undef,counter)
  vecB = Array{intType,1}(undef,counter)

  newcounter = 0
  b = 0
  while newcounter < counter
    b += 1
    search_bool = true
    for a = 1:size(pairs,1)
      if pairs[a,b]
        newcounter += 1
        vecA[newcounter] = a
        vecB[newcounter] = b
        search_bool = false
      end
    end
  end

  return vecA,vecB,pairs
end

function contractinds(A::directedtens,B::directedtens)
  return contractinds(A.T,B.T)
end

#  import ..Qtensor.*
"""
    *(A,B...)

Contracts `A` and any number of `B` along common indices; simple algorithm at present for the order
"""
function *(A::nametens,B::nametens,fct::Function=contract)

  vecA,vecB,pairs = contractinds(A,B)

  ynewnames = 0
  for y = 1:size(pairs,2)
    counter = 0
    @inbounds @simd for x = 1:size(pairs,1)
      counter += pairs[x,y]
    end
    if counter == 0
      ynewnames += 1
    end
  end

  xnewnames = 0
  for x = 1:size(pairs,1)
    counter = 0
    @inbounds @simd for y = 1:size(pairs,2)
      counter += pairs[x,y]
    end
    if counter == 0
      xnewnames += 1
    end
  end

  newnames = Array{String,1}(undef,xnewnames+ynewnames)

  name_counter = 0
  x = 0
  while name_counter < xnewnames
    x += 1
    counter = 0
    @inbounds @simd for y = 1:size(pairs,2)
      counter += pairs[x,y]
    end
    if counter == 0
      name_counter += 1
      newnames[name_counter] = A.names[x]
    end
  end

  y = 0
  while name_counter < xnewnames + ynewnames
    y += 1
    counter = 0
    @inbounds @simd for x = 1:size(pairs,1)
      counter += pairs[x,y]
    end
    if counter == 0
      name_counter += 1
      newnames[name_counter] = B.names[y]
    end
  end

  newTens = contract(A.N,vecA,B.N,vecB)

  return nametens(newTens,newnames)
end

function *(A::directedtens,B::directedtens)
  if A.conj && B.conj
    C = *(A.T,B.T,fct=ccontractc)
  elseif A.conj && !B.conj
    C = *(A.T,B.T,fct=ccontract)
  elseif !A.conj && B.conj
    C = *(A.T,B.T,fct=contractc)
  else
    C = *(A.T,B.T,fct=contract)
  end
  
  newarrows = Array{Bool,1}(undef,ndims(C))
  counter = 0
  for a = 1:length(C.names)
    if C.names[a] in A.T.names
      counter += 1
      newarrows[counter] = A.conj ? !A.arrows[a] : A.arrows[a]
    end
  end
  if counter != length(newarrows)
    for a = 1:length(C.names)
      if C.names[a] in B.T.names
        counter += 1
        newarrows[counter] = B.conj ? !B.arrows[a] : B.arrows[a]
      end
    end
  end
  return directedtens(C,newarrows,false)
end

function *(R::TNobj...)
  out = *(R[1],R[2])
  @simd for b = 3:length(R)
    out = *(out,R[b])
  end
  return out
end


function sum(R::TNobj)
  return sum(R.N)
end

"""
    *(a,b)

concatenates string `a` with integer `b` after converting integer to a string
"""
function *(a::String,b::Integer)
  return a*string(b)
end

import Base.permutedims
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
#  import ..Qtensor.permutedims!
"""
    permtuedims!(A,order)

Permutes named tensor `A` according to `order` (ex: [[1,2],[3,4]] or [["a","b"],["c","d"]])

See also: [`permutedims`](@ref)
"""
function permutedims!(A::TNobj,order::Array{W,1}) where W <: String
  newnumberorder = intType[]
  for i = 1:size(order,1)
    for j = 1:size(A.names,1)
      if order[i] == A.names[j]
        push!(newnumberorder,j)
        continue
      end
    end
  end
  return permutedims!(A,newnumberorder)
end

function permutedims!(B::TNobj,order::Array{W,1}) where W <: Integer

  A = typeof(B) <: nametens ? B : B.T

  A.N = permutedims!(A.N,order)
  A.names = A.names[order]
#    A.arrows = A.arrows[order]

  if typeof(B) <: directedtens
    B.T = A
  end

  return B
end

"""
    matchnames(AA,order,q)

Matches `order` (a length 2 vector of vectors of strings for indices) to the indices in `AA` for the left (right) with `q`=1 (2)
"""
function matchnames(AA::nametens,order::Array{B,1}) where B <: Union{Any,String}
  vect = Array{intType,1}(undef,length(order))
  for a = 1:length(order)
    condition = true
    w = 0
    while condition && w < length(AA.names)
      w += 1
      if order[a] == AA.names[w]
        vect[a] = w
        condition = false
      end
    end
  end
  return vect
end
#=
"""
    findinds(AA,order)

prepares return indices and tensor `AA` for decomposition
"""
function findinds(AA::TNobj,order::Array{Array{B,1},1}) where B <: Union{Any,String}
  left = matchnames(AA,order,1)
  right = matchnames(AA,order,2)
  return left,right
end
=#
#  import ..decompositions.svd
"""
    svd(A,order[,mag=,cutoff=,m=,name=,leftadd=,rightadd=])

Generates SVD of named tensor `A` according to `order`; same output as regular SVD but with named tensors

# Naming created index:
+ the index to the left of `D` is `vcat(name,leftadd)`
"""
function svd(AA::nametens,order::Array{Array{B,1},1};mag::Number = 0.,cutoff::Number = 0.,
              m::Integer = 0,power::Integer=2,name::String="svdind",leftadd::String="L",
              rightadd::String="R",nozeros::Bool=false) where B <: Union{Any,String}

  left = matchnames(AA,order[1])
  right = matchnames(AA,order[2])

  neworder = Array{intType,1}[left,right]
  leftname = name * leftadd
  rightname = name * rightadd

  U,D,V,truncerr,newmag = svd(AA.N,neworder,power=power,mag=mag,cutoff=cutoff,m=m,nozeros=nozeros)

  TNobjU = nametens(U,vcat(AA.names[left],[leftname]))
  TNobjD = nametens(D,[leftname,rightname])
  TNobjV = nametens(V,vcat([rightname],AA.names[right]))

  return TNobjU,TNobjD,TNobjV,truncerr,newmag
end

"""
    symsvd(A,order[,mag=,cutoff=,m=,name=,leftadd=,rightadd=])

Takes `svd` of `A` according to `order` and returns U*sqrt(D),sqrt(D)*V

See also: [`svd`](@ref)
"""
function symsvd(AA::TNobj,order::Array{Array{B,1},1};mag::Number = 0.,power::Integer=2,
                cutoff::Number = 0.,m::Integer = 0,name::String="svdind",rightadd::String="L",
                leftadd::String="R") where B <: Union{Any,String}

  U,D,V,truncerr,mag = svd(AA,order,power=power,mag=mag,cutoff=cutoff,m=m,name=name,leftadd=leftadd,rightadd=rightadd)
  S1 = sqrt!(D)
  return U*S1,S1*V,truncerr,mag
end
export symsvd

#  import ..decompositions.eigen
"""
    eigen(A,order[,mag=,cutoff=,m=,name=,leftadd=])

Generates eigenvalue decomposition of named tensor `A` according to `order`; same output as regular eigenvalue but with named tensors

# Naming created index:
+ the index to the left of `D` is `vcat(name,leftadd)`
"""
function eigen(AA::nametens,order::Array{Array{B,1},1};mag::Number = 0.,cutoff::Number = 0.,
                m::Integer = 0,name::String="eigind",leftadd::String="L") where B <: Union{Any,String}

  left = matchnames(AA,order[1])
  right = matchnames(AA,order[2])
  neworder = [left,right]
  leftname = name*leftadd

  D,U,truncerr,newmag = eigen(AA.N,order,mag=mag,cutoff=cutoff,m=m)

  TNobjD = nametens(D,[leftname,rightname])
  TNobjU = nametens(U,vcat(AA.names[left],[leftname]))
  return TNobjD,TNobjU,truncerr,newmag
end

"""
    conj(A)

Conjugates named MPS `A`

See also: [`conj!`](@ref)
"""
function conj(A::TNnetwork)
  return network([conj(A.net[i]) for i = 1:length(A)])
end

import Base.copy
"""
    copy(A)

Returns a copy of named tensor `A`
"""
function copy(A::nametens{W,B}) where {W <: Union{qarray,AbstractArray,denstens}, B <: Union{Any,String}}
  return nametens{W,B}(copy(A.N),copy(A.names))
end

"""
  copy(A)

Returns a copy of network of named tensors `A`
"""
function copy(A::TNnetwork)
  return network([copy(A.net[i]) for i = 1:length(A)])
end

import Base.println
"""
    println(A[,show=])

Prints named tensor `A`

# Outputs:
+ `size`: size of `A`
+ `index names`: current names on `A`
+ `arrowss`: fluxes for each index on `A`
+ `elements`: elements of `A` if reshaped into a vector (out to `show`)
"""
function println(A::TNobj;show::Integer=10)

  println("size = ",size(A))
  println("index names = ",A.names)
  if typeof(A.N) <: denstens ||  typeof(A.N) <: qarray
    temp = length(A.N.T)
    maxshow = min(show,temp)
    println("elements = ",A.N.T[1:maxshow])
  else
    rAA = reshape(A.N,prod(size(A)))
    temp = length(rAA)
    maxshow = min(show,temp)
    if length(rAA) > maxshow
      println("elements = ",rAA[1:maxshow],"...")
    else
      println("elements = ",rAA[1:maxshow])
    end
  end
  println()
  nothing
end

import Base.size
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
  return size(A.N,w)
end

"""
    norm(A)

Gives the norm of named tensor `A`
"""
function norm(A::TNobj)
  return norm(A.N)
end

"""
    div!(A,num)

Gives the division of named tensor `A` by number `num`

See also: [`/`](@ref)
"""
function div!(A::TNobj,num::Number)
  A.N = div!(A.N,num)
  return A
end

"""
    /(A,num)

Gives the division of named tensor `A` by number `num`

See also: [`div!`](@ref)
"""
function /(A::TNobj,num::Number)
  return div!(copy(A),num)
end

"""
    mult!(A,num)

Gives the multiplication of named tensor `A` by number `num`

See also: [`*`](@ref)
"""
function mult!(A::TNobj,num::Number)
  A.N = mult!(A.N,num)
  return A
end

"""
    *(A,num)

Gives the multiplication of named tensor `A` by number `num` (commutative)

See also: [`mult!`](@ref)
"""
function *(A::TNobj,num::Number)
  return mult!(copy(A),num)
end

function *(num::Number,A::TNobj)
  return A*num
end

function add!(A::TNobj,B::TNobj)
  reorder = matchnames(A,B.names)
  if !issorted(reorder)
    C = permutedims(B,reorder)
  else
    C = B
  end
  A.N = add!(A.N,C.N)
  return A
end

"""
    +(A,B)

Adds tensors `A` and `B`

See also: [`add!`](@ref)
"""
function +(A::TNobj,B::TNobj)
  return add!(copy(A),B)
end

"""
    sub!(A,B)

Subtracts tensor `A` from `B` (changes `A`)

See also: [`-`](@ref)
"""
function sub!(A::TNobj,B::TNobj)
  reorder = matchnames(A,B.names)
  if !issorted(reorder)
    C = permutedims(B,reorder)
  else
    C = B
  end
  A.N = sub!(A.N,C.N)
  return A
end

"""
    -(A,B)

Subtracts tensor `A` from `B`

See also: [`sub!`](@ref)
"""
function -(A::TNobj,B::TNobj)
  return sub!(copy(A),B)
end

import Base.sqrt
"""
    sqrt(A)

Takes the square root of named tensor `A`

See also: [`sqrt`](@ref)
"""
function sqrt(A::TNobj;root::Number=0.5)
  B = copy(A)
  return sqrt!(B,root=root)
end

#  import ..TENPACK.sqrt!
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

import Base.ndims
"""
    ndims(A)

Returns the number of indices of named tensor `A`
"""
function ndims(A::nametens)
  return length(A.names)
end

function ndims(A::directedtens)
  return length(A.arrows)
end

"""
    conj!(A)

Conjugates named tensor `A` in-place

See also: [`conj`](@ref)
"""
function conj!(A::nametens)
  conj!(A.N)
  nothing
end


function conj!(A::directedtens)
  @inbounds @simd for w = 1:ndims(A)
    A.arrows[w] = !A.arrows[w]
  end
  A.conj = !A.conj
  nothing
end

import LinearAlgebra.conj
"""
    conj(A)

Conjugates named tensor `A`

See also: [`conj!`](@ref)
"""
function conj(A::TNobj)::TNobj
  B = copy(A)
  conj!(B)
  return B
end

"""
    trace(A)

Computes the trace of named tensor `A` over indices with 1) the same name and 2) opposite arrowss
"""
function trace(B::TNobj)
  A = typeof(B) <: nametens ? B : B.T
  vect = Array{intType,1}[]
  for w = 1:length(A.names)
    condition = true
    z = w+1
    while condition && z < length(A.names)
      z += 1
      if A.names[w] == A.names[z]
        push!(vect,[w,z])
        condition = false
      end
    end
  end
  return trace(A.N,vect)
end

"""
    trace(A,inds)

Computes the trace of named tensor `A` with specified `inds` (integers, symbols, or strings--ex: [[1,2],[3,4],[5,6]])
"""
function trace(A::nametens,inds::Array{Array{W,1},1}) where W <: Union{Any,Integer}
  if W <: Integer
    return trace(A.N,inds)
  else
    vect = Array{intType,1}[zeros(intType,2) for i = 1:length(inds)]
    for w = 1:length(A.names)
      matchindex!(A,vect,inds,w,1)
      matchindex!(A,vect,inds,w,2)
    end
    return trace(A.N,vect)
  end
end

function trace(A::directedtens,inds::Array{Array{W,1},1}) where W <: Union{Any,Integer}
  B = trace(A.T,inds)
  newinds = vcat(inds...)
  leftoverinds = setdiff([i for i = 1:ndims(A)],newinds)
  newarrows = A.arrows[leftoverinds]
  return directedtens(B,newarrows,A.conj)
end

"""
    trace(A,inds)

Computes the trace of named tensor `A` with specified `inds` (integers, symbols, or strings--ex: [1,2])
"""
function trace(A::TNobj,inds::Array{W,1}) where W <: Union{Any,Integer}
  return trace(A,[inds])
end
export trace

"""
    matchindex!(A,vect,inds,w,q)

takes index names from `vect` over indices `inds` (position `w`, index `q`, ex: inds[w][q]) and converts into an integer; both `vect` and `index` are of the form Array{Array{?,1},1}

See also: [`trace`](@ref)
"""
function matchindex!(A::nametens,vect::Array{Array{P,1},1},inds::Array{Array{W,1},1},w::Integer,q::Integer) where {W <: Union{Any,Integer}, P <: Integer}
  convInds!(A,inds,vect)
  nothing
end

"""
    convInds!(A,inds,vect)

converts named indices in `A` to integers; finds only indices specified in `inds` and returns `vect` with integers; does nothing if its only integers
"""
function convInds!(A::nametens,inds::Array{Array{W,1},1},vect::Array{Array{P,1},1}) where {W <: Union{Any,Integer}, P <: Integer}
  if W <: Integer
    return inds
  end
  for a = 1:length(vect)
    for b = 1:length(vect[a])
      saveind = b > 1 ? vect[a][b-1] : 0
      for c = 1:length(A.names)
        if inds[a][b] == A.names[c] && saveind != c
          vect[a][b] = c
          saveind = c
        end
      end
    end
  end
  return vect
end

"""
  swapname!(A,labels)

Finds elements in `labels` (must be length 2) and interchanges the name. For example, `swapname!(A,["a","c"])` will find the label `"a"` and `"c"` and interchange them. A chain of swaps can also be requested (i.e., `swapname!(A,[["a","c"],["b","d"]])`)

Works as a pseudo-permute in many cases. Will not permute if the names are not found.

See also: [`swapnames!`](@ref)
"""
function swapname!(A::nametens,inds::Array{Array{W,1},1}) where W <: Any
  for c = 1:length(inds)
    x = 1
    while x < length(A.names) && A.names[x] != inds[c][1]
      x += 1
    end
    y = 1
    while y < length(A.names) && A.names[y] != inds[c][2]
      y += 1
    end
    if inds[c] == [A.names[x],A.names[y]]
      A.names[x],A.names[y] = A.names[y],A.names[x]
    end
  end
  nothing
end

function swapname!(A::nametens,inds::Array{W,1}) where W <: Any
  swapname!(A,[inds])
end
export swapname!

"""
  swapnames!(A,labels)

Finds elements in `labels` (must be length 2) and interchanges the name. For example, `swapname!(A,["a","c"])` will find the label `"a"` and `"c"` and interchange them. A chain of swaps can also be requested (i.e., `swapname!(A,[["a","c"],["b","d"]])`)

Works as a pseudo-permute in many cases. Will not permute if the names are not found.

See also: [`swapname!`](@ref)
"""
function swapnames!(A::nametens,inds::Array{Array{W,1},1}) where W <: Any
  swapname!(A,inds)
end

function swapnames!(A::nametens,inds::Array{W,1}) where W <: Any
  swapname!(A,[inds])
end
export swapnames!


"""
    rename!(A,inds)

replaces named indices in `A` with indices in `inds`; either format [string,[string,arrow]] or [string,string] or [string,[string]] is accepted for `inds`
"""
function rename!(A::nametens,inds::Array{Array{W,1},1}) where W <: Any
  for a = 1:length(inds)
    condition = true
    b = 0
    while condition && b < length(A.names)
      b += 1
      if A.names[b] == inds[a][1]
        if typeof(inds[a][2]) <: Array
          A.names[b] = inds[a][2][1]
        else
          A.names[b] = inds[a][2]
        end
      end
    end
  end
  nothing
end
#=            one = ["s1",["i1",false]]
          two = ["s2",["i2",false]]
          three = ["s3",["i3",true]]
          four = ["s4",["i4",true]]
          rename!(A1,[one,two,three,four])=#

"""
    rename!(A,currvar,newvar[,arrows])

replaces a string `currvar` in named indices of `A` with `newvar`; can also set arrows if needed
"""
function rename!(A::nametens,currvar::String,newvar::String)
  for a = 1:length(A.names)
    loc = findfirst(currvar,A.names[a])
    if !(typeof(loc) <: Nothing)
      first = loc[1] == 1 ? "" : A.names[a][1:loc[1]-1]
      last = loc[end] == length(A.names[a]) ? "" : A.names[a][loc[end]+1]
      newstring = first * newvar * last
      A.names[a] = newstring
    end
  end
  nothing
end
export rename!

function rename(A::nametens,inds::Array{Array{W,1},1}) where W <: Any
  B = copy(A)
  rename!(B,inds)
  return B
end
export rename

function addindex!(X::nametens,Y::nametens)
  if typeof(X.N) <: denstens || typeof(X.N) <: qarray
    X.N.size = (size(X.N)...,1)
  else
      error("trying to convert an array to the wrong type...switch to using the denstens class")
  end
  if typeof(Y.N) <: denstens || typeof(Y.N) <: qarray
    Y.N.size = (size(Y.N)...,1)
  elseif typeof(Y.N) <: AbstractArray
      error("trying to convert an array to the wrong type...switch to using the denstens class")
  end
  push!(X.names,"extra_ones")
  push!(Y.names,"extra_ones")
  nothing
end
export addindex!

function addindex(X::nametens,Y::nametens)
  A = copy(X)
  B = copy(Y)
  addindex!(A,B)
  return A,B
end
export addindex

function joinTens(X::nametens,Y::nametens)
  A,B = addindex(X,Y)
  return A*B
end
export joinTens





































 #
 # Kiana
 # 
 struct Indices
	names::Vector{String}
	dimensions::Vector{Int64}
end


function get_weight(tensor::nametens{tens{W}, String}) where W <: Number #stable2
	sum = 0
	for dim in tensor.N.size
		if dim != 1
			sum +=1 
		end
	end
	return sum
end


function find_start(graph::Vector{nametens{tens{W}, String}})::nametens{tens{W}, String} where W <: Number #stable2
	start = graph[1]
	min_num_ind = get_weight(start)
	min_cost = get_cost(start.N.size)

	for tensor in graph
		num_ind = get_weight(tensor)
		cost = get_cost(tensor.N.size)

		if num_ind<min_num_ind
			start = tensor
			min_num_ind = num_ind
			min_cost = cost

		elseif (num_ind==min_num_ind)
			if (cost<min_cost)
				start = tensor
				min_num_ind = num_ind
				min_cost = cost
			end
		end
	end
	return start
end


function connecting_edges(graph::Vector{nametens{tens{W}, String}})::Dict{String, Vector{nametens{tens{W}, String}}} where W <: Number #stable2
	shared_edges = Dict{String, Vector{nametens{tens{W}, String}}}()

	for tensor in graph
		for edge in tensor.names
			if edge in keys(shared_edges)
				push!(shared_edges[edge], tensor) # possible ways to make this more efficient???
			else
				shared_edges[edge] = [tensor]
			end
		end
	end
	return shared_edges
end


function common_info(temp_start::Indices, right_tensor::nametens{tens{W}, String}) where W <: Number  #stable2
	num_common = 0
	mult_commom = 1

	for (pos, edge_name) in enumerate(temp_start.names)
		if edge_name in right_tensor.names 
			num_common += 1
			mult_commom *= temp_start.dimensions[pos]
		end
	end
	return (num_common, mult_commom)
end


function get_cost(edges::Vector{Int64})::Int64 #stable2
	cost = 1
	for edge_dim in edges
		cost *= edge_dim
	end
	return cost
end 


function find_next(temp_start::Indices, start::nametens{tens{W}, String}, shared_edges::Dict{String, Vector{nametens{tens{W}, String}}})::Tuple{nametens{tens{W}, String}, Int64} where W <: Number #stable2
	new_edges = -1
	next = start
	min_cost = -1
	max_common = 0

	for edge in temp_start.names
		if haskey(shared_edges, edge)
			if length(shared_edges[edge]) != 0
				right_tensor = shared_edges[edge][1]

				common_details = common_info(temp_start, right_tensor) # consider returing the names of the indices they have in common

				num_common = common_details[1]
				added_edges = get_weight(right_tensor)-num_common
				cost_tot = (get_cost(temp_start.dimensions)*get_cost(right_tensor.N.size))÷common_details[2] 

				if (min_cost==-1)
					new_edges = added_edges
					min_cost = cost_tot
					max_common = num_common
					next = right_tensor

				elseif (num_common>max_common)
					new_edges = added_edges
					min_cost = cost_tot
					max_common = num_common
					next = right_tensor

				elseif (num_common==max_common)
					if (added_edges<new_edges)
						new_edges = added_edges
						min_cost = cost_tot
						max_common = num_common
						next = right_tensor

					elseif (added_edges==new_edges)
						if (cost_tot<min_cost)
							new_edges = added_edges
							min_cost = cost_tot
							max_common = num_common
							next = right_tensor
						end
					end
				end
			end
		end
	end
	return (next, max_common)
end


# tried to use vcat but it still takes 1 alloc for each vcat, so it is still 2 alloc to run the algorithm.
function update_temp(left::Indices, right::Indices, max_common::Int64) #stable2
	new_length = (length(left.names)+length(right.names))-(2*max_common)

	new_names = Vector{String}(undef, new_length) # takes 1 alloc
	new_dimensions = Vector{Int64}(undef, new_length) # takes 1 alloc 

	num_updated = 1
	for pos in 1:length(left.names)
		if !(left.names[pos] in right.names)
			new_names[num_updated] = left.names[pos]
			new_dimensions[num_updated] = left.dimensions[pos]

			num_updated += 1
		end
	end

	for pos in 1:length(right.names)
		if !(right.names[pos] in left.names)
			new_names[num_updated] = right.names[pos]
			new_dimensions[num_updated] = right.dimensions[pos]

			num_updated += 1
		end
	end
	return Indices(new_names, new_dimensions)
end


function update_edges(next::nametens{tens{W}, String}, shared_edges::Dict{String, Vector{nametens{tens{W}, String}}})::Dict{String, Vector{nametens{tens{W}, String}}} where W <: Number #stable2
	for edge in next.names
		adjacent = shared_edges[edge]
		location = findfirst(==(next), adjacent)
		deleteat!(adjacent, location)
	end
	return shared_edges
end


# only takes one allocation
function permute(edges::Indices, position) #stable2
	cost = 1
#=
	for (pos, edge) in enumerate(common_edges)
		index = findfirst(==(edge), edges.names)
		position[pos] = index

	end
=#
	sort!(position)

	if !(length(edges.names) in position) && !(1 in position)
		for edge_dim in edges.dimensions
			cost *= edge_dim
		end
		# weight = cost÷length(edges.names)
		return cost,position
	else
		for pos in range(2, length(position))
			if (position[pos]-position[pos-1]) != 1
				for edge_dim in edges.dimensions
					cost *= edge_dim
				end
				# weight = cost÷length(edges.names)
				return cost,position
			end
		end
	end
	return 0,position # returning a tuple versus  A, B seems to make no difference in terms of alloc
end


function find_common_edges(left_edges::Vector{String}, right_edges::Vector{String}, num_common::Int64) #stable2
	pos_left = Array{intType,1}(undef,num_common)
	pos_right = Array{intType,1}(undef,num_common)
	val = 1
	counter = 0
	while val <= num_common #&& counter < length(left_edges)
		counter += 1
		if left_edges[counter] in right_edges
			pos_left[val] = counter
			pos_right[val] = findfirst(==(left_edges[counter]),right_edges)
			val +=1
		end
	end
	return pos_left, pos_right
end


function order(left::Indices, right::Indices, left_order::Vector{Int64}) #stable2
	last_pos = -1
	cost = 1

	for pos in left_order
		edge = left.names[pos]
		index = findfirst(==(edge), right.names)
		if !(index>last_pos)
			for edge_dim in left.dimensions
				cost *= edge_dim
				
			end

			cost = cost÷length(left.names)
			for edge_dim in right.dimensions
				cost *= edge_dim
			end

			cost = cost÷length(right.names)
			return cost
		else
			last_pos = index
		end
	end
	return 0
end


function permute_cost(left::Indices, right::Indices, pos_left::Vector{Int64}, pos_right::Vector{Int64}) #stable2

	costA,posA = permute(left, pos_left)
	costB,posB = permute(right, pos_right)


	if !(costA==0)&&!(costB==0)
		return costA+costB
	elseif (costA==0)&&(costB==0)
		return order(left, right, posA)
	else
		return costA+costB
	end
end



function best_order(next_tensors::Vector{nametens{tens{W}, String}}, num_connecting::Vector{Int64}, temp_start::Indices) where W <: Number #stable2

	temp_next = Indices(next_tensors[1].names, next_tensors[1].N.size)

	left = best_order_helper(next_tensors, num_connecting, temp_start, 0, "", "left", 1)
	right = best_order_helper(next_tensors, num_connecting, temp_start, 0, "", "right", 1)

	if left[1]<=right[1]
		return left[2]

	else 
		return right[2]

	end

end

#THIS FUNCITON IS VERY EXPENSIVE
#new idea: don't check all the different combinations, just look left and right and make a decision on the added cost
# it will NOT pick the best but hopefully it will pick a "good" choice anyways
function best_order_helper(next_tensors::Vector{nametens{tens{W}, String}}, num_connecting::Vector{Int64}, temp_start::Indices, cost::Int64, order::String, side::String, depth::Int64) where W <: Number #stable2

	if length(order) == length(next_tensors)
		return (cost, order)

	elseif (side == "left")
		temp_next = Indices(next_tensors[depth].names, next_tensors[depth].N.size)
		pos_left,pos_right = find_common_edges(temp_next.names, temp_start.names, num_connecting[depth]) #takes (1 allocation: 64 bytes)

		cost += permute_cost(temp_next, temp_start, pos_left,pos_right) #takes 0.000001 seconds (4 allocations: 320 bytes)

		left = update_temp(temp_next, temp_start, num_connecting[depth]) #takes 0.000002 seconds (2 allocations: 208-224 bytes)
		order = order*"L"
		depth += 1

		next_left = best_order_helper(next_tensors, num_connecting, left, cost, order, "left", depth)
		next_right = best_order_helper(next_tensors, num_connecting, left, cost, order, "right", depth)

		if next_left[1]<=next_right[1]
			return next_left
		else 
			return next_right
		end

	else
		temp_next = Indices(next_tensors[depth].names, next_tensors[depth].N.size)
		pos_left,pos_right = find_common_edges(temp_start.names, temp_next.names, num_connecting[depth])

		cost += permute_cost(temp_start, temp_next, pos_left,pos_right)

		right = update_temp(temp_start, temp_next, num_connecting[depth])
		order = order*"R"
		depth += 1


		next_left = best_order_helper(next_tensors, num_connecting, right, cost, order, "left", depth)
		next_right = best_order_helper(next_tensors, num_connecting, right, cost, order, "right", depth)

		if next_left[1]<=next_right[1]
			return next_left
		else 
			return next_right
		end
	end
end

function contract!(graph::network) #stable2
	graph = graph.net
	contract_around = false

	start = find_start(graph) #takes 0 alloc
	
	location = findfirst(==(start), graph) #takes 0 alloc
	deleteat!(graph, location) #takes 0 alloc

  shared_edges = connecting_edges(graph) #takes 1.01 k allocations: 93.688 KiB
	
  initial_start = Indices(start.names, start.N.size)

	future = 3
	next_tensors = Vector{nametens{tens{Float64}, String}}(undef, future) #takes (1 allocation: 80 bytes)
	num_connecting = Vector{Int64}(undef, future) #takes (1 allocation: 80 bytes)

	while (length(graph)>0)
		temp_start = Indices(start.names, start.N.size)
		future = 3

		if (length(graph)<future)
			future = length(graph)
			next_tensors = Vector{nametens{tens{Float64}, String}}(undef, length(graph))
			num_connecting = Vector{Int64}(undef, length(graph))

		end

		while (length(graph)>0)&&(future>0) #the time of the algorithm seems to jump around a bit.. possible type instability?..

			if (contract_around) # we have contracted around the starting tensor so we can follow the alogorihtm as expected
				next_details = find_next(temp_start, start, shared_edges) #takes 0 alloc cause this to return the common edges?...
				next = next_details[1]

				temp_next = Indices(next.names, next.N.size)
				shared_edges = update_edges(next, shared_edges) #takes 0 alloc

				next_tensors[length(next_tensors)-future+1] = next #create a nicer looking index system?...
				num_connecting[length(next_tensors)-future+1] = next_details[2]

				location = findfirst(==(next), graph)
				deleteat!(graph, location)

				# this is semi expensive, takes about 600 alloc for the whole program (2*300 tensors)
				temp_start = update_temp(temp_start, temp_next, next_details[2]) #takes 2 allocations: 208-224 bytes
				future -= 1

			else
				next_details = find_next(initial_start, start, shared_edges)
				next = next_details[1]

				if (next==start)
					contract_around = true

				else

					temp_next = Indices(next.names, next.N.size)
					shared_edges = update_edges(next, shared_edges)

					next_tensors[length(next_tensors)-future+1] = next #create a nicer looking index system?...
					num_connecting[length(next_tensors)-future+1] = next_details[2]

					location = findfirst(==(next), graph)
					deleteat!(graph, location)

					temp_start = update_temp(temp_start, temp_next, next_details[2])
					future -= 1

				end

			end
		end

		temp_start = Indices(start.names, start.N.size)
		#THIS IS VERY EXPENSIVE (112*(300 tensors/3))
		answer = best_order(next_tensors, num_connecting, temp_start) #takes 0.000032 seconds (112 allocations: 7.953 KiB)

		#THIS IS VERY EXPENSIVE (163*(300 tensors/3))
		#takes about 163 alloc for each time this for loop is done
		for pos in 1:length(answer)
			if answer[pos] == 'L'
				start = next_tensors[pos]*start
			else 
				start = start*next_tensors[pos]
			end
		end

		#IN TOTAL THE 2 MOST EXPENSIVE PARTS TAKE ABOUT 27,500 ALLOC OUT OF 30,915 ALLOC (about 88.95%)
	end
	return start
end