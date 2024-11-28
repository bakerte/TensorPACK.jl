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
    Z = X*Y

Computes the equivalent of contract(X,ndims(X),Y,1) between any two `densTensType`
"""
function *(X::denstens,Y::denstens)
  #if !(typeof(X) <: Diagonal) && !(typeof(Y) <: Diagonal)

    X,Y = checkType(X,Y)

    m = ndims(X) == 1 ? 1 : prod(w->size(X,w),1:ndims(X)-1)
    k = size(X,ndims(X))
    n = ndims(Y) < 2 ? 1 : prod(w->size(Y,w),2:ndims(Y))

    out = libmult('N','N',X,Y,m,k,k,n)

    A,B = X,Y
    iA = (ndims(X),)
    iB = (1,)
    Aremain = ndims(A)-1
    Bremain = ndims(B)-1
    AAsizes = Array{intType,1}(undef,Aremain+Bremain)
    for w = 1:Aremain+Bremain
      AAsizes[w] = makesize(w,A,iA,B,iB)
    end
  return tens(AAsizes,out)
end

"""
   *(A,B)

Computes the contraction of diagonal tensor `A` with `B` of type `denstens` of `AbstractArray`
"""
function *(X::diagonal,Y::Union{denstens,AbstractArray})
  return dmul(X,Y)
end

"""
   *(A,B)

Computes the contraction of diagonal tensor `B` with `A` of type `denstens` of `AbstractArray`
"""
function *(X::Union{denstens,AbstractArray},Y::diagonal)
  return dmul(X,Y)
end

"""
   *(A,B)

Computes the contraction of array `A` with `B` of type `denstens`
"""
function *(X::Array,Y::denstens)
  return tens(X)*Y
end

"""
   *(A,B)

Computes the contraction of array `B` with `A` of type `denstens`
"""
function *(X::denstens,Y::Array)
  return X*tens(Y)
end

"""
   *(A,B)

Computes the contraction of array `A` with another array `B`
"""
function *(X::Array,Y::Array)
  if eltype(X) <: dualnum || eltype(Y) <: dualnum
#    num_variables = length(A[1,1].grad)
    derivC = Array{dualnum,2}(undef,size(A,1),size(B,2))
    for y = 1:size(B,2)
      for x = 1:size(A,1)
        derivC[x,y] = dualnum()
        for z = 1:size(A,2)
          derivC[x,y] += A[x,z]*B[z,y]
        end
      end
    end
    out = derivC
  else
    out = contract(X,ndims(X),Y,1)#Array(tens(X)*tens(Y))
  end
  return out
end

#=
"""
    Z = X*Y

Computes the equivalent of contract(X,ndims(X),Y,1) between a `diagonal` type tensor and a `denstens`
"""
function *(X::LinearAlgebra.Diagonal{R, Vector{R}},Y::tens{W}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds @simd for x = 1:size(X,1)
      Z[x + tempind] = X[x,x]*Y.T[x + tempind]
    end
  end
  return tens{outType}(Y.size,Z)
end
"""
    Z = X*Y

Computes the equivalent of contract(X,ndims(X),Y,1) between a `tens` type tensor and a `diagonal`
"""
function *(Y::tens{W},X::Diagonal{R}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for x = 1:longdim
    tempind = x - longdim
    @inbounds @simd for y = 1:size(X,1)
      zval = tempind + longdim*y
      Z[zval] = Y.T[zval]*X[y,y]
    end
  end
  return tens{outType}(Y.size,Z)
end
=#



#=

function *(X::LinearAlgebra.UpperTriangular{R, Matrix{R}},Y::tens{W}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds for x = 1:size(X,1)
      val = outType(0)
      @inbounds @simd for k = x:size(X,2)
        val += X[x,k]*Y.T[k + tempind]
      end
      Z[x + tempind] = val
    end
  end
  return tens{outType}(Y.size,Z)
end

function *(Y::tens{W},X::LinearAlgebra.UpperTriangular{R, Matrix{R}}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for x = 1:size(X,1)
    tempind = x - size(X,1)
    @inbounds for y = 1:longdim
      val = outType(0)
      @inbounds @simd for k = 1:x
        val += Y.T[tempind + size(X,1)*k]*X[k,y]
      end
      zval = tempind + size(X,1)*y
      Z[zval] = val
    end
  end
  return tens{outType}(Y.size,Z)
end



function *(X::LinearAlgebra.LowerTriangular{R, Matrix{R}},Y::tens{W}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds for x = 1:size(X,1)
      val = outType(0)
      @inbounds @simd for k = 1:x
        val += X[x,k]*Y.T[k + tempind]
      end
      Z[x + tempind] = val
    end
  end
  return tens{outType}(Y.size,Z)
end

function *(Y::tens{W},X::LinearAlgebra.LowerTriangular{R, Matrix{R}}) where {R <: Number, W <: Number}
  outType = typeof(R(1)*W(1))
  Z = Array{outType,1}(undef,length(Y))
  longdim = cld(length(Y.T),size(X,1))
  for x = 1:size(X,1)
    tempind = x - size(X,1)
    @inbounds for y = 1:longdim
      val = outType(0)
      @inbounds @simd for k = x:size(X,2)
        val += Y.T[tempind + size(X,1)*k]*X[k,y]
      end
      zval = tempind + size(X,1)*y
      Z[zval] = val
    end
  end
  return tens{outType}(Y.size,Z)
end
=#

"""
    Z = X*Y

Computes the equivalent of contract(X,ndims(X),Y,1) between any two `qarray`
"""
function *(X::qarray,Y::qarray)
  if ndims(X) == 1 && ndims(Y) == 2
    X = reshape(X,size(X,1),1)
  end
  return contract(X,ndims(X),Y,1)
end

"""
    vecA,vecB,pairs = contractinds(A,B[,check=false])

Finds the indices to contract over (`vecA` and `vecB`) and which pairs of indices are to be contracted from input `TNobjs` `A` and `B`; `check` allows for a quick check to verify that the indices exist and are paired/not duplicated
"""
function contractinds(A::TNobj,B::TNobj;check::Bool=false)
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

"""
    vecA,vecB,newnames = contractinfo(A,B)

Finds the indices to contract over (`vecA` and `vecB`) and the new names for resulting tensor contraction from input `TNobjs` `A` and `B`
"""
function contractinfo(A::TNobj,B::TNobj)
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
#  newarrows = Array{String,1}(undef,length(newnames))

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
  return vecA,vecB,newnames
end

"""
    C = gencontract(A,vecA,B,vecB)

A generic contraction function to identify which of `contract`, `ccontract`, `contractc`, `ccontractc` to use for star operation of any two `TNobj`

See also: [`contract`](@ref) [`ccontract`](@ref) [`contractc`](@ref) [`ccontractc`](@ref) [`TNobj`](@ref)
"""
function gencontract(A::TNobj,vecA::Array{W,1},B::TNobj,vecB::Array{W,1}) where W <: Integer
  if !A.conj && !B.conj
    out = contract(A.N,vecA,B.N,vecB)
  elseif !A.conj && B.conj
    out = contractc(A.N,vecA,B.N,vecB)
  elseif A.conj && !B.conj
    out = ccontract(A.N,vecA,B.N,vecB)
  elseif A.conj && B.conj
    out = ccontractc(A.N,vecA,B.N,vecB)
  end
  return out
end

#=
function contractinds(A::directedtens,B::directedtens)
  return contractinds(A.T,B.T)
end
=#
"""
    Z = X*Y

Computes the equivalent of contract(X,ndims(X),Y,1) between any two `nametens`
"""
function *(A::nametens,B::nametens)#;fct::Function=contract)
  vecA,vecB,newnames = contractinfo(A,B)
  out = gencontract(A,vecA,B,vecB)
  return nametens(out,newnames)
end

"""
    Z = X*Y

Computes the equivalent of contract(X,ndims(X),Y,1) between any two `directedtens`
"""
function *(A::directedtens,B::directedtens)
  vecA,vecB,newnames = contractinfo(A,B)
  
  out = gencontract(A,vecA,B,vecB)

  newarrows = Array{Int8,1}(undef,ndims(out))
  for w = 1:ndims(out)
    counter = 1
    while counter <= ndims(A.N) && newnames[w] != A.names[counter]
      counter += 1
    end
    if counter > ndims(A.N)
      counter = 1
      while counter <= ndims(B.N) && newnames[w] != B.names[counter]
        counter += 1
      end
      newarrows[w] = B.arrows[counter]
      if B.conj
        newarrows[w] *= -1
      end
    else
      newarrows[w] = A.arrows[counter]
      if A.conj
        newarrows[w] *= -1
      end
    end
  end
  return directedtens(out,newnames,false,newarrows)
end

"""
    Z = X*Y

Computes the equivalent of contract(X,ndims(X),Y,1) between any two `dtens`
"""
function *(A::dtens,B::dtens)
  C = contract(A[0],ndims(A[0]),B[0],1)
  dC = contract(A[0],ndims(A[0]),B[1],1)
  dC = contract(A[1],ndims(A[1]),B[0],1,dC)
  return dtens(C,dC)
end
