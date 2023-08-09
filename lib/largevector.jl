#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#


"""
  file_extension

A default file extension can be specified for the large types.

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeEnv`](@ref)
"""
const file_extension = ".tenpack"

abstract type bigvec end
export bigvec

"""
  `largeenvironment` 
  
Construct this container with `largeEnv`. struct to hold `largeEnv` tensors

# Fields:
+ `V::Array{String,1}`: filenames where the tensors are stored on disk
+ `type::DataType`: DataType of the stored tensors

See also: [`largeEnv`](@ref)
"""
struct largevector{W} <: bigvec where W <: DataType
  V::Array{String,1}
  type::W
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

function setindex!(H::bigvec,A::TensType,i::intType;ext::String=file_extension)
  tensor2disc(H.V[i],A,ext=ext)
  nothing
end

function length(H::bigvec)
  return length(H.V)
end

function eltype(op::bigvec)
  return op.type
end

"""
  G = loadMPO(Ns[,label="mpo_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If MPO tensors are stored on hard disk, then they can be retrieved by using `loadMPO`
"""
function loadlargevector(Ns::Integer;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnum = 1
  for i = 1:Ns
    name = names[i]
    A = tensorfromdisc(name,ext=ext)
    lastnum *= eltype(A)(1)
  end
  thistype = typeof(lastnum)
  return bigvec(names,thistype)
end
export loadMPO

"""
  G = copy(names,X[,ext=".dmrjulia",copyext=ext])

Copies tensors from `largeEnv` input `X` to a new tensor with a vector of strings `names` representing the new filenames
"""
function copy(names::Array{String,1},X::bigvec;ext::String=file_extension,copyext::String=ext)
  newObj = bigvec([names[(i-1) % length(names) + 1] for i = 1:length(X)],X.type)
  for i = 1:length(X)
    Y = tensorfromdisc(X.V[i],ext=ext)
    tensor2disc(newObj.V[i],Y,ext=copyext)
  end
  return newObj
end

function copy(names::String,X::bigvec;ext::String=file_extension,copyext::String=ext)
  return copy(names .* X.V,X,ext=ext,copyext=copyext)
end