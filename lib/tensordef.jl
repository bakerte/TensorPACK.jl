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
    tens{Z}

Regular tensor type; defined as `tens{W}` for W <: Number

# Fields:
+ `size::NTuple{N,intType}`: size of base tensor (unreshaped)
+ `T::Array{W,1}`: tensor reshaped into a vector
"""
mutable struct tens{W <: Number} <: denstens
  size::NTuple{G,intType} where G #Tuple{intType,Vararg{intType}} #Array{intType,1} #Tuple #Tuple{intType,Vararg{intType}}
  T::Array{W,1}
end

"""
    diagonal{W}

Diagonal tensor type; defined as `diagonal{W}` for W <: Number

# Fields:
+ `T::Array{W,1}`: Diagonal of a square matrix given as a vector

See also: [`Diagonal`](@ref)
"""
struct diagonal{W <: Number} <: denstens
  T::Array{W,1}
end
export diagonal

"""
    Diagonal

Alias for `diagonal`

See also: [`diagonal`](@ref)
"""
Diagonal = diagonal
export Diagonal

"""
    diagm

Alias for `diagonal`

See also: [`diagonal`](@ref)
"""
diagm = diagonal
export diagm


"""
    TNnetwork

Abstract type for a network of tensors
"""
abstract type TNnetwork end
export TNnetwork

"""
    nametens{W,B}

named tensor with tensor of type `W` and type of names `B`

# Fields:
+ `N::W`: Tensor stored
+ `names::Array{String,1}`: names of all indices
"""
mutable struct nametens{W} <: TNobj where W <: TensType
  N::W
  names::Array{String,1}
  conj::Bool
end

const nameTens = nametens

"""
    directedtens{W,B}

named tensor with named tensor of type `W` and vector of short integers (Int8) `B`. Creates a directed graph for use in MERA networks

# Fields:
+ `N::W`: Tensor stored
+ `names::Array{String,1}`: names of all indices
+ `conj::Bool`: whether the tensor is to be conjuated when contracted
+ `arrows::Array{Int8,1}`: arrow convention for each index

See also: [`nametens`](@ref)
"""
mutable struct directedtens{W} <: TNobj where W <: TensType
  N::W
  names::Array{String,1}
  conj::Bool
  arrows::Array{Int8,1}
end

"""
    network{N,W}

Generates a network of TNobjs that stores more than one named tensor

# Fields:
+ `net::NTuple{N,W}`: A network of named tensors
"""
mutable struct network{W} <: TNnetwork where W  <: TNobj
  net::Array{W,1}
  level::intType
end


"""
    dtens

storage type for automatic differentiation
"""
mutable struct dtens#={W,R}=# #<: denstens #where {W <: TensType, R <: TensType}
  d::Tuple#{W,R} #zero is one!
end


"""
    dualnum{W}

stores a value (`val`) and a gradient (`grad`) for each number. 
"""
struct dualnum{W <: Number} #<: Number
  val::W
  grad::W
end


"""
    Qtens{T,Q}

Qtensor; defined as `Qtens{T,Q}` for T <: Number and Q <: Qnum

# Fields:
+ `size::Array{intType,1}`: size of base tensor (unreshaped)
+ `T::Array{Z,1}`: Array containing non-zero blocks' values of the tensor
+ `ind::Array{intType,1}`: indices of the stored values
+ `QnumMat::Array{Array{Q,1},1}`: quantum numbers on each index
+ `QnumSum::Array{Array{Q,1},1}`: summary of quantum numbers on each index
+ `flux::Q`: total quantum number flux on tensor

See also: [`Qnum`](@ref) [`checkflux`](@ref)
"""
mutable struct Qtens{W <: Number,Q <: Qnum} <: qarray
  size::Array{Array{intType,1},1} #the size of the tensor if it were represented densely
  #^This is an array since it can change on reshape
  T::Union{Array{Array{W,2},1},Array{Diagonal{W},1}}
  ind::Array{NTuple{2,Array{intType,2}},1}
  currblock::NTuple{2,Array{intType,1}}
  Qblocksum::Array{NTuple{2,Q},1}
  QnumMat::Array{Array{intType,1},1} #quantum numbers on each index
  QnumSum::Array{Array{Q,1},1} #summary of indices on each index
  flux::Q #flux = sum of other indices
end
export Qtens

"""
    currblockTypes

some useful storage types `Union{NTuple{2,Array{intType,1}},Array{Array{intType,1},1}}`
"""
const currblockTypes = Union{NTuple{2,Array{intType,1}},Array{Array{intType,1},1}}

"""
    nameTensType

Any tensor with a .names field (`Union{nametens,directedtens}`)
"""
const nameTensType = Union{nametens,directedtens}
export nameTensType

"""
    nametenstype

Any tensor with a .names field (`Union{nametens,directedtens}`)
"""
const nametenstype = nameTensType
export nametenstype
