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

#=
"""
  directedtens(Qt,arrows)

constructor for named tensor `Qt` and vector of directed arrows `arrows`
"""
function directedtens(Qt::nametens{W},arrows::Array{B,1},conj::Bool) where {W <: TensType, B <: Integer}
  return directedtens(Qt,arrows,conj)
end
=#
"""
    A = directedtens(Qt)

constructor for named tensor `Qt` for a `directedtens`
"""
function directedtens(Qt::nametens) #where W <: TensType
  arrows = Int8[0 for i = 1:ndims(Qt)]
  return directedtens(Qt,arrows)
end

function directedtens(Qt::nametens,arrows::Array{intType,1};regTens::Bool=false,conj::Bool=false)
  if eltype(arrows) != Int8
    arrows = convert(Array{Int8,1},arrows)
  end
  return directedtens(Qt,arrows,conj)
end

"""
    A = directedtens(Qt,namez, arrows)

constructor for named tensor from a tensor `Qt` and vector of index names `namez`. `arrows` is a container with -1, 0, or 1 in all entries. -1 means this index goes to one level below, 0 maintains the level, and 1 increments to a tensor on the next level
"""
function directedtens(Qt::TensType,namez::Array{String,1},arrows::Array{intType,1};regTens::Bool=false,conj::Bool=false)
  if eltype(arrows) != Int8
    arrows = convert(Array{Int8,1},arrows)
  end
  return directedtens(Qt,namez,conj,arrows)
end

"""
    A = directedtens(Qt,string)

constructor for named tensor from a tensor `Qt` and vector of index names `string` where an integer is added onto `string` to make the full name of the index
"""
function directedtens(Qt::T,namez::String,arrows::Array{B,1};regTens::Bool=false,conj::Bool=false) where {T <: TensType, B <: Integer}
  return directedtens(Qt,namez,conj,arrows)
end
export directedtens
#=
function ndims(Qt::TNobj)
  return length(Qt.N)
end
=#