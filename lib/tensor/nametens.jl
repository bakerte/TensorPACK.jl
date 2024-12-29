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
    nametens(Qt,namez)

constructor for named tensor from a tensor `Qt` and vector of index names `namez`
"""
function nametens(Qt::Union{TensType,diagonal},namez::Union{Array{String,1},Memory{String}};regTens::Bool=false,conj::Bool=false)#,arrows::Array{Bool,1},conj::Bool;regTens::Bool=false)
  if (regTens && typeof(Qt) <: AbstractArray) || typeof(Qt) <: qarray
    newQt = Qt
  elseif typeof(Qt) <: diagonal
    newQt = Qt
  else
    newQt = tens(Qt)
  end

  return nametens{typeof(newQt)}(newQt,namez,conj)
end

"""
    nametens(Qt)

Converts any `directedtens` to a `nametens` by recycling the fields in `directedtens`
"""
function nametens(Qt::directedtens)
  return nametens(Qt.N,Qt.names,Qt.conj)
end

"""
    nametens(Qt,string)

constructor for named tensor from a tensor `Qt` and vector of index names `string` where an integer is added onto `string` to make the full name of the index
"""
function nametens(Qt::T,namez::String;regTens::Bool=false,conj::Bool=false) where T <: TensType
  return nametens(Qt,[namez*"$i" for i = 1:basedims(Qt)],regTens=regTens,conj=conj)
end



#=
function nametens(Qt::TensType,namez::Array{String,1};regTens::Bool=false)
  return nametens(Qt,namez,Bool[],false,regTens=regTens)
end

function nametens(Qt::TensType,namez::Array{String,1},arrows::Array{Bool,1};regTens::Bool=false)
  return nametens(Qt,namez,arrows,false,regTens=regTens)
end
=#
