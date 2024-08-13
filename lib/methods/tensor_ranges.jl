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
  G = get_denseranges(x,y)

Generates a vector `G` for a tensor of size `x` and selected dimensions `y` containing only `genColType` values. Function converts all to integers or vectors (no tuples for compatibility with Julia arrays)

See: [`genColType`])(@ref) [`getindex!`](@ref)
"""
function get_denseranges(sizes::NTuple{G,intType},a::genColType...) where G
  unitranges = Array{genColType,1}(undef,G)
  @inbounds for i = 1:G
    if typeof(a[i]) <: Colon
      unitranges[i] = 1:sizes[i]
    elseif typeof(a[i]) <: Array{intType,1} || typeof(a[i]) <: Tuple{intType,Vararg{intType}}
      unitranges[i] = [a[i][w] for w = 1:length(a[i])]
    elseif typeof(a[i]) <: UnitRange{intType}
      unitranges[i] = a[i] #UnitRange(a[i][1],a[i][end])
    elseif typeof(a[i]) <: StepRange{intType}
      unitranges[i] = a[i] #UnitRange(a[i][1],a[i][end])
    elseif typeof(a[i]) <: Integer
      unitranges[i] = a[i]
    end
  end
  return unitranges
end

"""
  G = get_denseranges_sizes(v,x,y)

Generates a vector `G` for a tensor of size `x` and selected dimensions `y` containing only `genColType` values. Function converts all to integers or vectors (no tuples for compatibility with Julia arrays)

Also returns a vector of sizes of the new tensor with known number of non-integer dimensions `v`

See: [`genColType`])(@ref) [`getindex!`](@ref)
"""
function get_denseranges_sizes(sizes::NTuple{G,intType},a::genColType...) where G
  unitranges = Array{genColType,1}(undef,G)
  unitsizes = Array{intType,1}(undef,G)
  counter = 0
  @inbounds for i = 1:G
    if typeof(a[i]) <: Colon
      unitranges[i] = 1:sizes[i]
      counter += 1
      unitsizes[counter] = sizes[i]
    elseif typeof(a[i]) <: Array{intType,1} || typeof(a[i]) <: Tuple{intType,Vararg{intType}}
      unitranges[i] = [a[i][w] for w = 1:length(a[i])]
      counter += 1
      unitsizes[counter] = length(a[i])
    elseif typeof(a[i]) <: UnitRange{intType}
      unitranges[i] = a[i] #UnitRange(a[i][1],a[i][end])
      counter += 1
      unitsizes[counter] = length(a[i])
    elseif typeof(a[i]) <: StepRange{intType}
      unitranges[i] = a[i] #UnitRange(a[i][1],a[i][end])
      counter += 1
      unitsizes[counter] = length(a[i])
    elseif typeof(a[i]) <: Integer
      unitranges[i] = a[i]
      counter += 1
      unitsizes[counter] = 1
    end
  end
  return unitranges,unitsizes
end