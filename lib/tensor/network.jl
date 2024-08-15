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
    A = network(Qts[,level=0])

constructor to generates a network of `TNobj`s that stores a vector of named tensors `Qts`
"""
function network(Qts::Tuple;level::Integer=0)
  type = eltype(Qts)
  altQts = Array{type,1}(undef,length(Qts))  
  for w = 1:length(Qts)
    altQts[w] = Qts[w]
  end
  return network(altQts,level)
end

"""
    A = network(Qts[,level=0])

constructor to generates a network of `Array`s that stores a vector of named tensors `Qts`
"""
function network(Qts::W;level::Integer=0) where W <: Array
  return network(Qts,level)
end

"""
    A = network(Qts...)

converts named tensor to a network with a single tensor element
"""
function network(Qts::W...) where W  <: TNobj
  return network{W}(W[Qts[i] for i = 1:length(Qts)],0)
end

"""
    A = network(Qts...)

converts inputs of `densTensType` to a network with a single tensor element
"""
function network(Qts::W...) where W  <: densTensType
  return network{W}(W[Qts[i] for i = 1:length(Qts)],0)
end

"""
    A = network(Qt,i)

converts named tensor `Qt` to a network with `i` copied elements not shallow copied
"""
function network(Qts::W,n::Integer) where W  <: TNobj
return network{W}(W[copy(Qts) for i = 1:n],0)
end
