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
    addindex!(X,Y)

adds an index "extra_ones" to `TNobj`s `X` and `Y`

See also: [`joinTens`](@ref)
"""
function addindex!(X::TNobj,Y::TNobj)
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

"""
    Z = addindex(X,Y)

adds an index "extra_ones" to `TNobj`s `X` and `Y`

See also: [`joinTens`](@ref)
"""
function addindex(X::nametens,Y::nametens)
  A = copy(X)
  B = copy(Y)
  addindex!(A,B)
  return A,B
end

"""
    addindex!(X,Y)

adds an index "extra_ones" to `TNobj`s `X` and `Y`

See also: [`joinTens`](@ref)
"""
function joinTens(X::nametens,Y::nametens)
  A,B = addindex(X,Y)
  return A*B
end