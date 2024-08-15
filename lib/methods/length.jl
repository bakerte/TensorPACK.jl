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
  G = length(M)

Returns number of elements `G` (integer) total in `denstens` `M`

See: [`denstens`](@ref)
"""
function length(M::denstens)::intType
  return length(M.T)
end

"""
  G = length(M)

Returns number of elements `G` (integer) total in `bigvec` `M`

See: [`bigvec`](@ref)
"""
function length(H::bigvec)
  return length(H.V)
end

"""
  G = length(M)

Returns number of elements `G` (integer) total in `TNnetwork` `M`

See: [`TNnetwork`](@ref)
"""
function length(Qts::TNnetwork)
  return length(Qts.net)
end

"""
  G = length(M)

Returns number of elements `G` (integer) total in `TNobj` `M`

See: [`TNobj`](@ref)
"""
function length(Qts::TNobj)
  return length(Qts.N)
end
#=
function length(Qts::nametens)
  return length(Qts.N)
end
=#
### not commented above here...needed?
