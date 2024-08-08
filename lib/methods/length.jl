
"""
  G = length(M)

Returns number of elements `G` (integer) total in `denstens` `M`

See: [`denstens`](@ref)
"""
function length(M::denstens)::intType
  return length(M.T)
end

function length(H::bigvec)
  return length(H.V)
end


function length(Qts::TNnetwork)
  return length(Qts.net)
end

function length(Qts::TNobj)
  return length(Qts.N)
end
#=
function length(Qts::nametens)
  return length(Qts.N)
end
=#
### not commented above here...needed?
