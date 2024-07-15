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
export addindex!

function addindex(X::nametens,Y::nametens)
  A = copy(X)
  B = copy(Y)
  addindex!(A,B)
  return A,B
end
export addindex

function joinTens(X::nametens,Y::nametens)
  A,B = addindex(X,Y)
  return A*B
end
export joinTens