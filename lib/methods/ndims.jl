
"""
  G = ndims(A)

Number of dimensions (rank) `G` of a `denstens` (identical usage to `Array` `ndims` call)

See also: [`denstens`](@ref) [`Array`](@ref)
"""
function ndims(A::denstens)
  return length(size(A))
end

"""
    ndims(A)

Returns the number of indices of named tensor `A`
"""
function ndims(A::TNobj)
  return length(A.names)
end


#=
function ndims(A::directedtens)
  return length(A.arrows)
end
=#


function ndims(A::dtens)
  return ndims(A[0])
end


"""
    ndims(A)

number of dimensions of a Qtensor (identical usage to dense `size` call)
"""
function ndims(A::qarray)
  return length(A.size)
end


"""
  basedims(A)

Finds the number of dimensions of the base quantum number tensor `A` no matter how it was reshaped
"""
function basedims(Qtensor::qarray)
  return length(Qtensor.QnumMat)
end

function basedims(A::denstens)
  return ndims(A)
end
