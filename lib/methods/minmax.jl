

"""
  minimum(T)

Finds minimum of `tens` or `Qtens` type
"""
function minimum(A::tens)
  return minimum(A.T)
end

"""
  maximum(T)

Finds minimum of `tens` or `Qtens` type
"""
function maximum(A::tens)
  return maximum(A.T)
end



function minimum(A::TNobj)
  return minimum(A.N)
end

function maximum(A::TNobj)
  return maximum(A.N)
end

"""
  minimum(T)

Finds minimum of `tens` or `Qtens` type
"""
function minimum(A::qarray)
  val = minimum(A.T[1])
  for q = 2:length(A.T)
    val = min(val,minimum(A.T[q]))
  end
  return val
end

"""
  maximum(T)

Finds minimum of `tens` or `Qtens` type
"""
function maximum(A::qarray)
  val = maximum(A.T[1])
  for q = 2:length(A.T)
    val = max(val,maximum(A.T[q]))
  end
  return val
end
