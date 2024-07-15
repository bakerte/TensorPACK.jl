import Base.minimum
"""
  minimum(T)

Finds minimum of `tens` or `Qtens` type
"""
function minimum(A::tens)
  return minimum(A.T)
end

import Base.maximum
"""
  maximum(T)

Finds minimum of `tens` or `Qtens` type
"""
function maximum(A::tens)
  return maximum(A.T)
end



import Base.minimum
function minimum(A::TNobj)
  return minimum(A.N)
end

import Base.maximum
function maximum(A::TNobj)
  return maximum(A.N)
end

import Base.minimum
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

import Base.maximum
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
