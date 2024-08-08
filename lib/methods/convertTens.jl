function convertTens(T::DataType,A::TensType)
  if typeof(A) <: denstens
    B = tens(T,A)
  elseif typeof(A) <: qarray
    B = Qtens(T,A)
  elseif typeof(A) <: nametens
    B = nametens(convertTens(T,A.N))
  elseif typeof(A) <: directedtens
    B = directedtens(convertTens(T,A.N))
  elseif typeof(A) <: dtens
    B = dtens(convertTens(T,A[0]),convertTens(T,A[1]))
  elseif typeof(A) <: Array
    B = Array(T,A)
  end
  return B
end