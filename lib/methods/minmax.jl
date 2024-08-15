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
  minimum(T)

Finds minimum of `denstens` type

See also: [`denstens`](@ref)
"""
function minimum(A::denstens)
  return minimum(A.T)
end

"""
  maximum(T)

Finds maximum of `denstens` type

See also: [`denstens`](@ref)
"""
function maximum(A::denstens)
  return maximum(A.T)
end


"""
  minimum(T)

Finds minimum of `TNobj` type

See also: [`TNobj`](@ref)
"""
function minimum(A::TNobj)
  return minimum(A.N)
end

"""
  maximum(T)

Finds maximum of `TNobj` type

See also: [`TNobj`](@ref)
"""
function maximum(A::TNobj)
  return maximum(A.N)
end

"""
  minimum(T)

Finds minimum of `qarray` type

See also: [`qarray`](@ref)
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

Finds maximum of `qarray` type

See also: [`qarray`](@ref)
"""
function maximum(A::qarray)
  val = maximum(A.T[1])
  for q = 2:length(A.T)
    val = max(val,maximum(A.T[q]))
  end
  return val
end
