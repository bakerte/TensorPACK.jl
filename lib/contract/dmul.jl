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
    Z = dmul!(X,Y)

Contraction of a `dentens` `X` onto a `diagonal` type `Y`; output `Z` replaces `X`
"""
function dmul!(Y::tens{R},X::Diagonal{W}) where {R <: Number, W <: Number}
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:size(X,1)
    zval = longdim*(y-1)
    val = X[y,y]
    @inbounds @simd for x = 1:longdim
      Y.T[x + zval] *= val
    end
  end
  return Y
end

"""
    Z = dmul!(X,Y)

Contraction of a `Array` `X` onto a `diagonal` type `Y`; output `Z` replaces `X`
"""
function dmul!(Y::Array{R,N},X::Diagonal{W}) where {R <: Number, N, W <: Number}
  longdim = cld(length(Y),size(X,1))
  for y = 1:size(X,1)
    zval = longdim*(y-1)
    val = X[y]
    @inbounds @simd for x = 1:longdim
      Y[x + zval] *= val
    end
  end
  return Y
end

"""
    Z = dmul!(X,Y)

Contraction of a `Number` `X` onto a `denstens` type `Y`; output `Z` replaces `Y`
"""
function dmul!(X::R,Y::tens{W}) where {R <: Number, W <: Number}
  return tensorcombination!((X,),Y)
end

"""
    Z = dmul!(X,Y)

Contraction of a `Number` `Y` onto a `denstens` type `X`; output `Z` replaces `X`
"""
function dmul!(Y::tens{W},X::R) where {R <: Number, W <: Number}
  return dmul!(X,Y)
end

"""
    Z = dmul!(X,Y)

Contraction of a `diagonal` `X` onto a `denstens` type `Y`; output `Z` replaces `Y`
"""
function dmul!(X::Diagonal{R},Y::tens{W}) where {R <: Number, W <: Number}
  longdim = cld(length(Y.T),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds @simd for x = 1:size(X,1)
      Y.T[x + tempind] *= X[x]
    end
  end
  return Y
end

"""
    Z = dmul!(X,Y)

Contraction of a `diagonal` `X` onto a `Array` type `Y`; output `Z` replaces `Y`
"""
function dmul!(X::Diagonal{R},Y::Array{W,N}) where {R <: Number, W <: Number, N}
  longdim = cld(length(Y),size(X,1))
  for y = 1:longdim
    tempind = size(X,1)*(y-1)
    @inbounds @simd for x = 1:size(X,1)
      Y[x + tempind] *= X[x]
    end
  end
  return Y
end

"""
    Z = dmul(X,Y)

Contraction of a `diagonal` `X` onto a `densTensType` type `Y`
"""
function dmul(X::Diagonal{R},Y::densTensType) where {R <: Number}
  return dmul!(X,copy(Y))
end

"""
    Z = dmul(X,Y)

Contraction of a `densTensType` `X` onto a `diagonal` type `Y`
"""
function dmul(X::densTensType,Y::Diagonal{R}) where {R <: Number}
  out = dmul!(copy(X),Y)
  return out
end
export dmul!

"""
    Z = dmul!(X,Y)

Contraction in-place of a `qarray` `X` onto a `qarray` type `Y`; one of these functions being a `diagonal` type will be cheaper to evaluate
"""
function dmul!(X::Qtens{R,Q},Y::Qtens{W,Q}) where {R <: Number, W <: Number, Q <: Qnum}
  return maincontractor(false,false,X,(ndims(X),),Y,(1,),inplace=true)
end

"""
    Z = dmul(X,Y)

Contraction of a `qarray` `X` onto a `qarray` type `Y`; one of these functions being a `diagonal` type will be cheaper to evaluate
"""
function dmul(X::Qtens{R,Q},Y::Qtens{W,Q}) where {R <: Number, W <: Number, Q <: Qnum}
  return maincontractor(false,false,X,(ndims(X),),Y,(1,),inplace=false)
end
