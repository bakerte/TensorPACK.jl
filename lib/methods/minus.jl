





"""
    -(A,B)

Subtracts tensor `A` from `B`

See also: [`sub!`](@ref)
"""
function -(A::TNobj,B::TNobj)
  return sub!(copy(A),B)
end

function -(x::dtens, y::dtens)
  r = x[0] - y[0]
  g = x[1] - y[1]
  return dtens(r, g)
end

###############
# Subtraction #
###############

"""
  -(x,y)

Subtraction of two `dualnum`s
"""
function -(x::dualnum, y::dualnum)
  r = x.val - y.val
  g = x.grad - y.grad
  return dualnum(r, g)
end

"""
  -(x,y)

Subtraction of a `dualnum` and a regular number
"""
function -(x::dualnum, y::Number)
  return dualnum(x.val - y, x.grad)
end
-(x::Number, y::dualnum) = y-x

function -(M::Qtens{W,Q}) where {W <: Number, Q <: Qnum}
  return tensorcombination(M,alpha=(W(-1),))
end

function -(x::dualnum)
  r = -x.val
  g = -x.grad
  return dualnum(r, g)
end

"""
  G = -(A,B)

Subtracts two tensors `A` and `B` (`A`-`B`) with output `G`

See also: [`sub!`](@ref)
"""
function -(A::TensType, B::TensType)
  mA,mB = checkType(A,B)
  return tensorcombination((eltype(A)(1),eltype(B)(-1)),mA,mB)
end


function -(A::Matrix{dualnum}, B::Matrix{Number})
  derivC = Array{dualnum,2}(undef,shape[1],shape[2])
  for i in 1:shape[1]
    for j in 1:shape[2]
      derivC[i, j] = A[i, j] - B[i, j]
    end
  end
  return derivC
end

function -(A::Matrix{Number}, B::Matrix{dualnum})
  shape = size(A)

  derivC = Array{dualnum,2}(undef,shape[1],shape[2])
  for i in 1:shape[1]
    for j in 1:shape[2]
      derivC[i, j] = A[i, j] - B[i, j]
    end
  end
  return derivC
end


function -(A::Vector{dualnum}, B::Vector{<:Number})
  shape = size(A)

  derivC = Array{dualnum,1}(undef,shape[1])
  for i in 1:shape[1]
    derivC[i] = A[i] - B[i]
  end
  return derivC
end

function -(A::Vector{Number}, B::Vector{dualnum})
  shape = size(A)

  derivC = Array{dualnum,1}(undef,shape[1])
  for i in 1:shape[1]
    derivC[i] = A[i] - B[i]
  end
  return derivC
end

function poi(A::Matrix{dualnum}, B::Union{Vector{dualnum}, Vector{Float64}})
  num_variables = length(A[1,1].gradient)
  derivC = Array{dualnum,2}(undef,size(A,1),size(B,2))
  for x = 1:size(A,1)
    derivC[x] = init_dualnum(0, num_variables)
    for z = 1:size(A,2)
      derivC[x] += A[x,z]*B[z]
    end
  end
  return derivC
end

