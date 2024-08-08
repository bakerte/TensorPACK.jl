


function *(val::Number,A::dtens)
  B = copy(A)
  B[0] *= val
  return B
end


##################
# Multiplication #
##################

"""
  *(A,B)

Multiply two `dualnum`s
"""
function *(x::dualnum, y::dualnum)
  r = x.val * y.val
  g = y.val * x.grad + x.val * y.grad
  return dualnum(r, g)
end

"""
  *(A,B)

Multiply a `dualnum` with a number
"""
function *(x::dualnum, y::Number)
  r = x.val * y
  g = x.grad * y
  return dualnum(r, g)
end
*(y::Number, x::dualnum) = x*y

function *(A::dtens,B::dtens)
  C = A[0]*B[0]
  dC = A[0]*B[1] 
  dC += A[1]*B[0]
  return dtens(C,dC)
end

function *(A::Matrix{dualnum},B::dualnum)
  C = Array{dualnum,2}(undef,size(A,1),size(A,2))
  for y = 1:size(A,2)
    for x = 1:size(A,1)
      C[x,y] = A[x,y]*B
    end
  end
  return C
end

function *(B::dualnum,A::Matrix{dualnum})
  return A*B
end

function *(A::Matrix{<:Number},B::dualnum)
  C = Array{dualnum,2}(undef,size(A,1),size(A,2))
  for y = 1:size(A,2)
    for x = 1:size(A,1)
      C[x,y] = A[x,y]*B
    end
  end
  return C
end

function *(B::dualnum,A::Matrix{<:Number})
  return A*B
end

# For some reason this will only work with Float64 in the vector and not Number
function *(A::Matrix{dualnum}, B::Vector{Float64})
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

function *(A::Matrix{dualnum},B::Matrix{dualnum})
  num_variables = length(A[1,1].gradient)
  derivC = Array{dualnum,2}(undef,size(A,1),size(B,2))
  for y = 1:size(B,2)
    for x = 1:size(A,1)
      derivC[x,y] = init_dualnum(0, num_variables)
      for z = 1:size(A,2)
        derivC[x,y] += A[x,z]*B[z,y]
      end
    end
  end
  return derivC
end


"""
    *(A,num)

Gives the multiplication of named tensor `A` by number `num` (commutative)

See also: [`mult!`](@ref)
"""
function *(A::TNobj,num::Number)
  return mult!(copy(A),num)
end

function *(num::Number,A::TNobj)
  return A*num
end

"""
    *(a,b)

concatenates string `a` with integer `b` after converting integer to a string
"""
function *(a::String,b::Integer)
  return a*string(b)
end

"""
  G = *(A,num)

Mutiplies a tensor `A` by a number `num` with output `G`

See also: [`mult!`](@ref)
"""
function *(num::Number, M::TensType)
  if !(typeof(num) <: eltype(M))
    num = convert(eltype(M),num)
  end
  return tensorcombination(M,alpha=(num,))
end

"""
  G = *(num,A)

Mutiplies a tensor `A` by a number `num` with output `G`. Ensures commutativity of the operation

See also: [`mult!`](@ref)
"""
function *(M::TensType, num::Number)
  return num * M
end
