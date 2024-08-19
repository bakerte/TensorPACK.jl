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
    rename!(A,inds)

replaces named indices of `TNobj` `A` in-place with indices in `inds`, a vector of length-2 vectors containing strings
"""
function rename!(A::TNobj,inds::Array{Array{W,1},1}) where W <: String
  for a = 1:length(inds)
    condition = true
    b = 0
    while condition && b < length(A.names)
      b += 1
      if A.names[b] == inds[a][1]
        if typeof(inds[a][2]) <: Array
          A.names[b] = inds[a][2][1]
        else
          A.names[b] = inds[a][2]
        end
      end
    end
  end
  nothing
end

"""
    B = rename!(A,inds)

Renames indices in-place of a `TNobj` `A` with array of strings (length 2) `inds`
"""
function rename!(A::TNobj,inds::Array{W,1}) where W <: String
  rename!(A,[inds])
end

"""
    rename!(A,currvar,newvar[,arrows])

replaces a string `currvar` in named indices of `A` with `newvar`; can also set arrows if needed
"""
function rename!(A::TNobj,currvar::String,newvar::String)
  for a = 1:length(A.names)
    loc = findfirst(currvar,A.names[a])
    if !(typeof(loc) <: Nothing)
      first = loc[1] == 1 ? "" : A.names[a][1:loc[1]-1]
      last = loc[end] == length(A.names[a]) ? "" : A.names[a][loc[end]+1]
      newstring = first * newvar * last
      A.names[a] = newstring
    end
  end
  nothing
end
export rename!

"""
    rename(A,inds)

replaces named indices of `TNobj` `A` with indices in `inds`, a vector of length-2 vectors containing strings
"""
function rename(A::TNobj,inds::Array{Array{W,1},1}) where W <: String
  B = copy(A)
  rename!(B,inds)
  return B
end
export rename

"""
    B = rename!(A,inds)

Renames indices of a `TNobj` `A` with array of strings (length 2) `inds`
"""
function rename(A::TNobj,inds::Array{W,1}) where W <: String
  rename(A,[inds])
end