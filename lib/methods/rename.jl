"""
    rename!(A,inds)

replaces named indices in `A` with indices in `inds`; either format [string,[string,arrow]] or [string,string] or [string,[string]] is accepted for `inds`
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

function rename(A::TNobj,inds::Array{Array{W,1},1}) where W <: String
  B = copy(A)
  rename!(B,inds)
  return B
end
export rename