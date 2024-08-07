"""
  swapname!(A,labels)

Finds elements in `labels` (must be length 2) and interchanges the name. For example, `swapname!(A,["a","c"])` will find the label `"a"` and `"c"` and interchange them. A chain of swaps can also be requested (i.e., `swapname!(A,[["a","c"],["b","d"]])`)

Works as a pseudo-permute in many cases. Will not permute if the names are not found.

See also: [`swapnames!`](@ref)
"""
function swapname!(A::TNobj,inds::Array{Array{W,1},1}) where W <: String
  for c = 1:length(inds)
    x = 1
    while x < length(A.names) && A.names[x] != inds[c][1]
      x += 1
    end
    y = 1
    while y < length(A.names) && A.names[y] != inds[c][2]
      y += 1
    end
    if inds[c] == [A.names[x],A.names[y]]
      A.names[x],A.names[y] = A.names[y],A.names[x]
    end
  end
  nothing
end

function swapname!(A::TNobj,inds::Array{W,1}) where W <: String
  swapname!(A,[inds])
end
export swapname!

"""
  swapnames!(A,labels)

Finds elements in `labels` (must be length 2) and interchanges the name. For example, `swapname!(A,["a","c"])` will find the label `"a"` and `"c"` and interchange them. A chain of swaps can also be requested (i.e., `swapname!(A,[["a","c"],["b","d"]])`)

Works as a pseudo-permute in many cases. Will not permute if the names are not found.

See also: [`swapname!`](@ref)
"""
function swapnames!(A::nametens,inds::Array{Array{W,1},1}) where W <: String
  swapname!(A,inds)
end

function swapnames!(A::nametens,inds::Array{W,1}) where W <: String
  swapname!(A,[inds])
end
export swapnames!