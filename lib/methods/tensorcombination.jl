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
  G = tensorcombination(M...[,alpha=(1,1,1...),fct=*])

Performs a linear combination of the input `denstens` `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination(M::tens{W}...;alpha::NTuple{R,W}=ntuple(i->W(1),length(M)),fct::Function=*) where {W <: Number, R}
  input_tup = ntuple(g->M[g].T,length(M))
  output_vector = tensorcombination(input_tup...,alpha=alpha,fct=fct)
  return tens{W}(M[1].size,output_vector)
end

"""
  G = tensorcombination(M...[,alpha=(1,1,1...),fct=*])

Performs a linear combination of the input `Diagonal` `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination(M::Diagonal{W}...;alpha::NTuple{R,W}=ntuple(i->W(1),length(M)),fct::Function=*) where {W <: Number, R}
  input_tup = ntuple(g->M[g].T,length(M))
  output_vector = tensorcombination(input_tup...,alpha=alpha,fct=fct)
  return Diagonal{W}(output_vector)
end

"""
  G = tensorcombination(M...[,alpha=(1,1,1...),fct=*])

Performs a linear combination of the input Array `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination(M::Array{W,N}...;alpha::NTuple{R,W}=ntuple(i->W(1),length(M)),fct::Function=*) where {W <: Number, N, R}
  nterms = min(length(M),length(alpha))
  newTensor = Array{W,N}(undef,size(M[1])...)
  for i = 1:length(M[1])
    out = W(0)
    @inbounds @simd for k = 1:nterms
      out += fct(M[k][i],alpha[k])
    end
    newTensor[i] = out
  end
  return newTensor
end

"""
  G = tensorcombination(alpha,M...[,fct=*])

Performs a linear combination of the input `denstens` `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination(alpha::NTuple{G,W},M::tens{W}...;fct::Function=*) where {W <: Number, G}
  return tensorcombination(M...,alpha=alpha,fct=fct)
end

"""
  G = tensorcombination(alpha,M...[,fct=*])

Performs a linear combination of the input `Diagonal` `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination(alpha::NTuple{G,W},M::Array{W,G}...;fct::Function=*) where {W <: Number, G}
  return tensorcombination(M...,alpha=alpha,fct=fct)
end

"""
  G = tensorcombination(alpha,M...[,fct=*])

Performs a linear combination of the input Array `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination(alpha::NTuple{G,W},M::Diagonal{W}...;fct::Function=*) where {W <: Number, G}
  return tensorcombination(M...,alpha=alpha,fct=fct)
end

"""
  G = tensorcombination!(M...[,alpha=,fct=])

Same as `tensorcombination` but alters the contents of the first input tensor in `M` and still outputs tensor `G`

See also: [`tensorcombination`](@ref)
"""
function (tensorcombination!(M::P...;alpha::NTuple{R,W}=ntuple(i->W(1),length(M)),fct::Function=*) where P <: Union{tens{W},Array{W,G},Diagonal{W}}) where {W <: Number, R, G}
  nterms = min(length(M),length(alpha))
  for i = 1:length(M[1])
    M[1][i] = fct(M[1][i],alpha[1])
#    out = W(0)
    @inbounds @simd for k = 2:nterms
      M[1][i] += fct(M[k][i],alpha[k])
    end
#    M[1][i] = out
  end
  return M[1]
end
#=
function tensorcombination!(M::tens{W}...;alpha::NTuple{R,W}=ntuple(i->W(1),length(M)),fct::Function=*) where {W <: Number, R}
  nterms = min(length(M),length(alpha))
  for i = 1:length(M[1])
    M[1][i] = fct(M[1][i],alpha[1])
#    out = W(0)
    @inbounds @simd for k = 2:nterms
      M[1][i] += fct(M[k][i],alpha[k])
    end
#    M[1][i] = out
  end
  return M[1]
end

function tensorcombination!(M::Array{W,G}...;alpha::NTuple{R,W}=ntuple(i->W(1),length(M)),fct::Function=*) where {W <: Number, G, R}
  nterms = min(length(M),length(alpha))
  for i = 1:length(M[1])
    M[1][i] = fct(M[1][i],alpha[1])
#    out = W(0)
    @inbounds @simd for k = 2:nterms
      M[1][i] += fct(M[k][i],alpha[k])
    end
#    M[1][i] = out
  end
  return M[1]
end

function tensorcombination!(M::LinearAlgebra.Diagonal{W,Vector{W}}...;alpha::NTuple{R,W}=ntuple(i->W(1),length(M)),fct::Function=*) where {W <: Number, R}
  nterms = min(length(M),length(alpha))
  for i = 1:size(M[1],1)
    M[1][i,i] = fct(M[1][i,i],alpha[1])
#    out = W(0)
    @inbounds @simd for k = 2:nterms
      M[1][i,i] += fct(M[k][i,i],alpha[k])
    end
#    M[1][i] = out
  end
  return M[1]
end

function tensorcombination!(alpha::NTuple{R,W},M::tens{W}...;fct::Function=*) where {R, W <: Number}
  return tensorcombination!(M...,alpha=alpha,fct=fct)
end

function tensorcombination!(alpha::NTuple{R,W},M::Array{W,G}...;fct::Function=*) where {R, G, W <: Number}
  return tensorcombination!(M...,alpha=alpha,fct=fct)
end

function tensorcombination!(alpha::NTuple{R,W},M::LinearAlgebra.Diagonal{W,Vector{W}}...;fct::Function=*) where {R, W <: Number}
  return tensorcombination!(M...,alpha=alpha,fct=fct)
end
=#

"""
  G = tensorcombination!(alpha,M...[,fct=*])

Performs a linear combination of the input `denstens`, Array, or `Diagonal` `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

This version converts input `alpha` to the same type as the element type of `M`

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function (tensorcombination!(alpha::NTuple{R,S},M::P...;fct::Function=*) where P <: Union{tens{W},Array{W,G},Diagonal{W}}) where {R, G, W <: Number, S <: Number}
  if W != S
    alpha = ntuple(w->convert(W,alpha[w]),length(alpha))
  end
  out = tensorcombination!(M...,alpha=alpha,fct=fct)
  return out
end



"""
  G = tensorcombination!(M...[,alpha=(1,1,1...),fct=*])

Performs a linear combination of the input `qarray` `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination!(M::Qtens{W,Q}...;alpha::NTuple{N,W}=ntuple(i->eltype(M[1])(1),length(M)),fct::Function=*) where {Q <: Qnum, W <: Number, N}
  if length(M) == 1
    A = tensorcombination!(M[1],alpha=(alpha[1],))
  else
    A = tensorcombination!(M[1],M[2],alpha=alpha[1:2])
    nterms = min(length(M),length(alpha))
    @inbounds for i = 3:nterms
      A = tensorcombination!(A,M[i],alpha=(W(1),alpha[i]),fct=fct)
    end
  end
  return A
end

"""
  G = tensorcombination(M[,alpha=(1,),fct=*])

Performs a linear combination of the input `qarray` `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination!(M::Qtens{W,Q};alpha::NTuple{N,W}=(W(1),),fct::Function=*) where {Q <: Qnum, W <: Number, N}
  if !isapprox(alpha[1],W(1))
    @inbounds for q = 1:length(M.T)
      thisM = M.T[q]
      for y = 1:size(thisM,2)
        @inbounds @simd for x = 1:size(thisM,1)
          thisM[x,y] = fct(thisM[x,y],alpha[1])
        end
      end
      M.T[q] = thisM
    end
  end
  return M
end

"""
  G = tensorcombination!(alpha,M...[,fct=*])

Performs a linear combination of the input `qarray` `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination!(alpha::NTuple{N,S},M::Qtens{W,Q}...;fct::Function=*) where {Q <: Qnum, W <: Number, N, S <: Number}
  return tensorcombination!(M...,alpha=alpha,fct=fct)
end

"""
    findvals(Lrows,newrows)

Finds values of a set of rows `Lrows` (or columns) from a an input vector of new row numbers `newrows`
"""
function findvals(Lrows::Array{intType,1},newrows::Array{intType,1})
  subsetL = Array{intType,1}(undef,length(Lrows))
  g = 1
  for c = 1:length(Lrows)
    val = Lrows[c]
    while newrows[g] < val
      g += 1
    end

    while newrows[g] > val
      g -= 1
    end
    subsetL[c] = g
  end
  return subsetL
end


"""
  G = tensorcombination!(A,B[,alpha=(1,1),fct=*])

Performs a linear combination of the input `qarray` `A` with `B` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination!(A::Qtens{W,Q},QtensB::Qtens{W,Q};alpha::NTuple{2,W}=(W(1),W(1)),fct::Function=*) where {Q <: Qnum, W <: Number}

  B = changeblock(QtensB,A.currblock)

  commoninds = matchblocks((false,false),A,B,ind=(1,2),matchQN=A.flux)

  @inbounds for q = 1:length(commoninds)
    Aqind = commoninds[q][1]
    Bqind = commoninds[q][2]

    checksum = 0
    w = 0
    while w < 2
      w += 1
      Aind = Bind = w

      rmax = length(A.ind[Aqind][Aind])
      checksum += rmax - length(B.ind[Bqind][Bind])

      r = 0
      @inbounds while r < rmax && checksum == 0
        r += 1
        checksum += A.ind[Aqind][Aind][r]-B.ind[Bqind][Bind][r]
      end
    end

    if checksum == 0
      tensorcombination!(A.T[Aqind],B.T[Bqind],alpha=alpha)
    else      
      
      indsAL = A.ind[Aqind][1]
      indsAR = A.ind[Aqind][2]
      indsBL = B.ind[Bqind][1]
      indsBR = B.ind[Bqind][2]

      Lrows = Array{intType,1}(undef,size(indsAL,2))
      Lcols = Array{intType,1}(undef,size(indsAR,2))
      Rrows = Array{intType,1}(undef,size(indsBL,2))
      Rcols = Array{intType,1}(undef,size(indsBR,2))

      rowcolvec = [Lrows,Lcols,Rrows,Rcols]
      indsvec = [indsAL,indsAR,indsBL,indsBR]

      @inbounds for p = 1:length(rowcolvec)
        G = rowcolvec[p]
        K = indsvec[p]

        Aind = (p-1) % 2 + 1
        blocksizes = ntuple(n->length(A.QnumMat[A.currblock[Aind][n]]),length(A.currblock[Aind]))

        for x = 1:length(G)
          z = K[end,x]
          @inbounds @simd for y = length(blocksizes)-1:-1:1
            z *= blocksizes[y]
            z += K[y,x]
          end
          G[x] = z+1
        end
      end

      newrows = sort!(unique(vcat(Lrows,Rrows)))
      newcols = sort!(unique(vcat(Lcols,Rcols)))

      newmat = zeros(W,length(newrows),length(newcols))
      newindsL = Array{intType,2}(undef,size(indsAL,1),length(newrows))
      newindsR = Array{intType,2}(undef,size(indsAR,1),length(newcols))

      subsetL = findvals(Lrows,newrows)
      subsetR = findvals(Lcols,newcols)

      for y = 1:length(subsetR)
        @inbounds @simd for x = 1:length(subsetL)
          newmat[subsetL[x],subsetR[y]] += fct(A.T[Aqind][x,y],alpha[1])
        end
      end
      newindsL[:,subsetL] = indsAL
      newindsR[:,subsetR] = indsAR

      subsetL = findvals(Rrows,newrows)
      subsetR = findvals(Rcols,newcols)

      for y = 1:length(subsetR)
        @inbounds @simd for x = 1:length(subsetL)
          newmat[subsetL[x],subsetR[y]] += fct(B.T[Bqind][x,y],alpha[2])
        end
      end
      newindsL[:,subsetL] = indsBL
      newindsR[:,subsetR] = indsBR

      A.T[Aqind] = newmat
      A.ind[Aqind] = (newindsL,newindsR)

    end
  end

  Bleftover = findextrablocks(B,commoninds)

  mult = alpha[2]
  if length(Bleftover) > 0
    AQblocks = length(A.T)
    newT = Array{Array{W,2},1}(undef,AQblocks+length(Bleftover))
    newind = Array{NTuple{2,Array{intType,2}},1}(undef,length(newT))
    newQblocksum = Array{NTuple{2,Q},1}(undef,length(newT))
    @inbounds for q = 1:AQblocks
      newT[q] = A.T[q]
      newind[q] = A.ind[q]
      newQblocksum[q] = A.Qblocksum[q]
    end
    @inbounds for q = 1:length(Bleftover)
      addq = Bleftover[q]
      if isapprox(mult,1)
        Bout = B.T[addq]
      else
        Bout = tensorcombination(B.T[addq],alpha=(mult,))
      end
      c = q+AQblocks
      newT[c] = Bout
      newind[c] = B.ind[addq]
      newQblocksum[c] = B.Qblocksum[addq]
    end
    A.T = newT
    A.ind = newind
    A.Qblocksum = newQblocksum
  end
  return A
end

"""
  G = tensorcombination(M...[,alpha=(1,1,1...),fct=*])

Performs a linear combination (making a copy of the first entry of `M`) of the input `qarray` `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination(M::Qtens{W,Q}...;alpha::NTuple{N,W}=ntuple(i->eltype(M[1])(1),length(M)),fct::Function=*) where {Q <: Qnum, W <: Number, N}
  A = copy(M[1])
  newtup = (A,Base.tail(M)...)
  return tensorcombination!(newtup...,alpha=alpha,fct=fct)
end

"""
  G = tensorcombination(alpha,M...[,fct=*])

Performs a linear combination of the input `qarray` `M` with coefficients `alpha` to output tensor `G`.  For example, `A*2 + B*3` is `tensorcombinaton(A,B,alpha=(2,3))`.

This version converts input `alpha` to the same type as the element type of `M`

The input function `fct` can be altered. For example,  A/2 + B/3 is tensorcombinaton(A,B,alpha=(2,3),fct=/).
"""
function tensorcombination(alpha::NTuple{N,W},M::Qtens{W,Q}...;fct::Function=*) where {Q <: Qnum, W <: Number, N}
  return tensorcombination(M...,alpha=alpha,fct=fct)
end