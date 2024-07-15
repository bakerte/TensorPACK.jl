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
























#=

"""
  orderblocks!(A,Aqind,B,Bqind)

Orders blocks from Qtens `A` (`Aqind` block) and `B` (`Bqind` block) to both match. Will search both the left and right columns of the blocks. Assumes same block structure between the tensors
"""
function orderblocks!(A::Qtens{W,Q},Aqind::Integer,B::Qtens{W,Q},Bqind::Integer) where {W <: Number, Q <: Qnum}

#  println(size(A)," ",size(B))

  for w = 1:2
    Aind = Bind = w

    checksum = 0

    minAB = min(length(A.ind[Aqind][Aind]),length(B.ind[Bqind][Bind]))
    r = 0
    @inbounds while r < minAB && checksum == 0
      r += 1
      checksum += A.ind[Aqind][Aind][r]-B.ind[Bqind][Bind][r]
    end

    mulblockA = A.T[Aqind]
    mulblockB = B.T[Bqind]
    indsA = A.ind[Aqind][Aind]
    indsB = B.ind[Bqind][Bind]

    if !(length(indsA) == length(indsB) && checksum == 0)
#      Lpos = Array{intType,1}(undef,length(A.currblock[Aqind]))
      blocksizes = ntuple(n->length(A.QnumMat[A.currblock[Aind][n]]),length(A.currblock[Aind]))
      Lrowcols = Array{intType,1}(undef,size(indsA,2))
      Rrowcols = Array{intType,1}(undef,size(indsB,2))

      for p = 1:2
        if p == 1
          G = Lrowcols
          K = indsA
        else
          G = Rrowcols
          K = indsB
        end
        for x = 1:length(G)
          z = K[end,x]
          @inbounds @simd for y = length(blocksizes)-1:-1:1
            z *= blocksizes[y]
            z += K[y,x]
          end
          G[x] = z+1
        end
      end

      equalinds = length(Lrowcols) == length(Rrowcols)
      if equalinds
        k = 0
        while equalinds && w < length(Lrowcols)
          k += 1
          equalinds = Lrowcols[k] == Rrowcols[k]
        end
      end
      if !equalinds
        commoninds = intersect(Lrowcols,Rrowcols)
        if !issorted(commoninds)
          sort!(commoninds)
        end
        orderL = sortperm(Lrowcols)
        orderR = sortperm(Rrowcols)

        keepL = Array{intType,1}(undef,length(commoninds))
        keepR = Array{intType,1}(undef,length(commoninds))


        for p = 1:length(commoninds)
          b = 1
          @inbounds while b < length(orderR) && Lrowcols[orderL[p]] != Rrowcols[orderR[b]]
            b += 1
          end
          keepL[p] = orderL[p]
          keepR[p] = orderR[b]
        end

        if w == 1
          mulblockA = mulblockA[keepL,:]
          mulblockB = mulblockB[keepR,:]
        else
          mulblockA = mulblockA[:,keepL]
          mulblockB = mulblockB[:,keepR]
        end

        A.T[Aqind] = mulblockA
        B.T[Bqind] = mulblockB

        indsA = indsA[:,keepL]
        indsB = indsB[:,keepR]

        if w == 1
          A.ind[Aqind] = (indsA,A.ind[Aqind][2])
          B.ind[Bqind] = (indsB,B.ind[Bqind][2])
        else
          A.ind[Aqind] = (A.ind[Aqind][1],indsA)
          B.ind[Bqind] = (B.ind[Bqind][1],indsB)
        end
      end
    end

  end
  nothing
end


"""
  orderblocks(A,Aqind,Aind,B,Bqind,Bind)

Orders blocks from Qtens `A` (`Aqind` block) and `B` (`Bqind` block) to both match. Will search the indexes of `Aind` and `Bind` (must be values 1 or 2 for left or right sets of indexes)
"""
function orderblocks!(A::Qtens{W,Q},B::Qtens{W,Q},ABinds::Array{NTuple{2,intType},1}) where {W <: Number, Q <: Qnum}
  for w = 1:length(ABinds)
    orderblocks!(A,ABinds[w][1],B,ABinds[w][2])
  end
  nothing
end


=#

















































































#=
"""
    Idhelper(A,iA)

generates the size of matrix equivalent of an identity matrix from tensor `A` with indices `iA`

#Output:
+`lsize::Int64`: size of matrix-equivalent of identity operator
+`finalsizes::Int64`: size of identity operator

See also: [`makeId`](@ref) [`trace`](@ref)
"""
function Idhelper(A::TensType,iA::Array{NTuple{2,P},1}) where P <: Integer
  lsize = prod(w->size(A,iA[w][1]),1:length(iA))
  leftsizes = ntuple(w->size(A,iA[w][1]),length(iA))
  rightsizes = ntuple(w->size(A,iA[w][2]),length(iA))
  finalsizes = (leftsizes...,rightsizes...)
  return lsize,finalsizes
end
=#




