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
    C = dot(A,B[,Lfct=adjoint,Rfct=identity])

contraction of two tensors `A` and `B` (`densTensType`) to a scalar `C`; takes `identity` or `adjoint` (or equivalently `conj`) for the `Lfct` and `Rfct`

See also: [`identity`](@ref) [`adjoint`](@ref) [`conj`](@ref)
"""
function dot(inA::densTensType,inB::densTensType;Lfct::Function=adjoint,Rfct::Function=identity)
  A = typeof(inA) <: denstens ? inA.T : inA
  B = typeof(inB) <: denstens ? inB.T : inB

  newtypeAB = typeof(eltype(A)(1) * eltype(B)(1))
  val = newtypeAB(0)
  dim1 = length(A) #size(inA,transA ? 1 : 2)
  @inbounds @simd for j = 1:dim1
    val += Lfct(A[j]) * Rfct(B[j])
  end
  return val
end
export dot

"""
    C = dot(A,H,B[,Lfct=adjoint,Rfct=identity])

The function will admit any dimension or input element type in terms of the arrays `A`, `H`, and `B` (`densTensType`). However, the function will only work properly when the total elements of `A` times those in `B` equal the elements in `H`

If more operators `H` should be contracted between `A` and `B`, then it is advised here to contract them first before using this function

"""
function dot(inA::densTensType,inH::densTensType,inB::densTensType;Lfct::Function=adjoint,Rfct::Function=identity)

  A = typeof(inA) <: denstens ? inA.T : inA
  H = typeof(inH) <: denstens ? inH.T : inH
  B = typeof(inB) <: denstens ? inB.T : inB

  newtypeAHB = typeof(eltype(A)(1) * eltype(H)(1) * eltype(B)(1))
  val = newtypeAHB(0)

  dim1 = length(A) #size(inA,transA ? 1 : 2)
  dim2 = length(B) #size(inB,1)
  newtypeAH = typeof(eltype(A)(1) * eltype(H)(1))
  @inbounds for j = 1:dim2
    ival = newtypeAH(0)
    savedim = dim1*(j-1)
    @inbounds @simd for i = 1:dim1
      ival += Lfct(A[i]) * H[i + savedim]
    end
    val += ival * Rfct(B[j])
  end
  return val
end
export dot

"""
    C = dot(A,B[,Lfct=adjoint,Rfct=identity])

contraction of two tensors `A` and `B` (`qarray`) to a scalar `C`; takes `identity` or `adjoint` (or equivalently `conj`) for the `Lfct` and `Rfct`

See also: [`identity`](@ref) [`adjoint`](@ref) [`conj`](@ref)
"""
function dot(C::Qtens{W,Q},D::Qtens{R,Q};Lfct::Function=adjoint,Rfct::Function=identity) where {W <: Number, R <: Number, Q <: Qnum}
  newtype = typeof(W(1)*R(1))

  A = changeblock(C,intType[],intType[i for i = 1:length(C.QnumMat)])
  B = changeblock(D,intType[i for i = 1:length(D.QnumMat)],intType[])
  conjA = Lfct != identity
  conjB = Rfct != identity
  commonblocks = matchblocks((conjA,conjB),A,B,ind=(2,1))

  val = newtype(0)
  @inbounds for q = 1:length(commonblocks)
    Aqind = commonblocks[q][1]
    Bqind = commonblocks[q][2]
    val += dot(A.T[Aqind],B.T[Bqind],Lfct=Lfct,Rfct=Rfct)
  end
  return val
end

#not sure how to assign matrix blocks...or how to get information from that to A and B vectors
#must be set up correctly with changeblock
"""
    C = dot(A,H,B[,Lfct=adjoint,Rfct=identity])

The function will admit any dimension or input element type in terms of the arrays `A`, `H`, and `B` (`qarray`). However, the function will only work properly when the total elements of `A` times those in `B` equal the elements in `H`

If more operators `H` should be contracted between `A` and `B`, then it is advised here to contract them first before using this function

"""
function dot(A::Qtens{W,Q},H::Qtens{Y,Q},B::Qtens{R,Q};Lfct::Function=adjoint,Rfct::Function=identity) where {W <: Number, Y <: Number, R <: Number, Q <: Qnum}
  Acommonblocks = matchblocks((conjA,false),A,H,ind=(2,1))
  Bcommonblocks = matchblocks((conjA,conjB),A,B,ind=(2,1))
  newtype = typeof(W(1)*R(1))
  val = newtype(0)
  @inbounds for q = 1:length(commonblocks)
    Aqind = Acommonblocks[q][1]
    Hqind = Acommonblocks[q][2]
    Bqind = Bcommonblocks[q][2]
    if length(A.T[Aqind]) * length(B.T[Bqind]) != length(H.T[Hqind])
      error("unequal sizes in dot for quantum number tensors for A block: $Aqind, H block: $Hqind, and B block: $Bqind")
    end
    if Aqind != 0 && Hqind != 0 && Bqind != 0
      val += dot(A.T[Aqind],H.T[Hqind],B.T[Bqind],Lfct=Lfct,Rfct=Rfct)
    end
  end
  return val
end


"""
    C = dot(A,B[,Lfct=adjoint,Rfct=identity])

contraction of two tensors `A` and `B` (`dtens`) to a scalar `C` and its derivative; takes `identity` or `adjoint` (or equivalently `conj`) for the `Lfct` and `Rfct`

See also: [`identity`](@ref) [`adjoint`](@ref) [`conj`](@ref)
"""
function dot(A::dtens,B::dtens;Lfct::Function=adjoint,Rfct::Function=identity)
  return dot(A[0],B[0],Lfct=Lfct,Rfct=Rfct),dot(A[0],B[1],Lfct=Lfct,Rfct=Rfct)+dot(A[1],B[0],Lfct=Lfct,Rfct=Rfct)
end
