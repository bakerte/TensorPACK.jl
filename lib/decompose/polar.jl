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
    leftT,rightT,D,truncerr,mag = polar(A[,cutoff=0.,m=0,mag=0.,minm=2,nozeros=false,power=1,effZero=defzero,keepdeg=false,decomposer=svd])

Performs a polar decomposition on tensor `A` with grouping `group` (default: [[1,2],[3]] for an MPS); if `left` (default), this returns U*D*U',U'*V else functoin returns U*V',V*D*V' from an `svd`

#Inputs:
+ `A`: Any `TensType` in the library

#Optional named inputs:
+ `right`: whether to place the `D` matrix on the left or the right output tensor
+ `cutoff`: total amount to truncate the eigenvalue decomposition
+ `m`: number of many body states to keep (`D` is an `m`x`m` matrix); value 0 is keep all
+ `mag`: magnitude of the input matrix (computed automatically be default, sets truncation parameters)
+ `merge`: merges reshapes for `Qtens` on output tensors
+ `minm`: will pad `D` with zeros if smaller size than `minm`
+ `nozeros`: removes all zero values from the decomposition
+ `power`: power of the eigenvalues to perform truncation (default: L-1 norm)
+ `effZero`: effective value of zero
+ `keepdeg`: `false` is a pure truncation, `true` does not truncate a degenerate value if represented on the last truncated value
+ `decomposer`: library function to perform decomposition

#Outputs:
+ `leftT`: left tensor from the polar decomposition
+ `rightT`: right tensor from the polar decomposition
+ `D`: A diagonal matrix containing eigenvalues
+ `truncerr`: total truncation error (L-1 norm)
+ `mag`: magnitude of the output tensor

See also: [`svd`](@ref)
"""
function polar(AA::TensType,group::Array{Array{W,1},1};
                right::Bool=true,cutoff::Float64 = 0.,m::Integer = 0,mag::Float64 = 0.,merge::Bool=true,decomposer::Function=svd,
                minm::Integer=2,nozeros::Bool=false,keepdeg::Bool=false,power::Number=2,effZero::Number=defzero) where W <: Integer

  U,D,V,truncerr,newmag = decomposer(AA,group,cutoff=cutoff,m=m,mag=mag,minm=minm,nozeros=nozeros,keepdeg=keepdeg,power=power,effZero=effZero)

  ndimU = ndims(U)
  ndimV = ndims(V)
  #polar decomposition
  if right
    rightTensor = ccontract(V,1,D*V,1)
    leftTensor = contract(U,ndimU,V,1)

    leftTensor = reshape!(leftTensor,[[[i] for i = 1:ndimU-1]...,[i+ndimU-1 for i = 1:ndimV-1]],merge=merge)
    rightTensor = reshape!(rightTensor,[[i for i = 1:ndimV-1],[[i+ndimV-1] for i = 1:ndimV-1]...],merge=merge)
  else
    leftTensor = contractc(U*D,ndimU,U,ndimU)
    rightTensor = contract(U,ndimU,V,1)

    leftTensor = reshape!(leftTensor,[[[i] for i = 1:ndimU-1]...,[i+ndimU-1 for i = 1:ndimU-1]],merge=merge)
    rightTensor = reshape!(rightTensor,[[i for i = 1:ndimU-1],[[i+ndimU-1] for i = 1:ndimV-1]...],merge=merge)
  end

  return leftTensor,rightTensor,D,truncerr,newmag
end
export polar