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
    L,Q,0.,1. = lq(A,vecA[,decomposer=liblq,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2.

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `L`: a weighted matrix
+ `Q`: a unitary matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function lq(AA::TensType,vecA::Array{Array{W,1},1};a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2]),decomposer::Function=liblq,leftflux::Bool=false) where W <: Integer
  return qr(AA,vecA,decomposer=decomposer,a=a,b=b)
end

"""
    L,Q,0.,1. = lq!(A,vecA[,decomposer=liblq!,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` for `svd` and then unreshapes U and V matrices on return; `vecA` is of the form [[1,2],[3,4,5]] and must be length 2. Overwrites some input data from `A`

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `L`: a weighted matrix
+ `Q`: a unitary matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function lq!(AA::TensType,vecA::Array{Array{W,1},1};a::Integer=findsize(AA,vecA[1]),b::Integer=findsize(AA,vecA[2]),decomposer::Function=liblq!,leftflux::Bool=false) where W <: Integer
  return qr(AA,vecA,decomposer=decomposer,a=a,b=b)
end

"""
    L,Q,0.,1. = lq(A[,decomposer=liblq,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` (`denstens`) for `svd` and then unreshapes U and V matrices on return

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `L`: a weighted matrix
+ `Q`: a unitary matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function lq(AA::densTensType;a::Integer=size(AA,1),b::Integer=size(AA,2),decomposer::Function=liblq,leftflux::Bool=false)
  return qr(AA,decomposer=decomposer,a=a,b=b)
end

"""
    L,Q,0.,1. = lq!(A[,decomposer=liblq!,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` (`denstens`) for `svd` and then unreshapes U and V matrices on return

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `L`: a weighted matrix
+ `Q`: a unitary matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function lq!(AA::densTensType;a::Integer=size(AA,1),b::Integer=size(AA,2),decomposer::Function=liblq!,leftflux::Bool=false)
  return qr(AA,decomposer=decomposer,a=a,b=b)
end


#       +------------------------+
#>------| Quantum number version |---------<
#       +------------------------+

"""
    L,Q,0.,1. = lq(A[,decomposer=liblq,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` (`Qtens`) for `svd` and then unreshapes U and V matrices on return

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `L`: a weighted matrix
+ `Q`: a unitary matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function lq(QtensA::Qtens{W,Q};leftflux::Bool=false,a::Integer=1,b::Integer=1,decomposer::Function=liblq) where {W <: Number, Q <: Qnum}
  return qr(QtensA,leftflux=leftflux,decomposer=decomposer,a=a,b=b)
end

function liblq(QtensA::Qtens{W,Q},a::Integer,b::Integer) where {W <: Number, Q <: Qnum}
  return qr(QtensA,a=a,b=b,decomposer=liblq)
end

"""
    L,Q,0.,1. = lq!(A[,decomposer=liblq!,a=size(A,1),b=size(A,2),leftflux=false])

Reshapes `A` (`Qtens`) for `svd` and then unreshapes U and V matrices on return; some operations may be in place

#Optional additional inputs:
+ `a`: total size of the first group
+ `b`: total size of the second group
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `L`: a weighted matrix
+ `Q`: a unitary matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function lq!(QtensA::Qtens{W,Q};leftflux::Bool=false,a::Integer=1,b::Integer=1,decomposer::Function=liblq!) where {W <: Number, Q <: Qnum}
  return qr(QtensA,leftflux=leftflux,decomposer=decomposer,a=a,b=b)
end

function liblq!(QtensA::Qtens{W,Q},a::Integer,b::Integer) where {W <: Number, Q <: Qnum}
  return qr(QtensA,a=a,b=b,decomposer=liblq)
end

"""
    L,Q,0.,1. = lq(A,order[,name="lqind",leftadd="L",leftflux=false,decomposer=liblq])

Generates lq decomposition of named tensor `A` according to `order`

# Naming created index:
+ the index to the left of `D` is `vcat(name,leftadd)`

#Inputs:
+ `A`: Any `TensType` in the library
+ `order`: vector containing index groups given as `String`

#Optional additional inputs:
+ `decomposer`: library function to perform decomposition
+ `leftflux`: where to attach total flux value to quantum number symmetry if applicable

#Outputs:
+ `L`: a weighted matrix
+ `Q`: a unitary matrix

The remaining outputs match `svd` and `eigen`

See also: [`svd`](@ref) [`eigen`](@ref)
"""
function lq(AA::nametens,order::Array{Array{B,1},1};decomposer::Function=liblq,name::String="lqind",leftadd::String="L",leftflux::Bool=false) where B <: String

  left = matchnames(AA,order[1])
  right = matchnames(AA,order[2])
  neworder = [left,right]
  leftname = name*leftadd

  Q,R = lq(AA.N,neworder,leftflux=leftflux,decomposer=decomposer)

  TNobjQ = nametens(Q,vcat(AA.names[left],[leftname]...))
  TNobjR = nametens(R,[leftname,AA.names[right]...])

  return TNobjQ,TNobjR
end