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
    denstens

Abstract type for either dense tensors

See also: [`tens`](@ref)
"""
abstract type denstens end
export denstens

"""
    qarray

Abstract type for either `Qtensors`

See also: [`Qtens`](@ref)
"""
abstract type qarray end
export qarray

"""
    TNobj

Abstract type for network type function

See also: [`directedtens`](@ref) [`nametens`](@ref)
"""
abstract type TNobj end
export TNobj


"""
    TensType

Abstract type for either Qtensors or AbstractArrays or dense tensors, directed tensors, or named tensors

See also: [`Qtens`](@ref) [`denstens`](@ref) [`directedtens`](@ref) [`nametens`](@ref)
"""
const TensType = Union{qarray,denstens,AbstractArray}
export TensType

"""
    densTensType

Abstract type for either AbstractArrays or dense tensors

See also: [`denstens`](@ref)
"""
const densTensType = Union{AbstractArray,denstens}
export densTensType

"""
    tenstype

Duplicate of `TensType`

See also: [`TensType`](@ref)
"""
const tenstype = TensType
export tenstype

"""
    intType

`intType` = Int64 (or Int32)
"""
const intType = typeof(1)
export intType

"""
    intvectype

equivalent to `Union{intType,Array{intType,1},Array{intType,2}}`

See also: [`intType`](@ref)
"""
const intvecType = Union{P,Array{P,1},Array{P,2},NTuple{G,P}} where {P <: Integer, G}
export intvecType

"""
    genColType

All types input into `getindex!` (UnitRange,intType,Array{intType,1},Colon)

See also: [`getindex!`](@ref)
"""
const genColType = Union{UnitRange{intType},intType,Array{intType,1},Colon,StepRange{intType},Tuple{intType,Vararg{intType}}}
export genColType

"""
    file_extension

A default file extension can be specified for the large types.

See also: [`largeMPS`](@ref) [`largeMPO`](@ref) [`largeEnv`](@ref)
"""
const file_extension = ".tenpack"

"""
    bigvec

Data type for vectors that are stored on the hard disk instead of ram (written to file)

See also: [`largevector`](@ref)
"""
abstract type bigvec end
export bigvec

"""
    Qnum

Abstract types for quantum numbers
"""
abstract type Qnum end

"""
    fermionQnum

Abstract types for fermionic quantum numbers that includes parity
"""
abstract type fermionQnum <: Qnum end
export Qnum,fermionQnum
