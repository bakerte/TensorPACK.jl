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
    `largevector` 
  
Construct this container with `bigvec`. struct to hold `bigvec` tensors

# Fields:
+ `V::Array{String,1}`: filenames where the tensors are stored on disk
+ `type::DataType`: DataType of the stored tensors

See also: [`bigvec`](@ref)
"""
struct largevector{W} <: bigvec where W <: DataType
  V::Array{String,1}
  type::W
end

"""
    G = loadlargevector(Ns[,label="mpo_",names=[label*"i" for i = 1:Ns],ext=".dmrjulia"])

If MPO tensors are stored on hard disk, then they can be retrieved by using `loadlargevector` for an `Ns` site system with labels `label` and `names` for the filenames with extension `ext`
"""
function loadlargevector(Ns::Integer;label::String="mpo_",names::Array{String,1}=[label*"$i" for i = 1:Ns],ext::String=file_extension,oc::Integer=0)
  lastnum = 1
  for i = 1:Ns
    name = names[i]
    A = tensorfromdisc(name,ext=ext)
    lastnum *= eltype(A)(1)
  end
  thistype = typeof(lastnum)
  return bigvec(names,thistype)
end
export loadlargevector

