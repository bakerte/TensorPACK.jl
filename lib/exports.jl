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

export dtens
export dualnum


export sin
export cos


export convertTens


export add!

export convIn


export det

export logdet


export norm
export norm!

export kron

export checkType


export Array
export tens
export network
export nametens
export directedtens

export eye

export exp!
export elnumtype
export div!

export invmat
export invmat!

export getindex!
export searchindex
export isinteger

export addindex!
export addindex
export joinTens

export root

export pos2ind
export pos2ind!
export position_incrementer!
export makepos!
export makepos

export mult!



export directsum
export directsum!
export joinindex
export joinindex!

##Contract
export contract!

export contract,ccontract,contractc,ccontractc

export diagcontract!

export findnotcons
export getinds

export dmul!
export dot


export tensorcombination
export tensorcombination!


export reshape!
export unreshape
export unreshape!



export multi_indexsummary
export recoverShape
export mergereshape
export mergereshape!
export newindexsizeone!
export getQnum




export swapname!
export swapnames!

export trace!
export trace

export sqrt!

export sub!


export tupsize




##Decompose

export eigen
export eigen!
export eigvals
export eigvals!





export krylov
export lanczos

export lq
export lq!

export qr,qr!

export nullspace
export polar


export svd
export svd!
export symsvd
export svdvals


export ⊕,⊗

