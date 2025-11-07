
import Serialization
file = testpath*"dict"*file_extension
if isfile(file)
  performancevals = Serialization.deserialize(file)
else
  performancevals = Dict()
end

println("#            +-------------------+")
println("#>-----------|  contractions.jl  |-----------<")
println("#            +-------------------+")
fulltest = true

A = rand(10,20,30,10,5)
tA = tens(A)
iA = [1,2,3,4]
iB = (iA...,)

testval = "TENPACK.permq(A,iA)[1] && TENPACK.permq(A,iB)[1]"
fulltest &= testfct(testval,"permq(Array,Vector || Tuple)",performancevals)

testval = "TENPACK.permq(tA,iA)[1] && TENPACK.permq(tA,iB)[1]"
fulltest &= testfct(testval,"permq(tens,Vector || Tuple)",performancevals)

A = rand(10,20,30,10,5)
iA = [2,3,4,5]
iB = (iA...,)

testval = "TENPACK.permq(A,iA)[2] && TENPACK.permq(A,iB)[2]"
fulltest &= testfct(testval,"permq(Array,Vector || Tuple)",performancevals)

testval = "TENPACK.permq(A,iA)[2] && TENPACK.permq(A,iB)[2]"
fulltest &= testfct(testval,"permq(tens,Vector || Tuple)",performancevals)

println()

A = TENPACK.willperm(true,ComplexF64,true,false)
testval = true == A[1] && 'C' == A[2]
A = TENPACK.willperm(true,ComplexF64,false,true)
testval &= true == A[1] && 'N' == A[2]
A = TENPACK.willperm(true,Float64,true,false)
testval &= true == A[1] && 'T' == A[2]
A = TENPACK.willperm(false,Float64,true,false)
testval &= true == A[1] && 'T' == A[2]

fulltest &= testfct(testval,"willperm")

println()

A = rand(ComplexF64,10,20,30,40,10,20)
iA = [1,2,3,4]
B = TENPACK.prepareT(A,iA,[5,6],false)

testval = isapprox(norm(A),norm(B))

A = rand(ComplexF64,10,20,30,40,10,20)
B = TENPACK.prepareT(A,[1,2,3,4],[5,6],true)

testval &= isapprox(norm(A),norm(B)')

fulltest &= testfct(testval,"prepareT")

println()

B = (0,)

Lsize,innersizeL = TENPACK.getsizes(A,(iA...,))
testval = Lsize == size(A,1)*size(A,2)
testval &= innersizeL == prod(w->size(A,w),3:6)

fulltest &= testfct(testval,"getsizes")

println()

A = rand(20,40)
B = rand(40,20)
C = rand(20,20)

iA = [2]
iB = [1]

checkC = TENPACK.maincontractor(false,false,A,iA,B,iB,alpha=2.)

testval = "isapprox(checkC,2. * A*B)#+2.*C)"
fulltest &= testfct(testval,"alpha*A*B [Array]",performancevals)


checkC = TENPACK.maincontractor(false,false,A,iA,B,iB,copy(C),alpha=2.,beta=3.)

testval = "isapprox(checkC,2. * A*B + 3. * C)"
fulltest &= testfct(testval,"alpha*A*B + beta*C [Array]",performancevals)

checkC = TENPACK.maincontractor(false,false,tens(A),iA,tens(B),iB,alpha=2.)

testval = "isapprox(checkC.T,reshape(2. * A*B,400))#+2.*C)"
fulltest &= testfct(testval,"alpha*A*B [tens]",performancevals)


checkC = TENPACK.maincontractor(false,false,tens(A),iA,tens(B),iB,tens(copy(C)),alpha=2.,beta=3.)

testval = "isapprox(checkC.T,reshape(2. * A*B + 3. * C,400))"
fulltest &= testfct(testval,"alpha*A*B + beta*C [tens]",performancevals)

println()

A = rand(20,40)
B = rand(20,40)

testval = "isapprox(dot(A,B),dot(tens(A),tens(B)))"
fulltest &= testfct(testval,"dot(A,B) [Array || tens]",performancevals)

A = rand(40)
C = rand(40,40)
B = rand(40)

tB = adjoint(B)
res = dot(A,C,B)
testval = isapprox(A'*C*B,res)
testval &= isapprox(res,dot(tens(A),tens(C),tens(B)))
fulltest &= testfct(testval,"dot(A,B,C) [Array || tens]")

println()

A = rand(20,5,6,10)
B = rand(10,6,20,5)

C = A*B
tC = tens(A)*tens(B)

testval = size(C) == (20,5,6,6,20,5)
testval &= size(tC) == (20,5,6,6,20,5)
testval = isapprox(reshape(C,prod(size(C))),tC.T)
fulltest &= testfct(testval,"*(A,B) [Array || tens]")

println()

import LinearAlgebra

A = rand(10,2,10)
B = Diagonal(rand(10))

C = A*B
D = B*A

tC = tens(A)*B
tD = B*tens(A)

testval = isapprox(reshape(C,prod(size(C))),tC.T)
testval = isapprox(reshape(D,prod(size(D))),tD.T)

fulltest &= testfct(testval,"*(A,diagonal) [Array || tens, commutes]")

println()

C = contract(copy(A),3,B,1)
tC = dmul!(copy(tens(A)),B)

testval = "isapprox(reshape(C,prod(size(C))),tC.T)"
fulltest &= testfct(testval,"dmul!(A,diagonal)",performancevals)

C = 3. * copy(A)
tC = dmul!(3.,copy(tens(A)))

testval = "isapprox(reshape(C,prod(size(C))),tC.T)"
fulltest &= testfct(testval,"dmul!(tens,number)",performancevals)

println()

D = contract(B,2,copy(A),1)
tD = dmul!(B,copy(tens(A)))

testval = "isapprox(reshape(D,prod(size(D))),tD.T)"
fulltest &= testfct(testval,"dmul!(diagonal,tens)",performancevals)

D = copy(A) * 3.
tD = dmul!(copy(tens(A)),3.)

testval = "isapprox(reshape(D,prod(size(D))),tD.T)"
fulltest &= testfct(testval,"dmul!(number,tens)",performancevals)


println()

A = rand(ComplexF64,10,10)
B = rand(ComplexF64,10,10)

testval = "isapprox(contract(A,2,B,1),A*B)"
fulltest &= testfct(testval,"contract(Array,vec,Array,vec)",performancevals)

testval = "isapprox(ccontract(A,2,B,1),conj(A)*B)"
fulltest &= testfct(testval,"ccontract(Array,vec,Array,vec)",performancevals)

testval = "isapprox(contractc(A,2,B,1),A*conj(B))"
fulltest &= testfct(testval,"contractc(Array,vec,Array,vec)",performancevals)

testval = "isapprox(ccontractc(A,2,B,1),conj(A)*conj(B))"
fulltest &= testfct(testval,"ccontractc(Array,vec,Array,vec)",performancevals)

A = tens(A)
B = tens(B)

testval = "isapprox(contract(A,2,B,1).T,(A*B).T)"
fulltest &= testfct(testval,"contract(tens,vec,tens,vec)",performancevals)

testval = "isapprox(ccontract(A,2,B,1).T,(conj(A)*B).T)"
fulltest &= testfct(testval,"ccontract(tens,vec,tens,vec)",performancevals)

testval = "isapprox(contractc(A,2,B,1).T,(A*conj(B)).T)"
fulltest &= testfct(testval,"contractc(tens,vec,tens,vec)",performancevals)

testval = "isapprox(ccontractc(A,2,B,1).T,conj(A*B).T)"
fulltest &= testfct(testval,"ccontractc(tens,vec,tens,vec)",performancevals)

println()



A = rand(ComplexF64,10,20,30,40)
B = rand(ComplexF64,10,20,30,40)

C = transpose(reshape(A,prod(size(A))))*reshape(B,prod(size(B)))

testval = "isapprox(contract(A,B),C)"
fulltest &= testfct(testval,"contract(Array,Array)",performancevals)

C = conj(transpose(reshape(A,prod(size(A)))))*reshape(B,prod(size(B)))

testval = "isapprox(ccontract(A,B),C)"
fulltest &= testfct(testval,"ccontract(Array,Array)",performancevals)

C = (transpose(reshape(A,prod(size(A)))))*conj(reshape(B,prod(size(B))))

testval = "isapprox(contractc(A,B),C)"
fulltest &= testfct(testval,"contractc(Array,Array)",performancevals)

C = conj(transpose(reshape(A,prod(size(A)))))*conj(reshape(B,prod(size(B))))

testval = "isapprox(ccontractc(A,B),C)"
fulltest &= testfct(testval,"ccontractc(Array,Array)",performancevals)

A = tens(A)
B = tens(B)

C = dot(A,B,Lfct=identity,Rfct=identity)

testval = "isapprox(contract(A,B),C)"
fulltest &= testfct(testval,"contract(tens,tens)",performancevals)

C = dot(A,B,Lfct=adjoint,Rfct=identity)

testval = "isapprox(ccontract(A,B),C)"
fulltest &= testfct(testval,"ccontract(tens,tens)",performancevals)


C = dot(A,B,Lfct=identity,Rfct=adjoint)

testval = "isapprox(contractc(A,B),C)"
fulltest &= testfct(testval,"contractc(tens,tens)",performancevals)


C = dot(A,B,Lfct=adjoint,Rfct=adjoint)

testval = "isapprox(ccontractc(A,B),C)"
fulltest &= testfct(testval,"ccontractc(tens,tens)",performancevals)

println()

testval = "contract(A,(1,2,3,4),B,(1,2,3,4))[1] == contract(A,(1,2,3,4),B)[1] == contract(A,B,(1,2,3,4))[1]"

fulltest &= testfct(testval,"contract(tens,[vec,]tens[,vec])",performancevals)

testval = "ccontract(A,(1,2,3,4),B,(1,2,3,4))[1] == ccontract(A,(1,2,3,4),B)[1] == ccontract(A,B,(1,2,3,4))[1]"

fulltest &= testfct(testval,"ccontract(tens,[vec,]tens[,vec])",performancevals)

testval = "contractc(A,(1,2,3,4),B,(1,2,3,4))[1] == contractc(A,(1,2,3,4),B)[1] == contractc(A,B,(1,2,3,4))[1]"

fulltest &= testfct(testval,"contractc(tens,[vec,]tens[,vec])",performancevals)

testval = "ccontractc(A,(1,2,3,4),B,(1,2,3,4))[1] == ccontractc(A,(1,2,3,4),B)[1] == ccontractc(A,B,(1,2,3,4))[1]"

fulltest &= testfct(testval,"ccontractc(tens,[vec,]tens[,vec])",performancevals)


println()

A = rand(10,20,10,30#=,40,40=#);
B = trace(copy(A),[[1,3]#=,[5,6]=#])

C = eye(copy(A),[1])
checkB = contract(copy(A),[1,3#=,5,6=#],C,[1,2#=,3,4=#])

testval = "size(B) == (20,30) && isapprox(checkB,B)"
fulltest &= testfct(testval,"trace(Array,[indices])",performancevals)



A = rand(10,20,10,30,40,40);
B = trace(copy(A),[[1,3],[5,6]])

A = tens(A)
tB = trace(A,[[1,3],[5,6]])

tC = eye(A,[1,5])
checkB = contract(A,[1,3,5,6],tC,[1,2,3,4])

testval = size(B) == (20,30)
testval &= isapprox(tB.T,checkB.T)
fulltest &= testfct(testval,"trace(tens,[indices])")

println()

A = rand(10,20,10,30#=,40,40=#);
B = trace!(copy(A),[[1,3]#=,[5,6]=#])

C = eye(copy(A),[1])
checkB = contract(copy(A),[1,3#=,5,6=#],C,[1,2#=,3,4=#])

testval = size(B) == (20,30) 
testval &= isapprox(checkB,B)
fulltest &= testfct(testval,"trace!(Array,[indices])")



A = rand(10,20,10,30,40,40);
B = trace!(copy(A),[[1,3],[5,6]])

A = tens(A)
tB = trace!(copy(A),[[1,3],[5,6]])

tC = eye(A,[1,5])
checkB = contract(A,[1,3,5,6],tC,[1,2,3,4])

testval = "size(B) == (20,30) && isapprox(tB.T,checkB.T)"
fulltest &= testfct(testval,"trace!(tens,[indices])",performancevals)


Serialization.serialize(file,performancevals)

fulltest