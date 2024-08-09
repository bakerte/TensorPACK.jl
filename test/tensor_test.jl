
import Serialization
file = testpath*"dict"*file_extension
if isfile(file)
  performancevals = Serialization.deserialize(file)
else
  performancevals = Dict()
end

println("#            +-------------+")
println("#>-----------|  tensor.jl  |-----------<")
println("#            +-------------+")
fulltest = true


ndim = max(round(Int64,rand()*10),2)
Asize = ntuple(w->max(2,round(Int64,rand()*10)),ndim)
A = rand(Asize...)


println("Array size (rank $ndim): ",Asize)

B = tA = tens(A)

sizetest = "(B.size...,) == Asize"
fulltest &= testfct(sizetest,"denstens .size test",performancevals)

normtest = "isapprox(norm(B.T),norm(A))"
fulltest &= testfct(normtest,"denstens .T field",performancevals)

println()

C = Diagonal(rand(10))

normtest = "isapprox(norm(C.T),norm(C))"
fulltest &= testfct(normtest,"Diagonal .T field",performancevals)

println()

println("input tests; default input auto-pass, tens(Array), by this point")

B = tens(type=ComplexF64)
loadtest1 = "B.size == (0,) && B.T == Array{ComplexF64,1}(undef,0)"
fulltest &= testfct(loadtest1,"tens(;type=)",performancevals)

B = tens(ComplexF64)
loadtest2 = "B.size == (0,) && B.T == Array{ComplexF64,1}(undef,0)"
fulltest &= testfct(loadtest2,"tens(type)",performancevals)


import LinearAlgebra
size_vec = rand(ndim)
C = LinearAlgebra.Diagonal(size_vec)
B = tens(ComplexF64,C)

testval = "B.size == (ndim,ndim) && isapprox(B.T,reshape(Array(C),ndim^2))"
fulltest &= testfct(testval,"tens(Type,AbstractArray)",performancevals)

B = tens(C)
testval = "B.size == (ndim,ndim) && isapprox(B.T,reshape(Array(C),ndim^2))"
fulltest &= testfct(testval,"tens(AbstractArray)",performancevals)

B = tens(ComplexF64,A)
testval = "B.size == tA.size && isapprox(B.T,convert(Array{ComplexF64,1},reshape(A,prod(size(A)))))"
fulltest &= testfct(testval,"tens(Type,Array{DiffType})",performancevals)

B = tens(Float64,A)
testval = B.size == tA.size && isapprox(B.T,convert(Array{ComplexF64,1},tA.T))
fulltest &= testfct(testval,"tens(Type,Array{Type})")


B = tens{ComplexF64}(A)
testval = "B.size == tA.size && isapprox(B.T,convert(Array{ComplexF64,1},tA.T))"
fulltest &= testfct(testval,"tens{DiffType}(Array{Type})",performancevals)

B = tens{ComplexF64}(tA)
testval = "B.size == tA.size && isapprox(B.T,convert(Array{ComplexF64,1},tA.T))"
fulltest &= testfct(testval,"tens{DiffType}(tens{Type})",performancevals)

B = tens{Float64}(tA)
testval = "B.size == tA.size && isapprox(B.T,convert(Array{Float64,1},tA.T))"
fulltest &= testfct(testval,"tens{Type}(tens{Type})",performancevals)

println()

B = rand(tA)
testval = "B.size == tA.size"
fulltest &= testfct(testval,"rand(denstens)",performancevals)

B = rand(C)
testval = "size(B) == size(C)"
fulltest &= testfct(testval,"rand(AbstractArray",performancevals)

println()

B = zeros(C)
testval = "size(B) == size(C)"
fulltest &= testfct(testval,"zeros(AbstractArray)",performancevals)

B = zeros(tA)
testval = "size(B) == size(tA)"
fulltest &= testfct(testval,"zeros(denstens)",performancevals)

B = zeros(ComplexF64,C)
testval = "size(B) == size(C) && eltype(B) == ComplexF64"
fulltest &= testfct(testval,"zeros(ComplexF64,AbstractArray)",performancevals)

B = zeros(ComplexF64,tA)
testval = "size(B) == size(tA) && eltype(B) == ComplexF64"
fulltest &= testfct(testval,"zeros(ComplexF64,denstens)",performancevals)

println()

B = zero(tA)
testval = "sum(B.T) == 0.0 && size(B) == size(tA)"
fulltest &= testfct(testval,"zero(denstens)",performancevals)

println()

ldim = round(Int64,rand(5:40,1)[1])
rdim = round(Int64,rand(5:40,1)[1])
println("  (ldim,rdim) = (",ldim,",",rdim,")")
A = eye(ComplexF64,ldim,rdim)
mindim = min(ldim,rdim)
testval = "size(A) == (ldim,rdim) && isapprox(sum(A),mindim) && eltype(A) == ComplexF64 && isapprox(Array(A[1:mindim,1:mindim]),zeros(mindim,mindim) + LinearAlgebra.I)"
fulltest &= testfct(testval,"eye",performancevals)

println()

A = eye(ComplexF64,ldim,ldim,addone=true,addRightDim=false)
testval = size(A) == (1,ldim,ldim) 
testval &= isapprox(sum(A),ldim) && eltype(A) == ComplexF64
testval &= isapprox(norm(A[1,1:mindim,1:mindim]),sqrt(mindim))
fulltest &= testfct(testval,"eye (square, left)")

A = eye(ComplexF64,ldim,rdim,addone=true,addRightDim=true)
testval = "size(A) == (ldim,rdim,1) && isapprox(sum(A),mindim) && eltype(A) == ComplexF64 && isapprox(Array(A[1:mindim,1:mindim,1]),zeros(mindim,mindim) + LinearAlgebra.I)"
fulltest &= testfct(testval,"eye (square, right)",performancevals)

A = eye(ComplexF64,ldim,rdim)
testval = "size(A) == (ldim,rdim) && isapprox(sum(A),mindim) && eltype(A) == ComplexF64 && isapprox(Array(A[1:mindim,1:mindim]),zeros(mindim,mindim) + LinearAlgebra.I)"
fulltest &= testfct(testval,"eye (rectangular)",performancevals)

println()
ldim,rdim = rdim,ldim
println("  (ldim,rdim) = (",ldim,",",rdim,")")
A = eye(ComplexF64,ldim,rdim)
mindim = min(ldim,rdim)
testval = size(A) == (ldim,rdim)
testval &= isapprox(sum(A),mindim)
testval &= eltype(A) == ComplexF64
testval &= isapprox(Array(A[1:mindim,1:mindim]),zeros(mindim,mindim) + LinearAlgebra.I)
fulltest &= testfct(testval,"eye")

println()

A = eye(ComplexF64,ldim,ldim,addone=true,addRightDim=false)
testval = size(A) == (1,ldim,ldim)
testval &= isapprox(sum(A),ldim) && eltype(A) == ComplexF64
testval &= isapprox(norm(A[1,1:mindim,1:mindim]),sqrt(mindim))
fulltest &= testfct(testval,"eye (square, left)")

A = eye(ComplexF64,ldim,rdim,addone=true,addRightDim=true)
testval = "size(A) == (ldim,rdim,1) && isapprox(sum(A),mindim) && eltype(A) == ComplexF64 && isapprox(Array(A[1:mindim,1:mindim,1]),zeros(mindim,mindim) + LinearAlgebra.I)"
fulltest &= testfct(testval,"eye (square, right)",performancevals)

A = eye(ComplexF64,ldim,rdim)
testval = "size(A) == (ldim,rdim) && isapprox(sum(A),mindim) && eltype(A) == ComplexF64 && isapprox(Array(A[1:mindim,1:mindim]),zeros(mindim,mindim) + LinearAlgebra.I)"
fulltest &= testfct(testval,"eye (rectangular)",performancevals)

A = rand(ldim,rdim,50,10)
B = eye(A,[1,2,3])

testval = "isapprox(sum(B),ldim*rdim*50) && size(B) == (ldim,ldim,rdim,rdim,50,50)"

fulltest &= testfct(testval,"eye (index inputs, [1,2,3])",performancevals)
#=
println()

B = convertTens(ComplexF64,tens(A))
testval = eltype(B) == ComplexF64 && size(B) == size(A) && isapprox(norm(B),norm(A)) && typeof(B) <: denstens
fulltest &= testfct(testval,"convertTens(denstens)")

B = convertTens(ComplexF64,A)
testval = eltype(B) == ComplexF64 && size(B) == size(A) && isapprox(norm(B),norm(A))
fulltest &= testfct(testval,"convertTens(Array)")
=#
println()

A = convIn([1,2,3])
testval = "typeof(A) <: Tuple && length(A) == 3 && norm([A[i]-i for i = 1:3]) == 0"
fulltest &= testfct(testval,"convIn(Array)",performancevals)

A = convIn([1;2;3])
testval = "typeof(A) <: Tuple && length(A) == 3 && norm([A[i]-i for i = 1:3]) == 0"
fulltest &= testfct(testval,"convIn(Matrix)",performancevals)

B = convIn(1)
testval = "typeof(B) <: Tuple"
fulltest &= testfct(testval,"convIn(Integer)",performancevals)

A = (1,2,3)
B = convIn(A)
testval = "A === B"
fulltest &= testfct(testval,"convIn(Tuple)",performancevals)

println()

nA = rand(9:100,1)[1]
vect = [rand(Bool) for i = 1:nA]
iA = Vector{Int64}(undef,sum(vect))
othervec = Vector{Int64}(undef,nA-length(iA))
counter = [0]
anticounter = [0]
for i = 1:nA
  if vect[i]
    counter[1] += 1
    iA[counter[1]] = i
  else
    anticounter[1] += 1
    othervec[anticounter[1]] = i
  end
end
B = findnotcons(nA,(iA...,))
testval = "length(B) == nA-length(iA) && othervec == B"
fulltest &= testfct(testval,"findnotcons",performancevals)

println()

A = Diagonal(rand(10))
B = tens(rand(ComplexF64,20,40,3))

C,D = checkType(A,B)
testval = "typeof(C) <: denstens && eltype(C) == ComplexF64 && typeof(D) <: denstens && eltype(D) == ComplexF64"
fulltest &= testfct(testval,"checkType(Diagonal,denstens)",performancevals)

C,D = checkType(B,A)
testval = "typeof(C) <: denstens && eltype(C) == ComplexF64 && typeof(D) <: denstens && eltype(D) == ComplexF64"
fulltest &= testfct(testval,"checkType(denstens,Diagonal)",performancevals)

C = checkType(A)
testval = "typeof(C) <: diagonal && isapprox(norm(A),norm(C))"
fulltest &= testfct(testval,"checkType(Diagonal)",performancevals)

C = checkType(B)
testval = "C === B"
fulltest &= testfct(testval,"checkType(tens)",performancevals)

A = tens(rand(20,3,40))
B = tens(rand(ComplexF64,20,40,3))
C,D = checkType(A,B)
testval = "typeof(C) <: denstens && eltype(C) == ComplexF64 && typeof(D) <: denstens && eltype(D) == ComplexF64"
fulltest &= testfct(testval,"checkType(denstens,denstens)",performancevals)

println()

pos = makepos(ndim)
B = vcat([0],ones(Int64,ndim-1))

testval = "pos == B"
fulltest &= testfct(testval,"makepos(Integer)",performancevals)

println()

S = ntuple(g->rand(5:10,1)[1],3)
vecS = [S[i] for i = 1:3]
currposvec = [makepos(3) for i = 1:prod(S)]

x = [i for i = 1:prod(S)]
y = Vector{intType}(undef,prod(S))

testval = [true]

for k = 1:length(currposvec)
  ind2pos!(currposvec,k,x,k,S)
  val = pos2ind((currposvec[k]...,),S)
  val2 = pos2ind((currposvec[k]...,),vecS)
  pos2ind!(y,k,currposvec[k],S)
  testval[1] &= val == x[k] == val2 && y[k] == x[k]
end
testval = currposvec[end] == vecS && testval[1]
fulltest &= testfct(testval,"ind2pos! && pos2ind(Vector) && pos2ind(Tuple) && pos2ind!")

println()

pos = makepos!(pos)
B = vcat([0],ones(Int64,ndim-1))
testval = "pos == B"
fulltest &= testfct(testval,"makepos!(Vector)",performancevals)

println()

S = (10,20,10,8,10,11)
cols = (:,[1,2,3,4],(1,2,3,4,5),1:5,1:5,1)
fullcols = get_denseranges(S,cols...)

testval = "fullcols == [1:10,[1,2,3,4],[1,2,3,4,5],1:5,1:5,1]"
fulltest &= testfct(testval,"get_denseranges(genColTypes)",performancevals)

println()

A = rand(ComplexF64,10,20)
tA = tens(A)
B = transpose(tA)

testval = "isapprox(norm(transpose(A) - Array(B)),0)"
fulltest &= testfct(testval,"adjoint",performancevals)

println()

A = rand(ComplexF64,10,20)
tA = tens(A)
B = adjoint(tA)

testval = "isapprox(norm(A' - Array(B)),0)"
fulltest &= testfct(testval,"adjoint",performancevals)

println()

Asize = [rand(3:10,1)[1] for i = 1:ndim]
A = tens(rand(Asize...))
B = copy(A)
testval = "!(A===B) && A.size == B.size && A.T == B.T && eltype(A) == eltype(B)"
fulltest &= testfct(testval,"copy(denstens)",performancevals)

println()

testval = "length(A) == prod(A.size)"
fulltest &= testfct(testval,"length(denstens)",performancevals)

println()

testval = "size(A) == (A.size...,)"
fulltest &= testfct(testval,"size(denstens)",performancevals)

testval = "ntuple(n->size(A,n),length(A.size)) == size(A)"
fulltest &= testfct(testval,"size(denstens,int)",performancevals)

println()

A = rand(ComplexF64,Asize...)
tA = tens(A)
sumA = sum(A)
testval = "isapprox(sumA,sum(tA))"
fulltest &= testfct(testval,"sum(denstens)",performancevals)

println()

testval = "isapprox(norm(A),norm(tA))"
fulltest &= testfct(testval,"norm(denstens)",performancevals)

println()

testval = "isapprox(sum(conj(tA)),sumA')"
fulltest &= testfct(testval,"conj(denstens)",performancevals)
testval = "isapprox(sum(conj!(tA)),sumA')"
fulltest &= testfct(testval,"conj!(denstens)",performancevals)

println()

testval = "length(tA.size) == ndims(tA)"
fulltest &= testfct(testval,"ndims(denstens)",performancevals)

println()

testval = "lastindex(tA,1) == size(tA,1)"
fulltest &= testfct(testval,"lastindex(denstens,i)",performancevals)

println()

testval = "eltype(tA) <: ComplexF64"
fulltest &= testfct(testval,"eltype(denstens)",performancevals)

println()

testval = "elnumtype(tA) <: ComplexF64"
fulltest &= testfct(testval,"enumtype(denstens)",performancevals)

println()

S = (10,20,10,8,10,11)
A = rand(S...)
tA = tens(A)


#println(A)

#println(norm(Array(tA[cols...])-A[fullcols...]))



testval = "isapprox(norm(Array(tA[cols...])-A[fullcols...]),0)" #Array(tA[cols...]) == A[fullcols...]
fulltest &= testfct(testval,"getindex(genColtype)",performancevals)

testval = "tA[S...] == A[S...]"
fulltest &= testfct(testval,"getindex(integer...)",performancevals)
#=
A = LinearAlgebra.Diagonal(rand(10))
testval = A[:,[1,2,3]] == A[:,1:3]
fulltest &= testfct(testval,"getindex()")
=#
println()

B = searchindex(tA,1,2,3,4,5,6)
testval = "B == searchindex(A,(1,2,3,4,5,6)) == A[1,2,3,4,5,6]"
fulltest &= testfct(testval,"searchindex(denstens,integer...)",performancevals)

println()

B = rand(2,2)
tB = tens(B)

C = copy(tA)
tA[1:2,[2,3],3,4,5,6] = tB
C[1:2,[2,3],3,4,5,6] = B
A[1:2,2:3,3,4,5,6] = tB
testval = isapprox(norm(B - Array(tA[1:2,2:3,3,4,5,6])),0)

testval &= isapprox(norm(B - Array(C[1:2,[2,3],3,4,5,6])),0)
testval &= isapprox(norm(B - A[1:2,2:3,3,4,5,6]),0)
fulltest &= testfct(testval,"setindex(denstens,integer...)")

println()

A = rand(100,100)
B = zeros(100,100)
loadM!(B,A)
testval = "A == B"
fulltest &= testfct(testval,"loadM!",performancevals)

println()

A = rand(10,20,30,10)
tA = tens(A)
B = rand(10,20,30,10)
tB = tens(B)
checkC = 2*A+2*B

testval = "Array(tensorcombination((2.,2.),tA,tB)) == tensorcombination(A,B,alpha=(2.,2.)) == checkC"
fulltest &= testfct(testval,"tensorcombination(denstens || Array)",performancevals)

println()

A = rand(10,20,30,10);
tA = tens(copy(A));
B = rand(10,20,30,10);
tB = tens(copy(B));
checkC = 2*A+2*B;
checktC = 2*tA+2*tB;

testtC = tensorcombination!((2.,2.),tA,tB)
testC = tensorcombination!(A,B,alpha=(2.,2.))

testval = "norm(tA - checktC) == 0 && norm(tA - testtC) == 0 && norm(checkC-testC) == 0"  # == A
fulltest &= testfct(testval,"tensorcombination!(denstens || Array)",performancevals)

println()

B = mult!(3,tA)
C = mult!(3,A)

testval = "norm(Array(B)-C) == 0"
fulltest &= testfct(testval,"mult!(denstens || Array)",performancevals)

println()

tA = tens(copy(A))
tC = tens(copy(C))
testval = "isapprox(norm(add!(copy(A),C,7) - (A+7*C)),0) == isapprox(norm(Array(add!(copy(tA),tC,7)) - (A+7*C)),0)"
fulltest &= testfct(testval,"add!(denstens || Array,number)",performancevals)

tA = tens(copy(A))
tC = tens(copy(C))
testval = "isapprox(norm(add!(copy(A),C) - (A+C)),0) == isapprox(norm(Array(add!(copy(tA),tC)) - (A+C)),0)"
fulltest &= testfct(testval,"add!(denstens || Array)",performancevals)

println()

tA = tens(copy(A))
tC = tens(copy(C))
testval = "isapprox(norm(sub!(copy(A),C,7) - A-7*C),0) == isapprox(norm(Array(sub!(copy(tA),tC,7)) - A-7*C),0)"
fulltest &= testfct(testval,"sub!(denstens || Array,number)",performancevals)

tA = tens(copy(A))
tC = tens(copy(C))
testval = "isapprox(norm(sub!(copy(A),C) - A-C),0) == isapprox(norm(Array(sub!(copy(tA),tC)) - A-C),0)"
fulltest &= testfct(testval,"sub!(denstens || Array)",performancevals)

println()

f = rand()
B = div!(copy(A),f)
C = A/f
testval = "isapprox(norm(B-C),0)"
fulltest &= testfct(testval,"div!(denstens || Array)",performancevals)

println()

A = norm!(A)
testval = "isapprox(norm(A),1)"
fulltest &= testfct(testval,"norm!(TensType)",performancevals)

println()

tA = tens(A)
tC = tens(C)
testval = "isapprox(Array(tA+tC),A+C) && isapprox(Array(A+tC),Array(tA+C))"
fulltest &= testfct(testval,"+(A,B))",performancevals)

println()

testval = "isapprox(Array(tA-tC),A-C) && isapprox(Array(A-tC),Array(tA-C))"
fulltest &= testfct(testval,"-(A,B))",performancevals)

println()

testval = "isapprox(Array(tA/8),Array((1/8)*tA))"
fulltest &= testfct(testval,"/(A,c)",performancevals)

println()

testval = "isapprox(sum(sqrt!(Diagonal(ones(ndim)/ndim))^2),1)"
fulltest &= testfct(testval,"sqrt!(Diagonal)",performancevals)

println()

testval = "isapprox(sum(sqrt(Diagonal(ones(ndim)/ndim))^2),1)"
fulltest &= testfct(testval,"sqrt(Diagonal)",performancevals)

println()

testval = "isapprox(sum(invmat(Diagonal(ones(ndim)/ndim))),ndim^2)"
fulltest &= testfct(testval,"invmat(Diagonal)",performancevals)

testval = "isapprox(sum(invmat(tens(Diagonal(ones(ndim)/ndim)))),ndim^2)"
fulltest &= testfct(testval,"invmat(denstens)",performancevals)

testval = "isapprox(sum(invmat(Array(Diagonal(ones(ndim)/ndim)))),ndim^2)"
fulltest &= testfct(testval,"invmat(Array)",performancevals)

println()

A = rand(10,10)
B = exp(A)
testval = "isapprox(exp(copy(A)),B)"
fulltest &= testfct(testval,"exp(Array)",performancevals)

testval = "isapprox(Array(exp(tens(copy(A)))),B)"
fulltest &= testfct(testval,"exp(denstens)",performancevals)

A = rand(10,10)
B = exp(2*A)
testval = "isapprox(exp!(copy(A),2),B)"
fulltest &= testfct(testval,"exp(Array,prefactor)",performancevals)

testval = "isapprox(exp(A,2),B)"
fulltest &= testfct(testval,"exp(denstens,prefactor)",performancevals)

println()

alpha = rand(10)
beta = rand(9)
testval = "isapprox(exp(2*LinearAlgebra.SymTridiagonal(alpha,beta)),exp(alpha,beta,2))"
fulltest &= testfct(testval,"exp(alpha,beta,prefactor)",performancevals)

testval = "isapprox(exp(LinearAlgebra.SymTridiagonal(alpha,beta)),exp(alpha,beta))"
fulltest &= testfct(testval,"exp(alpha,beta)",performancevals)

testval = "isapprox(exp(2*LinearAlgebra.SymTridiagonal(alpha,beta)),exp(LinearAlgebra.SymTridiagonal(alpha,beta),2))"
fulltest &= testfct(testval,"exp(alpha,beta,prefactor)",performancevals)

println()

A = rand(20,40,30,50)

rA = reshape(A,20,1200,50)
testval = "isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape(Array,integer)",performancevals)

rA = reshape(A,(20,1200,50))
testval = "isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape(Array,tuple)",performancevals)

rA = reshape(A,[20,1200,50])
testval = "isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape(Array,Vector)",performancevals)

tA = tens(rand(20,40,30,50))

rA = reshape(tA,20,1200,50)
testval = "isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape(tens,integer)",performancevals)

rA = reshape(tA,(20,1200,50))
testval = "isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape(tens,tuple)",performancevals)

rA = reshape(tA,[20,1200,50])
testval = "isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape(tens,Vector)",performancevals)

rA = reshape(copy(A),20,1200,50)
testval = "isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape(Array,integer)",performancevals)

rA = reshape(copy(A),(20,1200,50))
testval = "isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape(Array,tuple)",performancevals)

rA = reshape(copy(A),[20,1200,50])
testval = "isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape(Array,Vector)",performancevals)

rA = reshape(copy(tA),[[1],[2,3],[4]])
testval = "isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape(tens,[[indexes]])",performancevals)



tA = tens(rand(20,40,30,50))

rA = reshape(copy(tA),20,1200,50)
testval = "isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape(tens,integer)",performancevals)

rA = reshape(copy(tA),(20,1200,50))
testval = "isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape!(tens,tuple)",performancevals)

rA = reshape!(copy(tA),[20,1200,50])
testval = "isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape!(tens,Vector)",performancevals)

rA = reshape!(copy(tA),[[1],[2,3],[4]])
testval = "isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)"
fulltest &= testfct(testval,"reshape!(tens,[[indexes]])",performancevals)

println()

rA = unreshape(copy(A),20,1200,50)
testval = "isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)"
fulltest &= testfct(testval,"unreshape(Array,Vector)",performancevals)

rA = unreshape(copy(tA),20,1200,50)
testval = "isapprox(norm(rA.T),norm(tA.T)) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)"
fulltest &= testfct(testval,"unreshape(tens,Vector)",performancevals)

rA = unreshape!(copy(A),20,1200,50)
testval = "isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)"
fulltest &= testfct(testval,"unreshape!(Array,Vector)",performancevals)

rA = unreshape!(copy(tA),20,1200,50)
testval = "isapprox(norm(rA.T),norm(tA.T)) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)"
fulltest &= testfct(testval,"unreshape!(tens,Vector)",performancevals)

println()

permorder = [1,4,3,2]
B = permutedims!(copy(tA),permorder)
testval = "isapprox(sum([size(B,i) - size(tA,permorder[i]) for i = 1:length(permorder)]),0) && isapprox(norm(tA),norm(B))"
fulltest &= testfct(testval,"permutedims!(tens,Vector)",performancevals)

B = permutedims!(copy(tA),(permorder...,))
testval = "isapprox(sum([size(B,i) - size(tA,permorder[i]) for i = 1:length(permorder)]),0) && isapprox(norm(tA),norm(B))"
fulltest &= testfct(testval,"permutedims!(tens,Tuple)",performancevals)

C = Diagonal(rand(10))

permorder = [2,1]
B = permutedims!(copy(C),permorder)
testval = "isapprox(sum([size(B,i) - size(C,permorder[i]) for i = 1:length(permorder)]),0) && isapprox(norm(C),norm(B))"
fulltest &= testfct(testval,"permutedims!(Diagonal,Vector)",performancevals)

B = permutedims!(copy(C),(permorder...,))
testval = "isapprox(sum([size(B,i) - size(C,permorder[i]) for i = 1:length(permorder)]),0) && isapprox(norm(C),norm(B))"
fulltest &= testfct(testval,"permutedims!(Diagonal,Tuple)",performancevals)


permorder = [1,4,3,2]
B = permutedims(copy(tA),permorder)
testval = "isapprox(sum([size(B,i) - size(tA,permorder[i]) for i = 1:length(permorder)]),0) && isapprox(norm(tA),norm(B))"
fulltest &= testfct(testval,"permutedims(tens,Vector)",performancevals)

B = permutedims(copy(tA),(permorder...,))
testval = "isapprox(sum([size(B,i) - size(tA,permorder[i]) for i = 1:length(permorder)]),0) && isapprox(norm(tA),norm(B))"
fulltest &= testfct(testval,"permutedims(tens,Tuple)",performancevals)

C = Diagonal(rand(10))

permorder = [2,1]
B = permutedims(copy(C),permorder)
testval = "isapprox(sum([size(B,i) - size(C,permorder[i]) for i = 1:length(permorder)]),0) && isapprox(norm(C),norm(B))"
fulltest &= testfct(testval,"permutedims(Diagonal,Vector)",performancevals)

B = permutedims(copy(C),(permorder...,))
testval = "isapprox(sum([size(B,i) - size(C,permorder[i]) for i = 1:length(permorder)]),0) && isapprox(norm(C),norm(B))"
fulltest &= testfct(testval,"permutedims(Diagonal,Tuple)",performancevals)

println()

A = rand(30,20,10)
B = rand(10,20,10)

C = joinindex(A,B,1)
testval = "isapprox(sum(A)+sum(B),sum(C)) && size(C) == (40,20,10)"
fulltest &= testfct(testval,"joinindex(Array,integer)",performancevals)

C = joinindex(A,B,[1,2])
testval = "isapprox(sum(A)+sum(B),sum(C)) && size(C) == (40,40,10)"
fulltest &= testfct(testval,"joinindex(Array,Array)",performancevals)

tA = tens(A)
tB = tens(B)

C = joinindex(tA,tB,1)
testval = "isapprox(sum(tA)+sum(tB),sum(C)) && size(C) == (40,20,10)"
fulltest &= testfct(testval,"joinindex(tens,integer)",performancevals)

C = joinindex(tA,tB,[1,2])
testval = "isapprox(sum(tA)+sum(tB),sum(C)) && size(C) == (40,40,10)"
fulltest &= testfct(testval,"joinindex(tens,Array)",performancevals)



Serialization.serialize(file,performancevals)

