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

# rs: row start
# rl: row length
# cs: column start
# cl: column length
struct Block
  rs::intType
  rl::intType
  cs::intType
  cl::intType
end

"""
    submats = find_submatrices(transA,A,n)

Find and return a list of starting indices and sizes of sub-blocks in a matrix A where n defines the number of sub-blocks to be 2^n by 2^n. 

#Inputs:
+ `transA`: `AbstractChar` for whether to transpose input matrices
+ `A`: Input matrix
+ `n`: level of the Strassen algorithm
"""
function find_submatrices(transA::AbstractChar,A::Matrix{W}, n::intType) where W <: Number

  sqsize = 2^n
  if transA == 'N'
    total_rows = size(A,1)
    total_cols = size(A,2)
  else
    total_rows = size(A,2)
    total_cols = size(A,1)
  end

#  total_rows = size(A,1)
#  total_cols = size(A,2)

  base_row = cld(total_rows, sqsize)
  base_col = cld(total_cols, sqsize)



  subblock_indices = Array{Block,1}(undef,sqsize^2)

  if transA != 'N'
    nlevel = n
    trule = reshape([i for i = 1:4^nlevel],2^(nlevel),2^(nlevel))'
  end


  #=Threads.@threads=# for pos = 1:sqsize^2
    if transA == 'N'
      currpos = pos
    else
      currpos = trule[pos]
    end

    i = cld(currpos, sqsize)
    j = (currpos-1) % sqsize + 1

    col_start = 1 + base_col * (j-1)
    col_end = base_col * j

    row_start = 1 + base_row * (i-1)
    row_end = base_row * i

    # Prevent array overflow for rectangular case
    row_end = (row_end > total_rows) ?
               total_rows : row_end
		col_end = (col_end > total_cols) ?
							 total_cols : col_end


    row_len = row_end - row_start + 1
    col_len = col_end - col_start + 1

		subblock_indices[currpos] = transA == 'N' ? Block(row_start, row_len, col_start, col_len) : Block(col_start, col_len, row_start, row_len)
  end

  return subblock_indices
end


"""
    C = add_subblock(M,A,subblock,signA)

Add a subblock of A to matrix M.

#Inputs:
+ `C`: pre-initialized matrix to add to
+ `A`: matrix to add elements from
+ `subblock`: sub-block start index and size
+ `signA`: the signs associated with A
"""
function add_subblock(M::Matrix{W},A::Matrix{W},subblock::Block,signA::Number) where W <: Number

  row_start = subblock.rs
  col_start = subblock.cs

  if signA == 1
    for j in 1:subblock.cl
      @inbounds @simd for i in 1:subblock.rl
      M[i,j] += A[i + row_start - 1,
                  j + col_start - 1] #*signA
      end
    end
  else
    for j in 1:subblock.cl
      @inbounds @simd for i in 1:subblock.rl
      M[i,j] -= A[i + row_start - 1,
                  j + col_start - 1] #*signA
      end
    end
  end

  return M
end


"""
    C = add_component_matrices!(C,A,signA)

Add matrix A to matrix C.

#Inputs:
+ `C`: pre-initialized matrix to store result in
+ `A`: matrix to add elements from
+ `signA`: the signs associated with A
"""
function add_component_matrices!(C::Matrix{W},A::Matrix{W},signA::Number) where W <: Number

  rowA = size(A,1)
  colA = size(A,2)

  if isapprox(signA,1)
    for j in 1:colA
      @inbounds @simd for i in 1:rowA
        C[i,j] += A[i,j] #*signA
      end
    end
  else
    for j in 1:colA
      @inbounds @simd for i in 1:rowA
        C[i,j] -= A[i,j]
      end
    end
  end
  return C
end


"""
    M = add_submatrices(M,A,subblocks,combo,signs)

Add sub-blocks of A together.

#Inputs:
+ `M`: pre-initialized matrix to store result in
+ `A`: matrix to add elements from
+ `subblocks`: Starting indices and sizes of sub-block matrices of A
+ `combo`: the combination of submatrices to add
+ `signs`: the signs associated with each submatrix
"""
function add_submatrices(M::Matrix{W},A::Matrix{W},subblocks::Vector{Block},combo::Vector{intType},signs::Vector{intType}) where W <: Number

  row_start = subblocks[combo[1]].rs
  col_start = subblocks[combo[1]].cs

  # Populate M with the first submatrix.
  # Using an if statement seems to be more efficient than multiplying by the sign (for some reason)
  if signs[1] == 1
    for j in 1 : subblocks[combo[1]].cl
      @inbounds @simd for i in 1 : subblocks[combo[1]].rl
        M[i, j] = A[row_start + i - 1, col_start + j - 1] # *signs[1]
      end
    end
  else
    for j in 1 : subblocks[combo[1]].cl
      @inbounds @simd for i in 1 : subblocks[combo[1]].rl
        M[i,j] = -A[row_start + i - 1, col_start + j - 1] # *signs[1]
      end
    end
  end

  # Padding
  for j in ( subblocks[combo[1]].cl + 1 ) : size(M, 2)
    @inbounds @simd for i = 1:size(M,1)
      M[i,j] = 0
    end
  end

  for j in ( subblocks[combo[1]].rl + 1  ) : size(M, 1)
    @inbounds @simd for i = 1:size(M,2)
      M[j,i] = 0
    end
  end

  @inbounds for i in 2:length(combo)
    M = add_subblock(M, A, subblocks[combo[i]], W(signs[i]))
  end

  return M
end

"""
    M = calc_strassen_components(transA,transB,alpha,A,B,A_subblocks,B_subblocks,n)

Calculate the 7^n component matrices which are combined to form the resultant matrix in the Strassen algorithm.

#Inputs:
+ `transA`: `AbstractChar` for whether to transpose A matrix
+ `transB`: `AbstractChar` for whether to transpose B matrix
+ `alpha`: `Number` scalar multiplier
+ `A`: Input matrix
+ `B`: Input matrix
+ `A_subblocks`: Starting indices and sizes of sub-block matrices of A
+ `B_subblocks`: Starting indices and sizes of sub-block matrices of B
+ `n`: level of the Strassen algorithm
"""
function calc_strassen_components(transA::AbstractChar,transB::AbstractChar,alpha::W,A::Matrix{W},B::Matrix{W},A_subblocks::Vector{Block},B_subblocks::Vector{Block},n::Int64) where W <: Number

  components = Array{Matrix{W}}(undef, 7^n)

  row_len_A = cld(size(A, 1), 2^n)
  col_len_A = cld(size(A, 2), 2^n)
  row_len_B = cld(size(B, 1), 2^n)
  col_len_B = cld(size(B, 2), 2^n)

  # Arrays of matrices to store addition results
  nthreads = Threads.nthreads()
  Astore = [
    Array{W,2}(undef, row_len_A, col_len_A) for i = 1:nthreads
  ]
  Bstore = [
    Array{W,2}(undef, row_len_B, col_len_B) for i = 1:nthreads
  ]

  Threads.@threads for i = 1:7^n
    thisthread = Threads.threadid()

    # Add up submatrices according to A/B_sub_idxs with signs given by A/B_sub_signs.
    # Pass in full matrices A and B and index according to the subblock indices.
    newA = add_submatrices(Astore[thisthread], A, A_subblocks, A_sub_idxs[i], A_sub_signs[i])
    newB = add_submatrices(Bstore[thisthread], B, B_subblocks, B_sub_idxs[i], B_sub_signs[i])

    if transA == 'N'
      Lsize = size(newA,1)
      innersizeL = size(newA,2)
    else
      Lsize = size(newA,2)
      innersizeL = size(newA,1)
    end
  
    if transB == 'N'
      innersizeR = size(newB,1)
      Rsize = size(newB,2)
    else
      innersizeR = size(newB,2)
      Rsize = size(newB,1)
    end

    components[i] = matmul(transA,transB,alpha,newA,newB,Lsize,innersizeL,innersizeR,Rsize)
  end

  return components
end

"""
    C = add_strassen_components(components, k)

Combine 7 of the component matrices obtained in a previous step of the Strassen algorithm.

#Inputs:
+ `components`: component matrices to combine
+ `k`: total number of component matrices
"""
function add_strassen_components(components::Vector{Matrix{W}}, k::Number) where W <: Number
#  W = eltype(components[1])

  #=Note: it seems that components[i+2] and components[i+4] seem to be smaller than the other matrices in terms of columns. So, the result of the additions may be added in place... =#
  #=Threads.@threads=# for i in 1:7:(k-1)
    T1 = add_component_matrices!(components[i+6], components[i], W(1))
    T1 = add_component_matrices!(T1, components[i+4], W(-1))
    T1 = add_component_matrices!(T1, components[i+3], W(1))

    T4 = add_component_matrices!(components[i+5], components[i+2],W(1))
    T4 = add_component_matrices!(T4 ,components[i+1], W(-1))
    T4 = add_component_matrices!(T4, components[i], W(1))


    T2 = add_component_matrices!(components[i+4], components[i+2], W(1))
    T3 = add_component_matrices!(components[i+1], components[i+3], W(1))

    if size(T4) != (size(T3,1),size(T2,2))
      T4 = T4[1:size(T3,1), 1:size(T2,2)]
    end

    components[(i+6)÷7] = [T1 T2; T3 T4]
  end
	nothing
end

"""
    C = calc_strassen_result(components, n)

Combine the 7^n component matrices obtained in a previous step of the Strassen algorithm in groups of 7 to obtain 7^n-1 component matrices after each internal iteration. Return the resultatnt matrix of the matrix multiplication computation.

#Inputs:
+ `components`: component matrices to combine to give final Strassen result
+ `n`: level of the Strassen algorithm
"""
function calc_strassen_result(components::Vector{Matrix{W}}, n::Number) where W <: Number
  k = 7^n
  while n > 0
    add_strassen_components(components, k)

    k ÷= 7
    n -= 1
  end
  return components[1]
end

"""
    C = strassen(transA, transB, alpha, A, B[,n=])

Perform Strassen matrix multiplication on matrices A and B giving resulant matrix C. 

#Inputs:
+ `transA`: `AbstractChar` for whether to transpose A matrix
+ `transB`: `AbstractChar` for whether to transpose B matrix
+ `alpha`: `Number` scalar multiplier
+ `A`: Input matrix
+ `B`: Input matrix
+ `n`: level of the Strassen algorithm
"""
function strassen(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::Matrix{W}, B::Matrix{W}; n::intType=2) where W <: Number


  A_rules, B_rules = strassOPen(n)
  global A_sub_idxs, A_sub_signs = A_rules[1], A_rules[2]
  global B_sub_idxs, B_sub_signs = B_rules[1], B_rules[2]

  A_subblocks = find_submatrices(transA, A, n)
  B_subblocks = find_submatrices(transB, B, n)

  if typeof(alpha) == W
    newalpha = alpha
  else
    newalpha = convert(W,alpha)
  end

  components = calc_strassen_components(transA, transB, newalpha, A, B, A_subblocks, B_subblocks, n)

  return calc_strassen_result(components, n)
end

function strassen(transA::AbstractChar,transB::AbstractChar,A::Matrix{W}, B::Matrix{W}; n::intType=2) where W <: Number
  return strassen(transA,transB,W(1),A,B)
end

"""
    C = strassen(A, B[,n=])

Perform Strassen matrix multiplication on matrices A and B giving resulant matrix C.

#Inputs:
+ `A`: Input matrix
+ `B`: Input matrix
+ `n`: level of the Strassen algorithm
"""
function strassen(A::TensType,B::TensType;n::Int64=2)
  W = typeof(eltype(A)(1)*eltype(B)(1))
  return strassen('N','N',W(1),A,B,n=n)
end
export strassen

#=
function strassen(transA::AbstractChar,transB::AbstractChar,alpha::Number,A::TensType,B::TensType,Lsize::Integer,innerAsize::Integer,innerBsize::Integer,Rsize::Integer;n::Integer=2)
  expect_row = transA == 'N' ? size(A,1) : size(A,2)
  expect_col = transB == 'N' ? size(B,2) : size(B,1)

  A_sub = find_submatrices(transA,A,n)
  B_sub = find_submatrices(transB,B,n)

  combinations = calc_combinations(transA,transB,alpha,A_sub, B_sub,n)

  answer = next_level_tester(combinations,n)

  if size(answer) != (expect_row,expect_col)
    answer = answer[1:expect_row, 1:expect_col]
  end

  return answer
end

function strassen(A::TensType,B::TensType;n::Int64=2)
  W = typeof(eltype(A)(1)*eltype(B)(1))
  return strassen('N','N',W(1),A,B,size(A,1),size(A,2),size(B,1),size(B,2))
end

function strassen(transA::AbstractChar,transB::AbstractChar,A::TensType,B::TensType;n::Int64=2)
  W = typeof(eltype(A)(1)*eltype(B)(1))
  return strassen(transA,transB,W(1),A,B,size(A,1),size(A,2),size(B,1),size(B,2),n=n)
end
export strassen
=#