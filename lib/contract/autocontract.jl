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
    Indices

Names of indices on each site (`.names`) and dimensions of each site (`.dimensions`). Stores meta-data for contraction cost estimation in automatic contraction.

# Fields:
+ `names::Array{String,1}`: names of the indices of a tensor
+ `dimensions:NTuple{G,intType} where G`: dimension of the indices of a tensor
"""
struct Indices
	names::Array{String,1}
	dimensions::Array{intType,1} #NTuple{G,intType} where G
end

"""
    contract!(graph[,exclude=intType[]])

Contracts a tensor network into a single tensor. The values in `exclude` are the index positions of tensors to not contract. 
The algorithm will call either `greedy_contract` or `local_contract` depending on the network input.
"""
function contract!(graph::network; exclude::Vector{intType} = intType[])
	shared_edges, num_connections = makeNeighbourTable(graph, exclude=exclude)
	if ((num_connections+1) < (length(graph.net)-length(exclude)))
		error("The given network is disjoint (note: will remain error message to indicate when graphs are disjoint; if intended then contract them individually, but this flag will remain to notify)")
	elseif length(graph) == 2
		answer = graph[1]*graph[2]
  elseif ((length(graph.net)-num_connections)==1) # checks if the graph is a forest
		answer = greedy_contract(graph,shared_edges,exclude = exclude)
	else
		answer = local_contract(graph, shared_edges,exclude = exclude)
	end
	return answer
end

"""
    contract!(graph[,exclude=intType[]])

Contracts a array of tensors into a single tensor. The values in `exclude` are the index positions of tensors to not contract. 
The algorithm will call either `greedy_contract` or `local_contract` depending on the network input.
"""
function contract!(tensors::Vector{W}; exclude::Vector{intType} = intType[]) where W <: nametens
	graph = network(tensors)
	return contract!(graph, exclude = exclude)
end

"""
    contract(graph,[,exclude=intType[]])

Contracts a tensor network into a single tensor. The values in `exclude` are the index positions of tensors to not contract. 
The algorithm will call either `greedy_contract` or `local_contract` depending on the network input.
"""
function contract(tensors::W...) where W <: TNobj
	return contract(network(tensors...))
end

"""
    contract(graph[,exclude=[]])

contracts a network of tensors `graph` and excludes all tensors (ordered sequentially) in `exclude`
"""
function contract(graph::network; exclude::Vector{intType} = intType[])
	temp_graph = copy(graph)
	return contract!(temp_graph,exclude=exclude)
end

"""
    contract(tensors[,exclude=[]])

contracts an array of tensors `tensors` and excludes all tensors (ordered sequentially) in `exclude`
"""
function contract(tensors::Vector{W}; exclude::Vector{intType} = intType[]) where W <: nametens
	graph = network(tensors)
	return contract(graph, exclude = exclude)
end


"""
    getnames(X)

Gets the names of the indices of a `nametens` tensor.
"""
function getnames(X::TNobj)
  return X.names
end

# BOTH
"""
    makeNeighbourTable(G,[,exclude=intType[]])

Returns a dictionary where the index names are they keys and the values are the tensors connected to the indices. 
The values in `exclude` are the index positions of tensors to not include in the dictionary.
"""
function makeNeighbourTable(graph::network; exclude::Vector{intType} = intType[])
  shared_edges = Dict{String, Array{intType,1}}() # this takes 4 allocations
  tensors = graph.net
  num_connections = 0

  # iterates through each tensor in the network
  for pos in 1:length(tensors)
  	# this is a type instability!!! and changing the below line of current_tensor.names to tensors[pos].names is also unstable

  	if !(pos in exclude)
	  	current_tensor = tensors[pos]

	  	# adds the names of the indices
	    for edge in getnames(current_tensor)
	    	# if the index is already in the dictionary then the tensor is connected to another tensor
	      if edge in keys(shared_edges)
	        shared_edges[edge][2] = pos
	        num_connections += 1
	        # if the index is not in the dictionary then the index is free or the corresponding connected tensor is later in the list
	      else
	        shared_edges[edge] = Array{intType,1}(undef, 2) # this takes 1 allocation
	        shared_edges[edge][1] = pos
	        shared_edges[edge][2] = pos
	      end
	    end
	  end
  end
  return (shared_edges,num_connections)
end




"""
    get_cost(E)

Returns the product of all the dimensions of a tensor for given edges `E` (Vector)
"""
function get_cost(edges::Union{Memory{intType},Array{intType,1}}) #where G 
	cost = 1
	for edge_dim in edges
		cost *= edge_dim
	end
	return cost
end 


"""
    permute_cost(left,right,pos_left,pos_right)

Returns the total cost of any permutations due to contracting tensors. `left` and `right` correspond to the indices on the left and right tensors respectively.
'pos_left' and 'pos_right' correspond to the position of the contracted indices on the left and right tensor.
"""
function permute_cost(left::Indices, right::Indices, pos_left::Vector{intType}, pos_right::Vector{intType}) 

	costA,posA = permute(left, pos_left)
	costB,posB = permute(right, pos_right)

	if !(costA==0)&&!(costB==0)
		return costA+costB
	elseif (costA==0)&&(costB==0)
		return check_permute(left, right, posA)
	else
		return costA+costB
	end
end


"""
    permute(E,position)

Finds the cost of the permutation for a tensor that is being contracted according to `position`. If no permutation is needed then 0 is returned.
"""
function permute(edges::Indices, position::Array{W,1}) where W <: Integer
	cost = 1
	sort!(position)

	if !(length(edges.names) in position) && !(1 in position)
		for edge_dim in edges.dimensions
			cost *= edge_dim
		end
		return cost,position
	else
		for pos in range(2, length(position))
			if (position[pos]-position[pos-1]) != 1
				for edge_dim in edges.dimensions
					cost *= edge_dim
				end
				return cost,position
			end
		end
	end
	return 0,position 
end


"""
    check_permute(left,right,left_order)

Checks if a tensor needs to be permuted from input `Indices`. `left` and `right` are the indices of the tensors to contract and `left_order`
corresponds to the position of the contracted indices on the left tensor.
"""
function check_permute(left::Indices, right::Indices, left_order::Vector{intType})
	last_pos = -1
	cost = 1

	for pos in left_order
		edge = left.names[pos]
		index = findfirst(==(edge), right.names)
		if !(index>last_pos)
			for edge_dim in left.dimensions
				cost *= edge_dim
				
			end

			cost = cost÷length(left.names)
			for edge_dim in right.dimensions
				cost *= edge_dim
			end

			cost = cost÷length(right.names)
			return cost
		else
			last_pos = index
		end
	end
	return 0 #if nothing in the loop
end



# LOCAL

"""
    find_start(G)

Finds the starting tensor to contract onto from a graph `G`
"""
function find_start(graph::network)
	tensors = graph.net
	start_pos = 1 # the initial guess

	min_num_ind = effective_rank(size(tensors[start_pos]))
	min_cost = get_cost(size(tensors[start_pos]))

	for pos = 1:length(tensors)
		current_tensor = tensors[pos]

		indices = size(current_tensor)
		num_ind = effective_rank(indices)
		cost = get_cost(indices)

		if num_ind < min_num_ind
			start_pos = pos
			min_num_ind = num_ind
			min_cost = cost

		elseif (num_ind == min_num_ind)
			if (cost < min_cost)
				start_pos = pos
				min_num_ind = num_ind
				min_cost = cost
			end
		end
	end

	return start_pos
end


# The functions below are used in Permute_alg

"""
    effective_rank(dimensions)

Obtains the rank of a tensor(does not count indices with dimension of 1)
"""
function effective_rank(dimensions::Union{Memory{intType},Array{intType,1}}) #where G
	nrank = 0
	# does not consider dimension of value 1 to contribute to the rank
	for i in dimensions
		if i != 1
			nrank +=1 
		end
	end
	return nrank
end


"""
    FUTURE_START (=3)

How many tensors to look into future contractions for `local_contract`
"""
const FUTURE_START = 3

"""
    local_contract(graph,shared_edges,[,exclude=intType[]])

Contracts a tensor network into a single tensor. The values in `exclude` are the index positions of tensors to not contract. 
"""
function local_contract(graph::network, shared_edges::Dict{String, Vector{intType}}; exclude::Vector{intType} = intType[])
  num = length(graph)-length(exclude)
  contract_around = false # this should be false at the beginning of the algorithm

  start_pos = find_start(graph) # returns the position of start
  shared_edges = update_edges(getnames(graph.net[start_pos]),start_pos,shared_edges)	
  num -= 1

  start = graph.net[start_pos]

  initial_start = Indices(getnames(graph.net[start_pos]), size(graph.net[start_pos])) # need this
  future = FUTURE_START # how many tensors we will contract later
  next_tensors = Vector{intType}(undef, future) # the position of the tensors to contract next
  num_connecting = Vector{intType}(undef, future) # the number of connecting indices

  start_names = getnames(graph.net[start_pos])
  start_dimensions = size(graph.net[start_pos])

  while (num > 0)
    temp_start = Indices(start_names, start_dimensions)
    future = FUTURE_START

    if ((num)<future)
      future = num
      next_tensors = Vector{intType}(undef, future)
      num_connecting = Vector{intType}(undef, future)

    end

    while (num>0)&&(future>0)

      if (contract_around)

        next_details = find_next(temp_start, start_pos, graph,shared_edges) 
        next_pos = next_details[1]
        # this means that the network is disjoint
        if next_pos == start_pos
          error("The given network is disjoint")
        end
        temp_next = Indices(getnames(graph.net[next_pos]), size(graph.net[next_pos]))
        shared_edges = update_edges(getnames(graph.net[next_pos]),next_pos, shared_edges)

        next_tensors[length(next_tensors)-future+1] = next_pos
        num_connecting[length(next_tensors)-future+1] = next_details[2]

        temp_start = update_temp(temp_start, temp_next, next_details[2]) 
        future -= 1
        num -= 1

      else
        next_details = find_next(initial_start, start_pos,graph,shared_edges)
        next_pos = next_details[1]

        if (next_pos==start_pos)
          contract_around = true

        else
          temp_next = Indices(getnames(graph.net[next_pos]), size(graph.net[next_pos]))
          shared_edges = update_edges(getnames(graph.net[next_pos]),next_pos, shared_edges)

          num_common = common_info(temp_start, next_pos,graph)[1]

          next_tensors[length(next_tensors)-future+1] = next_pos
          num_connecting[length(next_tensors)-future+1] = num_common

          temp_start = update_temp(temp_start, temp_next, num_common)
          future -= 1
          num -=1
        end
      end
    end

    current_start = Indices(start_names, start_dimensions) # the current tensor we started with 
    order = best_order(next_tensors, num_connecting, current_start,graph)

    for pos = 1:length(order)
			next_tensor = graph.net[next_tensors[pos]]
			if order[pos] == 'L'
				start = next_tensor*start
			else 
				start = start*next_tensor
			end
		end

		start_names = getnames(start)
  	start_dimensions = size(start)

  end

  return start
end


"""
    find_next(E,start_pos,G, shared_edges)

Finds the next tensor to contract given `E`, the tensor located at `start_pos`, `G`, and shared edges (Dictionary) `shared_edges`
"""
function find_next(temp_start::Indices, start_pos::intType, graph::network, shared_edges::Dict{String, Vector{intType}})
	new_edges = -1
	next_pos = start_pos
	min_cost = -1
	max_common = 0

	for edge in temp_start.names
			if (length(shared_edges[edge])==1)
				right_tensor_pos = shared_edges[edge][1]
				right_tensor_indices = size(graph.net[right_tensor_pos])

				common_details = common_info(temp_start, right_tensor_pos,graph)

				num_common = common_details[1]
				added_edges = effective_rank(right_tensor_indices)-num_common
				cost_tot = (get_cost(temp_start.dimensions)*get_cost(right_tensor_indices))÷common_details[2] 

				if (min_cost==-1)
					new_edges = added_edges
					min_cost = cost_tot
					max_common = num_common
					next_pos = right_tensor_pos

				elseif (num_common>max_common)
					new_edges = added_edges
					min_cost = cost_tot
					max_common = num_common
					next_pos = right_tensor_pos


				elseif (num_common==max_common)
					if (added_edges<new_edges)
						new_edges = added_edges
						min_cost = cost_tot
						max_common = num_common
						next_pos = right_tensor_pos


					elseif (added_edges==new_edges)
						if (cost_tot<min_cost)
							new_edges = added_edges
							min_cost = cost_tot
							max_common = num_common
							next_pos = right_tensor_pos
	
						end
					end
				end
			end
	end

	return (next_pos, max_common)
end


"""
    common_info(E,right_tensor_pos,G)

Returns the number of edges in common and the product of the dimensions of the common vertices of `E` and the tensor found at position `right_tensor_pos`
"""
function common_info(temp_start::Indices, right_tensor_pos::intType, graph::network)  
	num_common = 0
	mult_commom = 1

	right_tensor_names = getnames(graph.net[right_tensor_pos])

	for (pos, edge_name) in enumerate(temp_start.names)
		if edge_name in right_tensor_names 
			num_common += 1
			mult_commom *= temp_start.dimensions[pos]
		end
	end

	return (num_common, mult_commom)
end


"""
    update_temp(left,right,max_common)

Updates the temp starting tensor to include the next contracted tensor for `left`, `right`, and `max_common` which is the number of edges in common between the two vertices
"""
function update_temp(left::Indices, right::Indices, max_common::intType) 
	new_length = (length(left.names)+length(right.names))-(2*max_common)

	new_names = Vector{String}(undef, new_length) 
	new_dimensions = Vector{intType}(undef, new_length) 

	num_updated = 1
	for pos in 1:length(left.names)
		if !(left.names[pos] in right.names)
			new_names[num_updated] = left.names[pos]
			new_dimensions[num_updated] = left.dimensions[pos]

			num_updated += 1
		end
	end

	for pos in 1:length(right.names)
		if !(right.names[pos] in left.names)
			new_names[num_updated] = right.names[pos]
			new_dimensions[num_updated] = right.dimensions[pos]

			num_updated += 1
		end
	end

	return Indices(new_names, new_dimensions)
end

"""
    best_order(next_tensors,num_connecting,temp_start,G)

Finds the best order to contract tensors to reduce the cost of permutations. 
`next_tensors` are the position of the tensors to contract next. `num_connecting` is the number of edges connecting to `temp_start`
"""
function best_order(next_tensors::Vector{intType}, num_connecting::Vector{intType}, temp_start::Indices, graph::network)
	temp = graph.net[next_tensors[1]]
	temp_next = Indices(getnames(temp), size(temp))

	left = best_order_helper(next_tensors, num_connecting, temp_start, graph, 0, "", "left", 1)
	right = best_order_helper(next_tensors, num_connecting, temp_start, graph, 0, "", "right", 1)

	if left[1]<=right[1]
		out = left[2]
	else
		out = right[2]
	end
  return out
end


"""
    best_order_helper(next_tensors,num_connecting,temp_start,cost,order,side,depth)

Finds the best order to contract tensors of a given network input `next_tensors` (vector of integers specifying the position in the network). 

#Other inputs:
+ `next_tensors` are the position of the tensors to contract next. 
+ `num_connecting` is the number of edges connecting to `temp_start`. 
+ `cost` is the cost of contracting, 
+ `order` is the order of the tensors to be contracted (left or right), 
+ `side` is to signal whether to check the left or the right side in the contraction, 
+ `depth` is how many tensors deep we search to find the best contraction order.
"""
function best_order_helper(next_tensors::Vector{intType}, num_connecting::Vector{intType}, temp_start::Indices, graph::network,cost::intType, order::String, side::String, depth::intType) 

	if length(order) == length(next_tensors)
		return (cost, order)

	elseif (side == "left")
		temp = graph.net[next_tensors[depth]]
		temp_next = Indices(getnames(temp), size(temp))
		pos_left,pos_right = find_common_edges(temp_next.names, temp_start.names, num_connecting[depth])

		cost += permute_cost(temp_next, temp_start, pos_left,pos_right)

		left = update_temp(temp_next, temp_start, num_connecting[depth])
		order = order*"L"
		depth += 1

		next_left = best_order_helper(next_tensors, num_connecting, left, graph,cost, order, "left", depth)
		next_right = best_order_helper(next_tensors, num_connecting, left, graph,cost, order, "right", depth)

		if next_left[1]<=next_right[1]
			return next_left
		else 
			return next_right
		end

	else
		temp = graph.net[next_tensors[depth]]
		temp_next = Indices(getnames(temp), size(temp))
		pos_left,pos_right = find_common_edges(temp_start.names, temp_next.names, num_connecting[depth])

		cost += permute_cost(temp_start, temp_next, pos_left,pos_right)

		right = update_temp(temp_start, temp_next, num_connecting[depth])
		order = order*"R"
		depth += 1


		next_left = best_order_helper(next_tensors, num_connecting, right, graph,cost, order, "left", depth)
		next_right = best_order_helper(next_tensors, num_connecting, right, graph,cost, order, "right", depth)

		if next_left[1]<=next_right[1]
			return next_left
		else 
			return next_right
		end
	end
end

# finds the names and dimensions of common edges between two tensors
"""
    find_common_edges(left_edges,right_edges,num_common)

Finds the position of the indices for two tensors that are connected. `left_edges` are the names of the left edges, `right_edges` are the names of the right edges, and `num_common` is the number of common indices between the `left_edges` and the 	`right_edges`
"""
function find_common_edges(left_edges::Vector{String}, right_edges::Vector{String}, num_common::intType) 
	pos_left = Array{intType,1}(undef,num_common)
	pos_right = Array{intType,1}(undef,num_common)
	val = 1
	counter = 0

	while val <= num_common 
		counter += 1
		if left_edges[counter] in right_edges
			pos_left[val] = counter
			pos_right[val] = findfirst(==(left_edges[counter]),right_edges)
			val +=1
		end
	end
	return pos_left, pos_right
end

"""
    update_edges(next_names,next_pos,shared_edges)

Removes value in dictionary `shared_edges` in order to grab the first index of the resulting dictionary in a subsequent step (removing costs no extra allocations). `next_pos` is the position of the input tensor that is to be removed.
`next_names` are the names of the indices of the tensor to remove
"""
function update_edges(next_names::Union{Vector{String},Memory{String}},next_pos::intType, shared_edges::Dict{String,Vector{intType}})
	for edge in next_names
		adjacent = shared_edges[edge]

		if (length(adjacent)==2)&&(adjacent[1] != adjacent[2])
			location = findfirst(==(next_pos), adjacent)
			deleteat!(adjacent, location)

		elseif (length(adjacent)==1)
			location = findfirst(==(next_pos), adjacent)
			deleteat!(adjacent, location)

		end

	end
	return shared_edges
end


# GLOBAL
"""
    greedy_contract(graph,shared_edges,[,exclude=intType[]])

Contracts a tensor network into a single tensor. The values in `exclude` are the index positions of tensors to not contract. 
"""
function greedy_contract(graph::network, shared_edges::Dict{String, Vector{intType}}; exclude::Vector{intType} = intType[])

  num_tensors = length(graph.net)-length(exclude)

  for i in 1:num_tensors-1

    shared_edges,num_connections = makeNeighbourTable(graph)

    to_contract = lowest_cost(shared_edges,graph)

    x = to_contract[1]
    y = to_contract[2]

   	if to_contract==(-1,-1)
   		error("The given network is disjoint")

   	end

    new_tensor = graph.net[x]*graph.net[y]

    graph.net[x] = new_tensor
    deleteat!(graph.net,y)

  end

  return graph.net[1]
end

"""
    lowest_cost(shared_edges, G)

Find the least expensive contraction within a network `G`
"""
function lowest_cost(shared_edges::Dict{String, Vector{intType}},graph::network)
	min_cost = -1
	all_keys = keys(shared_edges)
	to_contract = (-1,-1) # this may take an extra allocation

	for key_val in all_keys 
		position = shared_edges[key_val]
		pos1 = position[1]
		pos2 = position[2]

		left_tensor = graph[pos1]
		right_tensor = graph[pos2]

		if !(pos1 == pos2)
			left_details = Indices(left_tensor.names, size(left_tensor))
			right_details = Indices(right_tensor.names, size(right_tensor))

			num_common = val_common(left_details,right_details)
			common_edges,left_pos,right_pos = find_common_edges(left_details, right_details,num_common)

			basic_cost = (get_cost(left_details.dimensions) + get_cost(right_details.dimensions))÷prod(common_edges)
			cost_tot1 = basic_cost+permute_cost(left_details,right_details,left_pos,right_pos)
			cost_tot2 = basic_cost+permute_cost(right_details,left_details,right_pos,left_pos)


			if (cost_tot1<min_cost)||(cost_tot2<min_cost)||(min_cost==-1)
				if cost_tot1<cost_tot2
          to_contract = (pos1,pos2)
					min_cost = cost_tot1

				else
          to_contract = (pos2,pos1)
					min_cost = cost_tot2
				end
			end
		end
	end

	return to_contract
end


"""
    val_common(left_details,right_details)

Returns the number of indices in common between `left_details` and `right_details`
"""
function val_common(left_details::Indices,right_details::Indices)
	num = 0
	for edge in left_details.names
		if edge in right_details.names
			num += 1
		end
	end
	return num
end

"""
    find_common_edges(left_edges, right_edges, num_common)

Finds the name of common indices, and the positions of the index on the right and left tensor.
"""
function find_common_edges(left_edges::Indices, right_edges::Indices, num_common::intType) 
	pos_left = Array{intType,1}(undef,num_common)
	pos_right = Array{intType,1}(undef,num_common)
	name_common = Array{intType,1}(undef,num_common)

	val = 1
	counter = 0
	while val <= num_common 
		counter += 1
		if left_edges.names[counter] in right_edges.names
			pos_left[val] = counter
			pos_right[val] = findfirst(==(left_edges.names[counter]),right_edges.names)
			name_common[val] = left_edges.dimensions[counter]
			val += 1

		end
	end

	return name_common,pos_left, pos_right
end
