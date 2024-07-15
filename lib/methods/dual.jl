function dual(A::TNnetwork)
  return dual!(copy(A))
end

function dual!(A::TNnetwork)
  A.level += 1
  return A
end