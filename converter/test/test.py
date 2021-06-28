# - getNodesFromType (pyTree) -
import re
import Converter.PyTree as C
import Generator.PyTree as G
import converter.internal as I

a = G.cart((0,0,0), (1,1,1), (10,10,10))
t = C.newPyTree(['Base', a])
print(f"[{I.get_name(t)}, .., {I.get_label(t)}]")

# Return nodes of type 'Zone_t' from predicate as lambda
zones0 = I.getNodesFromType(t, 'Zone_t'); # from Internal
zones1 = I.get_nodes_from_predicate(t, lambda n: I.get_label(n) == 'Zone_t'); # from additional
assert(all([I.is_same_name(z0, z1) for z0, z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

# Return nodes of type 'Zone_t' from predicate with 'dfs'
zones0 = I.getNodesFromType(t, 'Zone_t'); # from Internal
zones1 = I.get_nodes_from_predicate(t, lambda n: I.get_label(n) == 'Zone_t', method='dfs'); # from additional
assert(all([I.is_same_name(z0, z1) for z0, z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

# Return nodes of type 'Zone_t' from predicate with 'bfs' and depth=2
zones0 = I.getNodesFromType(t, 'Zone_t'); # from Internal
zones1 = I.get_nodes_from_predicate(t, lambda n: I.get_label(n) == 'Zone_t', method='dfs', depth=2); # from additional
assert(all([I.is_same_name(z0, z1) for z0, z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

# Return nodes of type 'Zone_t' from predicate with 'bfs' and explore='shallow'
zones0 = I.getNodesFromType(t, 'Zone_t'); # from Internal
zones1 = I.get_nodes_from_predicate(t, lambda n: I.get_label(n) == 'Zone_t', method='dfs', explore='shallow'); # from additional
assert(all([I.is_same_name(z0, z1) for z0, z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

# Return nodes of type 'Zone_t'
zones0 = I.getNodesFromType(t, 'Zone_t'); # from Internal
zones1 = I.get_nodes_from_label(t, 'Zone_t'); # from additional
# I.print_tree(zones1)
assert(all([I.is_same_name(z0, z1) for z0, z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

# Limit search to 2 levels (faster)
zones0 = I.getNodesFromType1(t, 'Zone_t'); # from Internal
zones1 = I.get_nodes_from_label2(t, 'Zone_t');
# I.print_tree(zones1)
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

# Limit search with shallow explore (faster)
zones0 = I.getNodesFromType1(t, 'Zone_t'); # from Internal
zones1 = I.get_shallow_nodes_from_label(t, 'Zone_t');
# I.print_tree(zones1)
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

# Return nodes of type 'Zone_t' (iterator)
zones0 = I.getNodesFromType(t, 'Zone_t'); # from Internal
zones1 = I.iter_nodes_from_label(t, 'Zone_t'); # from additional
# I.print_tree(zones1)
assert(all([I.is_same_name(z0, z1) for z0, z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

# Limit search to 2 levels (faster) (iterator)
zones0 = I.getNodesFromType1(t, 'Zone_t'); # from Internal
zones1 = I.iter_nodes_from_label2(t, 'Zone_t');
# I.print_tree(zones1)
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

# Limit search with shallow explore (faster) (iterator)
zones0 = I.getNodesFromType1(t, 'Zone_t'); # from Internal
zones1 = I.iter_shallow_nodes_from_label(t, 'Zone_t');
# I.print_tree(zones1)
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

# Walker as iterator
walker = I.NodesWalker(t, lambda n: I.get_label(n) == 'Zone_t')
zones1 = walker(method='dfs', explore='deep');
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones0, zones1)]))
# Can repeat the generator
zones2 = walker(method='bfs', explore='shallow');
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones0, zones2)]))

# Walker as list with caching
walker = I.NodesWalker(t, lambda n: I.get_label(n) == 'Zone_t', caching=True)
zones1 = walker(method='bfs', explore='shallow');
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones0, zones1)]))
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones1, walker.cache)]))

# iter_nodes_by_matching
zones1 = I.iter_nodes_by_matching(t, 'CGNSBase_t/Zone_t')
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones0, zones1)]))
zones1 = I.iter_nodes_by_matching(t, ['CGNSBase_t', 'Zone_t'])
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones0, zones1)]))
zones1 = I.iter_nodes_by_matching(t, ['CGNSBase_t', lambda n : I.get_label('Zone_t')])
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones0, zones1)]))

# Test check_is_label
@I.check_is_label("Zone_t")
def apply_on_zone(zone_node: I.TreeNode):
  I.printTree(zone_node)

for node in I.get_all_zone(t):
  apply_on_zone(node)

try:
  for node in I.get_all_base(t):
    apply_on_zone(node)
except I.CGNSLabelNotEqualError as e:
  pass

# Test get_nodes_from_predicate
coordxy = I.get_nodes_from_predicate(t, lambda n: re.match(r"Coordinate[XY]", I.get_name(n)))
assert([I.get_name(n) for n in coordxy] == ["CoordinateX", "CoordinateY"])

coordxy = I.get_nodes_from_predicate(t, lambda n: re.match(r"Coordinate[XY]", I.get_name(n)), depth=4)
assert([I.get_name(n) for n in coordxy] == ["CoordinateX", "CoordinateY"])

# Test get_child_from_predicate
coordz = I.get_node_from_predicate(t, lambda n: re.match(r"CoordinateZ", I.get_name(n)))
assert(I.get_name(coordz) == "CoordinateZ")

try:
  coordz = I.get_node_from_predicate(t, lambda n: re.match(r"CoordinateH", I.get_name(n)))
except I.CGNSNodeFromPredicateNotFoundError as e:
  pass

# Test request_node_from_predicate
coordz = I.request_node_from_predicate(t, lambda n: re.match(r"CoordinateZ", I.get_name(n)))
assert(I.get_name(coordz) == "CoordinateZ")

coordz = I.request_node_from_predicate(t, lambda n: re.match(r"CoordinateH", I.get_name(n)))
assert(coordz == None)

# Test iter_nodes_by_matching
for zone in I.get_all_zone(t):
  coordxy = I.iter_nodes_by_matching(zone, ['GridCoordinates_t', lambda n: re.match(r"Coordinate[XY]", I.get_name(n))])
  assert [I.getName(n) for n in coordxy] == ['CoordinateX', 'CoordinateY']