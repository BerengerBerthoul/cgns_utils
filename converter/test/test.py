# - getNodesFromType (pyTree) -
import re
import Converter.PyTree as C
import Generator.PyTree as G
import converter.internal as I

a = G.cart((0,0,0), (1,1,1), (10,10,10))
t = C.newPyTree(['Base', a])

# Return nodes of type 'Zone_t'
zones0 = I.getNodesFromType(t, 'Zone_t');
zones1 = I.get_nodes_from_label(t, 'Zone_t'); # from Internal = getNodesFromType
assert(all([I.is_same_name(z0, z1) for z0, z1 in zip(zones0, zones1)]))

zones0 = I.get_nodes_from_label(t, 'Zone_t'); # from Internal = getNodesFromType
zones1 = I.get_children_from_label(t, 'Zone_t'); # from additional
I.print_tree(zones1)
assert(all([I.is_same_name(z0, z1) for z0, z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

# Limit search to 2 levels (faster)
zones0 = I.getNodesFromType(t, 'Zone_t');
zones1 = I.get_nodes_from_label(t, 'Zone_t'); # from Internal = getNodesFromType
assert(all([I.is_same_name(z0, z1) for z0, z1 in zip(zones0, zones1)]))

zones0 = I.get_nodes_from_label1(t, 'Zone_t'); # from Internal = getNodesFromType1
zones1 = I.get_children_from_label2(t, 'Zone_t');
I.print_tree(zones1)
assert(all([I.is_same_name(z0,z1) for z0,z1 in zip(zones0, zones1)]))
#>> [['cart', array(..), [..], 'Zone_t']]

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

# Test get_children_from_predicate
coordxy = I.get_children_from_predicate(t, lambda n: re.match(r"Coordinate[XY]", I.get_name(n)))
assert([I.get_name(n) for n in coordxy] == ["CoordinateX", "CoordinateY"])

coordxy = I.get_children_from_predicate(t, lambda n: re.match(r"Coordinate[XY]", I.get_name(n)), depth=4)
assert([I.get_name(n) for n in coordxy] == ["CoordinateX", "CoordinateY"])

# Test get_child_from_predicate
coordz = I.get_child_from_predicate(t, lambda n: re.match(r"CoordinateZ", I.get_name(n)))
assert(I.get_name(coordz) == "CoordinateZ")

try:
  coordz = I.get_child_from_predicate(t, lambda n: re.match(r"CoordinateH", I.get_name(n)))
except I.CGNSNodeFromPredicateNotFoundError as e:
  pass

# Test request_child_from_predicate
coordz = I.request_child_from_predicate(t, lambda n: re.match(r"CoordinateZ", I.get_name(n)))
assert(I.get_name(coordz) == "CoordinateZ")

coordz = I.request_child_from_predicate(t, lambda n: re.match(r"CoordinateH", I.get_name(n)))
assert(coordz == None)

# Test iter_nodes_by_matching
for zone in I.get_all_zone(t):
  coordxy = I.iter_nodes_by_matching(zone, ['GridCoordinates_t', lambda n: re.match(r"Coordinate[XY]", I.get_name(n))])
  assert [I.getName(n) for n in coordxy] == ['CoordinateX', 'CoordinateY']