# -- PyTree Node Management --

from typing import List, Optional, NoReturn, Union, Tuple, Callable, Any

import fnmatch
import numpy

# Declare a type alias for type hint checkers
# For Python>=3.9 it is possible to set the MaxLen
# from typing import Annotated
# TreeNode = Annotated[List[Union[str, Optional[numpy.ndarray], List['TreeNode']]], MaxLen(4)]
TreeNode = List[Union[str, Optional[numpy.ndarray], List["TreeNode"]]]

# Keys to access TreeNode values
__NAME__ = 0
__DATA__ = 1
__CHILDREN__ = 2
__LABEL__ = 3

# Default Containers naming
__GridCoordinates__ = "GridCoordinates"
__FlowSolutionNodes__ = "FlowSolution"
__FlowSolutionCenters__ = "FlowSolution#Centers"

# For unstructured BCs (mixed latest/old CGNS version)
__ELEMENTRANGE__ = "ElementRange"  # ElementRange/PointRange
__ELEMENTLIST__ = "PointList"  # ElementList/PointList
__FACELIST__ = "PointList"  # FaceList/PointList
__FACERANGE__ = "PointRange"  # FaceRange/PointRange
__POINTLIST__ = "PointList"  # Vertex list

# alternative variable names to CGNS var names
name_to_CGNS = {
    "x": "CoordinateX",
    "y": "CoordinateY",
    "z": "CoordinateZ",
    "x3d": "CoordinateX",
    "y3d": "CoordinateY",
    "z3d": "CoordinateZ",
    "X": "CoordinateX",
    "Y": "CoordinateY",
    "Z": "CoordinateZ",
    "ro": "Density",
    "rou": "MomentumX",
    "rov": "MomentumY",
    "row": "MomentumZ",
    "rovx": "MomentumX",
    "rovy": "MomentumY",
    "rovz": "MomentumZ",
    "roe": "EnergyStagnationDensity",
    "roE": "EnergyStagnationDensity",
    "rok": "TurbulentEnergyKineticDensity",
    "ronutilde": "TurbulentSANuTildeDensity",
    "roeps": "TurbulentDissipationDensity",
    "roomega": "TurbulentDissipationRateDensity",
    "mach": "Mach",
    "psta": "Pressure",
    "tsta": "Temperature",
    "viscrapp": "Viscosity_EddyMolecularRatio",
    "walldistance": "TurbulentDistance",
    "wallglobalindex": "TurbulentDistanceIndex",
}

# Known BCs
KNOWN_BCS = [
    "BCWall",
    "BCWallInviscid",
    "BCWallViscous",
    "BCWallViscousIsothermal",
    "BCFarfield",
    "BCExtrapolate",
    "BCInflow",
    "BCInflowSubsonic",
    "BCOutflow",
    "BCMatch",
    "BCNearMatch",
    "BCOverlap",
    "BCSymmetryPlane",
    "BCDegenerateLine",
    "BCDegeneratePoint",
    "BCStage",
    "UserDefined",
]

# import math
__DEG2RAD__ = 0.017453292519943295  # math.pi/180.
__RAD2DEG__ = 57.29577951308232  # 180./math.pi


# ==============================================================================
# -- is? --
# ==============================================================================


def is_tree_root(node: TreeNode) -> bool:
    """Return True if node is a CGNSTree_t node at the top."""
    if len(node) != 4:
        return False
    if node[__LABEL__] == "CGNSTree_t":
        return True
    return False


def is_std_node(node) -> int:
    """Return 0 if node is a list of standard pyTree nodes,
     -1 if node is a standard pyTree node, -2 otherwise.
    """
    if not isinstance(node, list):
        return -2
    if len(node) == 0:
        return 0
    node0 = node[0]
    if (
        isinstance(node0, str)
        and len(node) == 4
        and isinstance(node[__CHILDREN__], list)
    ):
        return -1
    if not isinstance(node0, list):
        return -2
    if (
        isinstance(node0[__NAME__], str)
        and len(node0) == 4
        and isinstance(node0[__CHILDREN__], list)
    ):
        return 0
    return -2


# -- get_enum_type
# Returns 1 if node is a Zone_t
# Returns 2 if node is a Zone_t list
# Returns 3 if node is a tree root
# Returns 4 if node is a Base_t
# Returns 5 if node is a Base_t list
# Else returns -1
def get_enum_type(node) -> int:
    """Returns the type of node as an integer."""
    ret = -1
    if not isinstance(node, list):
        return ret
    node_size = len(node)
    if node_size == 4 and isinstance(node[__LABEL__], str):
        label = node[__LABEL__]
        if label == "Zone_t":
            ret = 1
        elif label == "CGNSTree_t":
            ret = 3
        elif label == "CGNSBase_t":
            ret = 4
    elif node_size > 0:
        node1 = node[0]
        if len(node1) == 4 and isinstance(node1[__LABEL__], str):
            if node1[__LABEL__] == "Zone_t":
                ret = 2
            elif node1[__LABEL__] == "CGNSBase_t":
                ret = 5
    return ret


# -- is_type
# Compare given type and node type. Accepts widlcards.
# IN: label: string (Zone_t...)
# IN: node: pyTree node
def is_type(node: TreeNode, label: str) -> bool:
    """Return True if node is of given type."""
    tnode = node[__LABEL__]
    if ("*" in label) | ("?" in label) | ("[" in label):
        return fnmatch.fnmatch(tnode, label)
    return tnode == label


# -- is_name
# Compare given name and node name. Accepts widlcards.
# IN: name: string ou numpy char array
# IN: node: pyTree node
def is_name(node: TreeNode, name: Union[str, numpy.ndarray]) -> bool:
    """Return True if node has given name."""
    if isinstance(name, numpy.ndarray):
        sname = name.tobytes().decode("ascii", "strict")
    else:
        sname = str(name)
    snode = node[__NAME__]
    if ("*" in sname) | ("?" in sname) | ("[" in sname):
        return fnmatch.fnmatch(snode, sname)
    return snode == sname


# -- is_value
# Compare given value and node value. Accepts wildcards.
def is_value(node: TreeNode, value: Any) -> bool:
    """Return True if node has given value."""
    # Handle None case
    if node[__DATA__] is None:
        return node[__DATA__] == value
    # if not isinstance(node[__DATA__], numpy.ndarray):
    #    raise TypeError("node is not valid")

    # bool for string value
    is_str_value = isinstance(value, str)
    is_bytes_value = isinstance(value, bytes)
    is_nstr_value = isinstance(value, numpy.ndarray) and value.dtype.char in ["c", "S"]
    # Not string case
    if not (is_str_value or is_bytes_value or is_nstr_value):
        # Value to ndarray
        if not isinstance(value, numpy.ndarray):
            value = numpy.array([value])  # float or int
        # Node Value to ndarray
        if not isinstance(node[__DATA__], numpy.ndarray):
            node_value = numpy.array(
                [node[__DATA__]]
            )  # float or int not CGNS/Python compliant
        else:
            node_value = node[__DATA__]
        # Comparison on ndarray
        if value.dtype != node_value.dtype:
            return False
        if value.shape != node_value.shape:
            return False
        res = value == node_value
        return res.all()
    # String Case
    node_value = get_value(node)
    if not isinstance(node_value, str):
        return False
    if is_nstr_value:
        value = value.tobytes().decode("ascii", "strict")
    elif is_bytes_value:
        value = value.decode("ascii", "strict")
    # Comparison
    if ("*" in value) | ("?" in value) | ("[" in value):
        res = fnmatch.fnmatch(node_value, value)
    else:
        res = node_value == value
    return res


# -- is_name_and_type
# Compare type and name of node with given (name, label)
def is_name_and_type(node: TreeNode, name: str, label: str) -> bool:
    return is_name(node, name) and is_type(node, label)


# -- is descendant
# Return true if node is a descendant of parent (even at deep levels)
def is_descendant(parent: TreeNode, node: TreeNode) -> bool:
    """Return True if node is a descendant of node parent."""
    if parent is node:
        return True
    for child in parent[__CHILDREN__]:
        if is_descendant(child, node):
            return True
    return False


# alias of is_descendant
def is_subchild(parent: TreeNode, node: TreeNode) -> bool:
    """Return True if node is a descendant of node parent."""
    return is_descendant(parent, node)


def is_child(parent: TreeNode, node: TreeNode) -> bool:
    """Return True if node is a direct child of node parent."""
    if parent is node:
        return True
    for child in parent[__CHILDREN__]:
        if child is node:
            return True
    return False


def is_grandchild(parent: TreeNode, node: TreeNode) -> bool:
    """Return True if node is a grandchild of node parent."""
    if parent is node:
        return True
    for child in parent[__CHILDREN__]:
        if child is node:
            return True
        for grandchild in child[__CHILDREN__]:
            if grandchild is node:
                return True
    return False


def has_child_name(parent: TreeNode, name: str) -> bool:
    """Return True if name exists as child of node parent."""
    if not parent:
        return False
    if parent[__CHILDREN__] is None:
        return False
    for child in parent[__CHILDREN__]:
        if child[__NAME__] == name:
            return True
    return False


# ============================================================================
# -- Node access --
# ============================================================================


def get_name(node: TreeNode) -> str:
    """Return node name.

    :param node: a Python/CGNS node
    :return: node name
    """
    return node[__NAME__]


# Short version of get_name
def name(node: TreeNode) -> str:
    """Return node name.

    :param node: a Python/CGNS node
    :return: node name
    """
    return node[__NAME__]


def get_label(node: TreeNode) -> str:
    """Return node label.

    :param node: a Python/CGNS node
    :return: node label
    """
    return node[__LABEL__]


# Short version of get_label
def label(node: TreeNode) -> str:
    """Return node label.

    :param node: a Python/CGNS node
    :return: node label
    """
    return node[__LABEL__]


# -- Retourne les enfants d'un noeud
def get_children(node: TreeNode) -> List[TreeNode]:
    """Return children list of a node.

    :param node: a Python/CGNS node TreeNode
    :return: childen list
    """
    return node[__CHILDREN__]


# Short version of get_children
def children(node: TreeNode) -> List[TreeNode]:
    """Return children list of a node.

    :param node: a Python/CGNS node
    :return: childen list
    """
    return node[__CHILDREN__]


def get_data(node: TreeNode) -> Optional[numpy.ndarray]:
    """Returns node value, could be `None` or a `numpy.ndarray`.

    :param node: a Python/CGNS node
    :return: numpy array
    """
    _v = node[__DATA__]
    if not isinstance(_v, numpy.ndarray):
        return None
    if _v.dtype.kind in ["S", "a"] or _v.dtype.name in [
        "float32",
        "float64",
        "int32",
        "int64",
        "complex32",
        "complex64",
    ]:
        return _v
    return None


# short version of get_data
def data(node: TreeNode) -> Optional[numpy.ndarray]:
    return get_data(node)


def get_data_as_str(node: TreeNode) -> str:
    """Returns the value of the node as a Python string

    :param node: a Python/CGNS node
    :return: a string
    """
    return node[__DATA__].tobytes().decode("ascii", "strict")


def data_as_str(node: TreeNode) -> str:
    """Returns the value of the node as a Python string"""
    return node[__DATA__].tobytes().decode("ascii", "strict")


def get_value(node: TreeNode):
    """Return the value of a node."""
    node_data = node[__DATA__]
    ret = node_data
    if isinstance(node_data, numpy.ndarray):
        if node_data.dtype.char == "S" or node_data.dtype.char == "c":
            if len(node_data.shape) == 1:
                ret = node_data.tobytes().decode("ascii", "strict")
            else:
                out = []
                for i in range(node_data.shape[1]):
                    _v = node_data[:, i].tobytes().decode("ascii", "strict")
                    out.append(_v.strip())
                ret = out
        elif node_data.dtype == numpy.int32 and node_data.size == 1:
            ret = int(node_data.flat[0])
        elif node_data.dtype == numpy.int64 and node_data.size == 1:
            ret = int(node_data.flat[0])
        elif node_data.dtype == numpy.float64 and node_data.size == 1:
            ret = float(node_data.flat[0])
        elif node_data.dtype == numpy.float32:
            if node_data.size == 1:
                ret = float(node_data.flat[0])
            else:
                arr = numpy.empty(node_data.shape, dtype=numpy.float64)
                arr[:] = node_data[:]
                ret = arr
    return ret


def value(node: TreeNode):
    return get_value(node)


def get_val(node: TreeNode) -> Optional[numpy.ndarray]:
    """Return the value of a node always as a numpy."""
    return node[__DATA__]


def val(node: TreeNode) -> Optional[numpy.ndarray]:
    return get_val(node)


def get_path(tree: TreeNode, node: TreeNode, clean_root: bool = False) -> str:
    """Return the path of node."""
    if tree is node:
        return ""
    if clean_root and tree[__NAME__] == "CGNSTree":
        path = ""
    else:
        path = tree[__NAME__]
    path = __get_path(tree, node, path)
    return path


def __get_path(tree: TreeNode, node: TreeNode, path: str = "") -> Optional[str]:
    if tree is node:
        return path
    for child in tree[__CHILDREN__]:
        result_path = __get_path(child, node, path + "/" + child[__NAME__])
        if result_path:
            return result_path
    return None


# =============================================================================
# -- Set/create nodes --
# =============================================================================


def set_name(node: TreeNode, name: str) -> NoReturn:
    """Set name in node.

    :param node: a Python/CGNS node
    :param name: a string
    :return:
    """
    if not isinstance(name, str):
        raise TypeError(
            "set_name: name of node must be a string({:s})".format(
                name.__repr__()[: min(len(name.__repr__()), 60)]
            )
        )
    node[__NAME__] = name


def set_label(node: TreeNode, label: str) -> NoReturn:
    if not isinstance(label, str):
        raise TypeError(
            "set_label: type of a node must be a string ({:s})".format(
                label.__repr__()[: min(len(label.__repr__()), 60)]
            )
        )
    node[__LABEL__] = label


def str_to_ascii_array(value: str) -> numpy.ndarray:
    """convert string to numpy array"""
    lst_bytes = [c.encode("ascii") for c in value]
    return numpy.array(lst_bytes, dtype="|S", order="F")


def set_str_data(node: TreeNode, value: str = None) -> NoReturn:
    if not isinstance(value, str):
        raise TypeError(
            "set_str_data: value of a node must be a string ({:s})".format(
                label.__repr__()[: min(len(label.__repr__()), 60)]
            )
        )
    node[__DATA__] = str_to_ascii_array(value)


def set_data(node: TreeNode, value: numpy.ndarray = None) -> NoReturn:
    if node is None:
        return
    if not isinstance(value, numpy.ndarray):
        node[__DATA__] = None
    elif value.dtype.kind in ["S", "a"] or value.dtype.name in [
        "float32",
        "float64",
        "int32",
        "int64",
        "complex32",
        "complex64",
    ]:
        node[__DATA__] = value


def set_value(node: TreeNode, value=None) -> NoReturn:
    """Set given value in node."""
    if node is None:
        return
    if node[__LABEL__] == "CGNSLibraryVersion_t":
        if isinstance(value, (float, int)):
            node[__DATA__] = numpy.array([value], dtype=numpy.float32)
        elif isinstance(value, numpy.ndarray):
            node[__DATA__] = numpy.array(value, dtype=numpy.float32)
        else:
            raise TypeError(
                "setValue: CGNSLibraryVersion node value should be a float."
            )
        return
    if value is None:
        node[__DATA__] = None
    elif isinstance(value, numpy.ndarray):
        if value.flags.f_contiguous:
            node[__DATA__] = value
        else:
            node[__DATA__] = numpy.asfortranarray(value)
    elif isinstance(value, (int, numpy.int32, numpy.int64)):
        node[__DATA__] = numpy.array([value], dtype=numpy.int32)
    elif isinstance(value, (float, numpy.float32, numpy.float64)):
        node[__DATA__] = numpy.array([value], dtype=numpy.float64)
    elif isinstance(value, str):
        node[__DATA__] = str_to_ascii_array(value)
    elif isinstance(value, (list, tuple)):
        test_value = value
        while isinstance(test_value, (list, tuple)):
            test_value = test_value[0]
        if isinstance(test_value, (float, numpy.float32, numpy.float64)):
            node[__DATA__] = numpy.array(value, dtype=numpy.float64, order="F")
        elif isinstance(test_value, (int, numpy.int32, numpy.int64)):
            node[__DATA__] = numpy.array(value, dtype=numpy.int32, order="F")
        elif isinstance(test_value, str):
            if isinstance(value[0], str):
                arr = numpy.empty((32, len(value)), dtype=numpy.bytes_, order="F")
                for num, item in enumerate(value):
                    size = min(len(item), 32)
                    arr[:, num] = b" "
                    arr[0:size, num] = str_to_ascii_array(item)[0:size]
                node[__DATA__] = arr
            elif isinstance(value[0], (list, tuple)):
                arr = numpy.empty(
                    (32, len(value[0]), len(value)), dtype=numpy.bytes_, order="F"
                )
                for gid, group in enumerate(value):
                    for num, item in enumerate(group):
                        size = min(len(item), 32)
                        arr[:, num, gid] = b" "
                        arr[0:size, num, gid] = str_to_ascii_array(item)[0:size]
                node[__DATA__] = arr
    else:
        node[__DATA__] = numpy.array([value])


def create_node(
    name: str,
    label: str,
    value: Optional[numpy.ndarray] = None,
    children: List[TreeNode] = None,
    parent: TreeNode = None,
) -> TreeNode:
    """Create a pyTree node."""
    node = [None, None, [], None]
    set_name(node, name)
    set_label(node, label)
    set_data(node, value)
    if children is not None:
        node[__CHILDREN__] = children
    if parent is not None:
        parent[__CHILDREN__].append(node)
    return node


def add_child(parent: TreeNode, child: TreeNode, pos: int = -1) -> TreeNode:
    """Add a child node to node's children."""
    __add_child(parent, child, pos=pos)
    return child


def __add_child(
    parent: TreeNode, child: Union[TreeNode, List[TreeNode]], pos: int = -1
) -> NoReturn:
    """Add a child node to node's children."""
    is_std = is_std_node(child)
    if is_std == -1:
        if pos < 0:
            parent[__CHILDREN__].append(child)
        else:
            parent[__CHILDREN__].insert(pos, child)
    elif is_std == 0:
        if pos < 0:
            parent[__CHILDREN__].extend(child)
        elif pos == 0:
            parent[__CHILDREN__] = child + parent[__CHILDREN__]
        else:
            parent[__CHILDREN__] = (
                parent[__CHILDREN__][:pos] + child + parent[__CHILDREN__][pos:]
            )


def create_child(
    parent: TreeNode,
    name: str,
    label: str,
    value: Optional[Union[numpy.ndarray, str, list]] = None,
    children: List[TreeNode] = None,
    pos: int = -1,
    check: bool = True,
) -> TreeNode:
    exist_idx = -1
    if check:
        for idx, child in enumerate(parent[__CHILDREN__]):
            if child[__NAME__] == name:
                exist_idx = idx
                break
    if exist_idx == -1:
        if pos < 0:
            child = create_node(name, label, value, children, parent)
        else:
            child = create_node(name, label, value, children)
            child = add_child(parent, child, pos)
    else:
        child = parent[__CHILDREN__][exist_idx]
        set_label(child, label)
        set_value(child, value)
        if children is not None:
            child[__CHILDREN__] = children
    return child


def append(tree: TreeNode, node: TreeNode, path: str) -> TreeNode:
    """Append a node to t specifying its path in t."""
    parent_elem = get_node_from_path(tree, path)
    child = get_child_by_name(parent_elem, node[__NAME__])
    if child is None:
        child = add_child(parent_elem, node)
    elif node[__DATA__] is not None:
        child[__DATA__] = node[__DATA__]
        child[__LABEL__] = node[__LABEL__]
        for subchild in node[__CHILDREN__]:  # append node children
            append(child, subchild, node[__NAME__])
    else:
        for subchild in node[__CHILDREN__]:
            append(child, subchild, node[__NAME__])
    return child


def create_node_from_path(tree: TreeNode, path: str):
    tokens = path.split("/")
    result_tokens = []
    for token in tokens:
        if token in ["", "."]:
            continue
        if result_tokens and token == "..":
            result_tokens.pop()
        else:
            result_tokens.append(token)
    tokens = result_tokens
    parent = tree
    for token in tokens:
        child = get_child_by_name(parent, token)
        if child is None:
            parent = create_child(parent, token, "DataArray_t", None)
        else:
            parent = child


# ==============================================================================
# -- Node retrieval --
# ==============================================================================

# -- Returns a node according to xpath
# IN: node: starting point for searching
# IN: path: relative path with respect to node
def get_node_from_path(tree: TreeNode, path: str) -> Optional[TreeNode]:
    """Returns a node from a path."""
    if path in ("", "/"):
        return tree
    if path[0] == "/":
        path = path[1:]
    if path[-1] == "/":
        path = path[:-1]
    if tree[__NAME__] == path:
        return tree
    if tree[__LABEL__] == "CGNSTree_t":
        clean_path = path.replace(tree[__NAME__] + "/", "")
    else:
        clean_path = path
    tokens = clean_path.split("/")
    is_std = is_std_node(tree)
    if is_std >= 0:
        for item in tree:
            if item[__NAME__] == tokens[0]:
                break
        else:
            return None
        if len(tokens) == 1:
            return item
        return __get_node_from_path(item, tokens[1:])
    return __get_node_from_path(tree, tokens)


def __get_node_from_path(node: TreeNode, paths: List[str]) -> Optional[TreeNode]:
    _node = node
    for token in paths:
        _node = get_child_by_name(_node, token)
        if _node is None:
            return None
    return _node


get_node_by_path = get_node_from_path  # alias


def get_child_by_name(node, name):
    for child in node[__CHILDREN__]:
        if child[__NAME__] == name:
            return child
    return None


def get_child_by_label(node, label):
    for child in node[__CHILDREN__]:
        if child[__LABEL__] == label:
            return child
    return None


def get_children_by_label(node, label):
    result = []
    for child in node[__CHILDREN__]:
        if child[__LABEL__] == label:
            result.append(child)
    return result


def get_children_by_labels(node, labels):
    result = []
    for child in node[__CHILDREN__]:
        if child[__LABEL__] in labels:
            result.append(child)
    return result


def get_parent_of_node(
    tree: Union[TreeNode, List[TreeNode]], node: TreeNode
) -> Tuple[Optional[TreeNode], int]:
    """Return the parent of given node."""
    is_std = is_std_node(tree)
    if is_std == 0:
        for idx, item in enumerate(tree):
            if item is node:
                return tree, idx
        for idx, item in enumerate(tree):
            res, pos = __get_parent_of_node(item, node)
            if res is not None:
                return res, pos
        return None, 0
    if is_std == -1:
        return __get_parent_of_node(tree, node)
    return None, 0


def get_parent_node(tree: TreeNode, node: TreeNode) -> Optional[TreeNode]:
    res, _ = __get_parent_of_node(tree, node)
    return res


def __get_parent_of_node(
    tree: TreeNode, node: TreeNode
) -> Tuple[Optional[TreeNode], int]:
    result, pos = None, 0
    for idx, child in enumerate(tree[__CHILDREN__]):
        if child is node:
            return tree, idx
        result, pos = __get_parent_of_node(child, node)
        if result is not None:
            return result, pos
    return result, pos


# -- Return only one node (no wildcard) - Fast
def get_node_by_name(tree, name):
    """Return the first matching node with given name."""
    is_std = is_std_node(tree)
    if is_std >= 0:
        for item in tree:
            ret = get_node_by_name(item, name)
            if ret is not None:
                return ret
        return None
    return __get_node_by_name(tree, name)


def __get_node_by_name(node, name):
    if node[__NAME__] == name:
        return node
    for item in node[__CHILDREN__]:
        ret = __get_node_by_name(item, name)
        if ret is not None:
            return ret
    return None


# search depth: 1.
def get_node_by_name_1(node, name):
    if node[__NAME__] == name:
        return node
    for child in node[__CHILDREN__]:
        if child[__NAME__] == name:
            return child
    return None


# search depth: 2.
def get_node_by_name_2(node, name):
    if node[__NAME__] == name:
        return node
    for child in node[__CHILDREN__]:
        if child[__NAME__] == name:
            return child
        for subchild in child[__CHILDREN__]:
            if subchild[__NAME__] == name:
                return subchild
    return None


# depth search 3
def get_node_by_name_3(node, name):
    if node[__NAME__] == name:
        return node
    for child in node[__CHILDREN__]:
        if child[__NAME__] == name:
            return child
        for subchild in child[__CHILDREN__]:
            if subchild[__NAME__] == name:
                return subchild
            for elem in subchild[__CHILDREN__]:
                if elem[__NAME__] == name:
                    return elem
    return None


# -- Returns only onde node (no wildcard) - Fast
def get_node_by_label(tree, label):
    """Return the first matching node with given type."""
    is_std = is_std_node(tree)
    if is_std >= 0:
        for item in tree:
            ret = get_node_by_label(item, label)
            if ret is not None:
                return ret
        return None
    return __get_node_by_label(tree, label)


def __get_node_by_label(node, label):
    if node[__LABEL__] == label:
        return node
    for child in node[__CHILDREN__]:
        ret = __get_node_by_label(child, label)
        if ret is not None:
            return ret
    return None


# One level of recursion, node is a pyTree
def get_node_by_label_1(node, label):
    if node[__LABEL__] == label:
        return node
    for child in node[__CHILDREN__]:
        if child[__LABEL__] == label:
            return child
    return None


# search with two level of recursion
def get_node_by_label_2(node, label):
    if node[__LABEL__] == label:
        return node
    for child in node[__CHILDREN__]:
        if child[__LABEL__] == label:
            return child
        for subchild in child[__CHILDREN__]:
            if subchild[__LABEL__] == label:
                return subchild
    return None


# node is a pyTree
def get_node_by_label_3(node, label):
    if node[__LABEL__] == label:
        return node
    for child in node[__CHILDREN__]:
        if child[__LABEL__] == label:
            return child
        for item in child[__CHILDREN__]:
            if item[__LABEL__] == label:
                return item
            for elem in item[__CHILDREN__]:
                if elem[__LABEL__] == label:
                    return elem
    return None


# -- node collections

# -- Returns a node list of node with the same 'label'.
# Traversal starts from node. Complete tree traversal.
def get_nodes_by_label(tree: TreeNode, label: str) -> List[TreeNode]:
    result = []
    is_std = is_std_node(tree)
    if is_std == 0:
        for item in tree:
            __get_nodes_by_label(item, label, result)
    else:
        __get_nodes_by_label(tree, label, result)
    return result


def __get_nodes_by_label(node, label, result):
    if node[__LABEL__] == label:
        result.append(node)
    for c in node[__CHILDREN__]:
        __get_nodes_by_label(c, label, result)


# Parcours 1 niveau de recursivite seulement
def get_close_nodes_by_label(node, label, depth=1):
    if depth == 1:
        func = __get_nodes_by_label_1
    elif depth == 2:
        func = __get_nodes_by_label_2
    elif depth == 3:
        func = __get_nodes_by_label_3
    else:
        raise ValueError("Request label search is too deep")
    result = []
    is_std = is_std_node(node)
    if is_std >= 0:
        for item in node:
            func(item, label, result)
    else:
        func(node, label, result)
    return result


def __get_nodes_by_label_1(node, label, result):
    if node[__LABEL__] == label:
        result.append(node)
    for child in node[__CHILDREN__]:
        if child[__LABEL__] == label:
            result.append(child)


def __get_nodes_by_label_2(node, label, result):
    if node[__LABEL__] == label:
        result.append(node)
    for child in node[__CHILDREN__]:
        if child[__LABEL__] == label:
            result.append(child)
        for subchild in child[__CHILDREN__]:
            if subchild[__LABEL__] == label:
                result.append(subchild)


def __get_nodes_by_label_3(node, label, result):
    if node[__LABEL__] == label:
        result.append(node)
    for child in node[__CHILDREN__]:
        if child[__LABEL__] == label:
            result.append(child)
        for d in child[__CHILDREN__]:
            if d[__LABEL__] == label:
                result.append(d)
            for e in d[__CHILDREN__]:
                if e[__LABEL__] == label:
                    result.append(e)


######
# -- Returns a list of nodes all having name 'name' and type 'label'
# Traversal starts from node
# Complete tree traversal. Wildcards allowed.
def get_nodes_by_name_and_label(tree, name, label):
    """Return a list of nodes matching given name and type."""
    result = []
    is_std = is_std_node(tree)
    if is_std >= 0:
        if ("*" in name) | ("?" in name) | ("[" in name) or ("*" in label) | (
            "?" in label
        ) | ("[" in label):
            for item in tree:
                __get_nodes_by_name_and_label_match(item, name, label, result)
        else:
            for item in tree:
                __get_nodes_by_name_and_label(item, name, label, result)
    else:
        if ("*" in name) | ("?" in name) | ("[" in name) or ("*" in label) | (
            "?" in label
        ) | ("[" in label):
            __get_nodes_by_name_and_label_match(tree, name, label, result)
        else:
            __get_nodes_by_name_and_label(tree, name, label, result)
    return result


def __get_nodes_by_name_and_label(node, name, label, result):
    if node[__NAME__] == name and node[__LABEL__] == label:
        result.append(node)
    for child in node[__CHILDREN__]:
        __get_nodes_by_name_and_label(child, name, label, result)


def __get_nodes_by_name_and_label_match(node, name, label, result):
    if fnmatch.fnmatch(node[__NAME__], name) and fnmatch.fnmatch(
        node[__LABEL__], label
    ):
        result.append(node)
    for child in node[__CHILDREN__]:
        __get_nodes_by_name_and_label_match(child, name, label, result)


# -- Returns only one node (no wildcard) - Fast
def get_node_by_name_and_label(tree, name, label):
    """Return the first matching node with given name and type."""
    is_std = is_std_node(tree)
    if is_std >= 0:
        for node in tree:
            ret = get_node_by_name_and_label(node, name, label)
            if ret is not None:
                return ret
        return None
    return __get_node_by_name_and_label(tree, name, label)


def __get_node_by_name_and_label(node, name, label):
    if node[__NAME__] == name and node[__LABEL__] == label:
        return node
    for item in node[__CHILDREN__]:
        ret = __get_node_by_name_and_label(item, name, label)
        if ret is not None:
            return ret
    return None


# Custom search
# -- Retourne les noeuds CGNSBase_t --
def get_bases(tree):
    """Return a list of all CGNSBase_t nodes."""
    result, label = [], "CGNSBase_t"
    is_std = is_std_node(tree)
    if is_std >= 0:
        for node in tree:
            __get_nodes_by_label_1(node, label, result)
    else:
        __get_nodes_by_label_1(tree, label, result)
    return result


# -- Returns Zone_t nodes --
def get_zones(tree):
    """Return a list of all Zone_t nodes."""
    result, label = [], "Zone_t"
    is_std = is_std_node(tree)
    if is_std >= 0:
        for node in tree:
            __get_nodes_by_label_2(node, label, result)
    else:
        __get_nodes_by_label_2(tree, label, result)
    return result


# alias
get_all_zone = get_zones


def get_child_value_by_name(node, name):
    return data(get_child_by_name(node, name))


# =============================================================================
# -- rm Nodes --
# =============================================================================


def rm_node(tree: TreeNode, node: TreeNode) -> NoReturn:
    """Search and remove a given node from t.

    :param tree: a CGNS/Python tree or a list of CGNS/Python nodes
    :param node: the node to remove
    :return: None
    """
    (parent, ichild) = get_parent_of_node(tree, node)
    if parent is not None:
        if is_std_node(tree) == 0 and parent is tree:
            del parent[ichild]
        else:
            del parent[__CHILDREN__][ichild]


def rm_child(parent: TreeNode, node: TreeNode) -> NoReturn:
    for num, item in enumerate(parent[__CHILDREN__]):
        if item == node:
            del parent[__CHILDREN__][num]
            return


def rm_child_by_name(parent: TreeNode, name: str) -> NoReturn:
    for num, node in enumerate(parent[__CHILDREN__]):
        if node[__NAME__] == name:
            del parent[__CHILDREN__][num]
            return


def rm_child_by_label(parent: TreeNode, label: str) -> NoReturn:
    for num, node in enumerate(parent[__CHILDREN__]):
        if node[__LABEL__] == label:
            del parent[__CHILDREN__][num]
            return


def rm_children_by_label(parent: TreeNode, label: str) -> NoReturn:
    to_delete = []
    for num, node in enumerate(parent[__CHILDREN__]):
        if node[__LABEL__] == label:
            to_delete.append(num)
    for num in reversed(to_delete):
        del parent[__CHILDREN__][num]


# Alternate version
# def rm_children_by_label(parent: TreeNode, label: str) -> NoReturn:
#    children = parent[__CHILDREN__]
#    children_range = range(len(children)-1, -1, -1)
#    for num in children_range:
#        if children[num][__LABEL__] == label:
#           del children[num]


def rm_children_by_labels(parent: TreeNode, labels: List[str]) -> NoReturn:
    to_delete = []
    for num, node in enumerate(parent[__CHILDREN__]):
        if node[__LABEL__] in labels:
            to_delete.append(num)
    for num in reversed(to_delete):
        del parent[__CHILDREN__][num]


def rm_nodes_by_name(tree: TreeNode, name: str) -> NoReturn:
    is_std = is_std_node(tree)
    if is_std == 0:
        for item in tree:
            __rm_nodes_by_name(item, name)
    else:
        __rm_nodes_by_name(tree, name)


def __rm_nodes_by_name(tree: TreeNode, name: str) -> NoReturn:
    to_delete = []
    for ichild, child in enumerate(tree[__CHILDREN__]):
        if is_name(child, name):
            to_delete.append(ichild)
        else:
            __rm_nodes_by_name(child, name)
    for ichild in reversed(to_delete):
        del tree[__CHILDREN__][ichild]


rm_nodes_from_name = rm_nodes_by_name  # alias


def rm_grandchildren_by_name(tree: TreeNode, name: str) -> NoReturn:
    child_list = tree[__CHILDREN__]
    child_range = range(len(child_list) - 1, -1, -1)
    for ichild in child_range:
        if child_list[ichild][__NAME__] == name:
            del child_list[ichild]
    for child in child_list:
        grandchildren = child[__CHILDREN__]
        node_range = range(len(grandchildren) - 1, -1, -1)
        for idx in node_range:
            if grandchildren[idx][__NAME__] == name:
                del grandchildren[idx]


# aliases
rm_nodes_by_name_1 = rm_child_by_name
rm_nodes_by_name_2 = rm_grandchildren_by_name

rm_nodes_by_label_1 = rm_children_by_label


# -- rm_nodes_by_label
def rm_nodes_by_label(tree: TreeNode, label: str) -> NoReturn:
    """Remove nodes of given type."""
    is_std = is_std_node(tree)
    if is_std == 0:
        for item in tree:
            __rm_nodes_by_label(item, label)
    else:
        __rm_nodes_by_label(tree, label)


def __rm_nodes_by_label(tree: TreeNode, label: str) -> NoReturn:
    child_range = range(len(tree[__CHILDREN__]) - 1, -1, -1)
    for ichild in child_range:
        if tree[__CHILDREN__][ichild][__LABEL__] == label:
            del tree[__CHILDREN__][ichild]
        else:
            __rm_nodes_by_label(tree[__CHILDREN__][ichild], label)


# -- rm_nodes_by_labels
def rm_nodes_by_labels(tree: TreeNode, labels: List[str]) -> NoReturn:
    """Remove nodes of given type."""
    is_std = is_std_node(tree)
    if is_std == 0:
        for item in tree:
            __rm_nodes_by_labels(item, labels)
    else:
        __rm_nodes_by_labels(tree, labels)


def __rm_nodes_by_labels(tree: TreeNode, labels: List[str]) -> NoReturn:
    child_range = range(len(tree[__CHILDREN__]) - 1, -1, -1)
    for ichild in child_range:
        if tree[__CHILDREN__][ichild][__LABEL__] in labels:
            del tree[__CHILDREN__][ichild]
        else:
            __rm_nodes_by_labels(tree[__CHILDREN__][ichild], labels)


def rm_grandchildren_by_label(tree: TreeNode, label: str) -> NoReturn:
    child_list = tree[__CHILDREN__]
    child_range = range(len(child_list) - 1, -1, -1)
    for ichild in child_range:
        if child_list[ichild][__LABEL__] == label:
            del child_list[ichild]
    for child in child_list:
        grandchildren = child[__CHILDREN__]
        node_range = range(len(grandchildren) - 1, -1, -1)
        for idx in node_range:
            if grandchildren[idx][__LABEL__] == label:
                del grandchildren[idx]


# alias
rm_nodes_by_label2 = rm_grandchildren_by_label


def rm_nodes_by_name_and_label(tree: TreeNode, name: str, label: str) -> NoReturn:
    is_std = is_std_node(tree)
    if is_std >= 0:
        for item in tree:
            __rm_nodes_by_name_and_label(item, name, label)
    else:
        __rm_nodes_by_name_and_label(tree, name, label)


def __rm_nodes_by_name_and_label(tree: TreeNode, name: str, label: str) -> NoReturn:
    child_list = tree[__CHILDREN__]
    children_range = range(len(child_list) - 1, -1, -1)
    for ichild in children_range:
        if child_list[ichild][__LABEL__] == label and is_name(child_list[ichild], name):
            del child_list[ichild]
        else:
            __rm_nodes_by_name_and_label(child_list[ichild], name, label)


def rm_node_by_path(tree: TreeNode, path: str) -> NoReturn:
    if path in ["", "/", "."]:
        # Cannot delete t itself
        return
    if path[0] == "/":
        path = path[1:]
    if path[-1] == "/":
        path = path[:-1]
    if tree[__NAME__] == path:
        # Cannot delete t itself
        return
    if tree[__LABEL__] == "CGNSTree_t":
        clean_path = path.replace(tree[__NAME__] + "/", "")
    else:
        clean_path = path
    tokens = clean_path.split("/")
    is_std = is_std_node(tree)
    if is_std >= 0:
        for node in tree:
            __rm_node_by_path(node, tokens[1:])
    else:
        __rm_node_by_path(tree, tokens)


def __rm_node_by_path(tree: TreeNode, paths: List[str]) -> NoReturn:
    def child_index(node, key):
        for num, item in enumerate(node[__CHILDREN__]):
            if item[__name__] == key:
                return num
        return None

    parent, idx = tree, None
    for token in paths:
        idx = child_index(parent, token)
        if idx is None:
            break
        parent = parent[idx]
    if idx:
        del parent[idx]


# ==============================================================================
# -- node copy --
# ==============================================================================

# -- Duplique un arbre ou un sous-arbre par references
def __duptree(node, parent):
    dup = [node[__NAME__], node[__DATA__], [], node[__LABEL__]]
    if len(parent) == 4:
        parent[__CHILDREN__].append(dup)
    for item in node[__CHILDREN__]:
        __duptree(item, dup)
    return dup


# -- Copy un arbre en gardant des references sur les numpy
def copy_ref(node):
    """Copy a tree sharing node values."""
    is_std = is_std_node(node)
    if is_std == -1:
        dup = __duptree(node, [])
        return dup
    if is_std == 0:
        dup_list = list(node)
        for idx, item in enumerate(dup_list):
            dup_list[idx] = __duptree(item, [])
        return dup_list
    return node


# To be coherent with python copy should be a shallow copy
# or else provide API copy(x, deep=True)
#
# -- deepcopy of CGNS/Python tree or subtree
def copy_tree(node, parent=None, order="F"):
    """Fully copy a tree."""
    if isinstance(node[__DATA__], numpy.ndarray):
        newn = [node[0], node[1].copy(order), [], node[3]]
    else:
        newn = [node[0], node[1], [], node[3]]
    if parent is not None and len(parent) == 4:
        parent[__CHILDREN__].append(newn)
    for child in node[__CHILDREN__]:
        copy_tree(child, newn, order=order)
    return newn


# -- Copy a node (deepcopy of data but not of children)
def copy_node(node, newname=None):
    """Copy only this node (no recursion)."""
    newn = [
        newname if newname else node[__NAME__],
        None,
        node[__CHILDREN__],
        node[__LABEL__],
    ]
    if isinstance(node[__DATA__], numpy.ndarray):
        newn[__DATA__] = node[__DATA__].copy("F")
    else:
        newn[__DATA__] = node[__DATA__]
    return newn


# ==============================================================================
# -- tree traversal --
# ==============================================================================


class TreeWalker:
    """Deep First Walker of pyTree"""

    def __init__(
        self,
        tree: TreeNode,
        condition: Callable[[TreeNode], bool],
        caching: bool = False,
    ):
        self._tree = tree
        self._condition = condition
        self._cache = []
        self._caching = caching

    def _gen_nodes(self, node: TreeNode) -> TreeNode:
        if self._condition(node):
            yield node
        for child in children(node):
            yield from self._gen_nodes(child)

    def items(self):
        """Generator of nodes with condition"""
        if self._caching and self._cache:
            for item in self._cache:
                yield item
            return
        for item in self._gen_nodes(self._tree):
            if self._caching:
                self._cache.append(item)
            yield item

    def walk(self, func: Callable, *args, **kwargs):
        """Walk through the tree and apply func to selected nodes"""
        for item in self.items():
            func(item, *args, **kwargs)

    def reset(self):
        """reset the cache"""
        self._cache = []


class TreeBreathFirstWalker:
    """Breath First Walker of pyTree"""

    def __init__(
        self,
        tree: TreeNode,
        condition: Callable[[TreeNode], bool],
        caching: Optional[bool] = False,
    ):
        self._tree = tree
        self._condition = condition
        self._cache = []
        self._caching = caching
        self._level = 100

    @staticmethod
    def _bfs(root: TreeNode, depth: int):
        import queue

        temp = queue.Queue()
        temp.put((0, root))
        while not temp.empty():
            level, node = temp.get()
            yield node
            if level == depth:
                continue
            for child in node[__CHILDREN__]:
                temp.put((level + 1, child))

    def _gen_nodes(self, node: TreeNode, depth: int) -> TreeNode:
        for item in self._bfs(node, depth):
            if self._condition(item):
                yield item

    def items(self):
        """Generator of nodes with condition"""
        if self._caching and self._cache:
            for item in self._cache:
                yield item
            return
        for item in self._gen_nodes(self._tree, self._level):
            if self._caching:
                self._cache.append(item)
            yield item

    def walk(self, func: Callable, *args, **kwargs):
        """Walk through the tree and apply func to selected nodes"""
        for item in self.items():
            func(item, *args, **kwargs)

    def reset(self):
        self._cache = []


class TreeDepthWalker:
    """Tree Walker with specific depth limit"""

    def __init__(
        self,
        tree: TreeNode,
        condition: Callable[[TreeNode], bool],
        level=None,
        caching: bool = False,
    ):
        self._tree = tree
        self._condition = condition
        self._cache = []
        self._caching = caching
        self._level = 100 if level is None else level

    def _gen_lvl_nodes(self, node, depth):
        if self._condition(node):
            yield node
        if depth == self._level:
            return
        for child in children(node):
            yield from self._gen_lvl_nodes(child, depth + 1)

    def items(self):
        """Generator of nodes with condition"""
        if self._caching and self._cache:
            for item in self._cache:
                yield item
            return
        for item in self._gen_lvl_nodes(self._tree, 0):
            yield item
            if self._caching:
                self._cache.append(item)

    def walk(self, func: Callable, *args, **kwargs):
        """Walk through the tree and apply func to selected nodes"""
        for item in self.items():
            func(item, *args, **kwargs)

    def reset(self):
        self._cache = []


class LabelWalker(TreeWalker):
    """Tree Walker wich select nodes based on label"""

    def __init__(self, tree: TreeNode, label=None, caching: bool = False):
        if isinstance(label, str):

            def condition(node):
                return node[__LABEL__] == label

        elif isinstance(label, list):

            def condition(node):
                return node[__LABEL__] in label

        else:
            raise TypeError("Unexpected label type")
        super().__init__(tree, condition=condition, caching=caching)


class LabelDepthWalker(TreeDepthWalker):
    """Tree Walker wich select nodes based on label with limited recursion level """

    def __init__(
        self, tree: TreeNode, label=None, level: int = None, caching: bool = False
    ):
        if isinstance(label, str):

            def condition(node):
                return node[__LABEL__] == label

        elif isinstance(label, list):

            def condition(node):
                return node[__LABEL__] in label

        else:
            raise TypeError("Unexpected label type")
        super().__init__(tree, condition=condition, level=level, caching=caching)


# Warning One shot generator functions:
def iter_children_by_label(node, label):
    """generator of children node by label"""
    for child in node[__CHILDREN__]:
        if child[__LABEL__] == label:
            yield child


def iter_children_by_labels(node, labels):
    """generator of children node by labels"""
    for child in node[__CHILDREN__]:
        if child[__LABEL__] in labels:
            yield child


def iter_nodes_by_labels(tree, typeset):
    """generator of pyTree nodes by labels"""

    def _gen_nodes_from_labels(typelist, cnode, cpath):
        for item in cnode:
            if item[__LABEL__] in typelist:
                yield ("%s/%s" % (cpath, item[__NAME__]), item)
            for curpath, curnode in _gen_nodes_from_labels(
                typelist, item[__CHILDREN__], "%s/%s" % (cpath, item[__NAME__])
            ):
                yield (curpath, curnode)

    if tree[__LABEL__] == "CGNSTree_t":
        start = ""
    else:
        start = "%s" % tree[__NAME__]
    for path, node in _gen_nodes_from_labels(typeset, tree[__CHILDREN__], start):
        yield (path, node)


def iter_shallow_nodes_by_labels(tree, typeset):
    """generator of pyTree nodes by labels
     don't go deeper when node label is found"""

    def _gen_shallow_nodes_from_labels(typelist, cnode, cpath):
        for item in cnode:
            if item[__LABEL__] in typelist:
                yield ("%s/%s" % (cpath, item[__NAME__]), item)
            else:
                for curpath, curnode in _gen_shallow_nodes_from_labels(
                    typelist, item[__CHILDREN__], "%s/%s" % (cpath, item[__NAME__])
                ):
                    yield (curpath, curnode)

    if tree[__LABEL__] == "CGNSTree_t":
        start = ""
    else:
        start = "%s" % tree[__NAME__]
    for path, node in _gen_shallow_nodes_from_labels(
        typeset, tree[__CHILDREN__], start
    ):
        yield (path, node)


def iter_nodes_by_labels_1(tree, typeset):
    if tree[__LABEL__] == "CGNSTree_t":
        start = ""
    else:
        start = "{}".format(tree[__NAME__])
    for child in tree[__CHILDREN__]:
        path = "{}/{}".format(start, child[__NAME__])
        if child[__LABEL__] in typeset:
            yield (path, child)


def iter_nodes_by_labels_2(tree, typeset):
    if tree[__LABEL__] == "CGNSTree_t":
        start = ""
    else:
        start = "{}".format(tree[__NAME__])
    for child in tree[__CHILDREN__]:
        path = "{}/{}".format(start, child[__NAME__])
        if child[__LABEL__] in typeset:
            yield (path, child)
        for subchild in child[__CHILDREN__]:
            if subchild[__LABEL__] in typeset:
                yield ("{}/{}".format(path, subchild[__NAME__]), subchild)


# ==============================================================================
# -- specific CGNS nodes creation, Class like ? --
# ==============================================================================

# -- Create a CGNS version node
def new_cgns_lib_version(version):
    """Create a CGNS version node for compatibility with MLL"""
    version_node = [
        "CGNSLibraryVersion",
        numpy.array([version], dtype=numpy.float32),
        [],
        "CGNSLibraryVersion_t",
    ]
    return version_node


class CGNSLibraryVersion:
    def __new__(cls, version):
        version_node = [
            "CGNSLibraryVersion",
            numpy.array([version], dtype=numpy.float32),
            [],
            "CGNSLibraryVersion_t",
        ]
        return version_node


# -- Create the root node of a tree
def new_root_node(name):
    """Create a root node of a pyTree."""
    return [name, None, [], "CGNSTree_t"]


class RootNode:
    def __new__(cls, name):
        return [name, None, [], "CGNSTree_t"]


# -- newCGNSTree
# create a pyTree
def new_tree(version=3.1):
    """Create a new pyTree."""
    cgnslib_node = new_cgns_lib_version(version)
    root_node = ["CGNSTree", None, [cgnslib_node], "CGNSTree_t"]
    return root_node


class CGNSTree:
    def __new__(cls, version=3.1):
        cgnslib_node = CGNSLibraryVersion(version)
        root_node = ["CGNSTree", None, [cgnslib_node], "CGNSTree_t"]
        return root_node


# -- newCGNSBase


def new_cgns_base(name="Base", cell_dim=3, phys_dim=3, parent=None):
    """Create a new Base node."""
    if cell_dim not in [1, 2, 3]:
        raise ValueError("Invalid cell dim")
    if phys_dim not in [1, 2, 3]:
        raise ValueError("Invalid physical dim")
    if phys_dim < cell_dim:
        raise ValueError("cell dim is greater than physical dim")
    if parent is None:
        node = create_node(
            name,
            "CGNSBase_t",
            value=numpy.array([cell_dim, phys_dim], dtype=numpy.int32, order="F"),
        )
    else:
        node = create_child(
            parent,
            name,
            "CGNSBase_t",
            value=numpy.array([cell_dim, phys_dim], dtype=numpy.int32, order="F"),
            check=True,
        )
    return node


class CGNSBase:
    label = "CGNSBase_t"

    def __new__(cls, name="Base", cell_dim=3, phys_dim=3, parent=None):
        if cell_dim not in [1, 2, 3]:
            raise ValueError("Invalid cell dim")
        if phys_dim not in [1, 2, 3]:
            raise ValueError("Invalid physical dim")
        if phys_dim < cell_dim:
            raise ValueError("cell dim is greater than physical dim")
        if parent is None:
            node = create_node(
                name,
                cls.label,
                value=numpy.array([cell_dim, phys_dim], dtype=numpy.int32, order="F"),
            )
        else:
            node = create_child(
                parent,
                name,
                cls.label,
                value=numpy.array([cell_dim, phys_dim], dtype=numpy.int32, order="F"),
                check=True,
            )
        return node


# -- newZone
def new_zone(name="Zone", zsize=None, ztype="Structured", family=None, parent=None):
    """Create a new Zone node."""
    zonetype_l = ["Null", "UserDefined", "Structured", "Unstructured"]
    if ztype not in zonetype_l:
        raise ValueError("Zone: ztype must be in {}".format(zonetype_l))
    if parent is None:
        node = create_node(name, "Zone_t", value=zsize)
    else:
        node = create_child(parent, name, "Zone_t", value=zsize, check=True)
    create_child(node, "ZoneType", "ZoneType_t", value=str_to_ascii_array(ztype))
    if family is not None:
        create_child(node, "FamilyName", "FamilyName_t", value=family)
    return node


class Zone:
    label = "Zone_t"

    def __new__(
        cls, name="Zone", zsize=None, ztype="Structured", family=None, parent=None
    ):
        zonetype_l = ["Null", "UserDefined", "Structured", "Unstructured"]
        if ztype not in zonetype_l:
            raise ValueError("Zone: ztype must be in {}".format(zonetype_l))
        if parent is None:
            node = create_node(name, cls.label, value=zsize)
        else:
            node = create_child(parent, name, cls.label, value=zsize, check=True)
        create_child(node, "ZoneType", "ZoneType_t", value=str_to_ascii_array(ztype))
        if family is not None:
            create_child(node, "FamilyName", "FamilyName_t", value=family)
        return node


# -- newDataArray
def new_data_array(name="Data", value=None, parent=None):
    """Create a new DataArray node."""
    if parent is None:
        node = create_node(name, "DataArray_t", value=value)
    else:
        node = create_child(parent, name, "DataArray_t", value=value, check=True)
    return node


class DataArray:
    label = "DataArray_t"

    def __new__(cls, name="Data", value=None, parent=None):
        if parent is None:
            node = create_node(name, cls.label, value=value)
        else:
            node = create_child(parent, name, cls.label, value=value, check=True)
        return node


# -- newGridCoordinates
def new_grid_coordinates(name=__GridCoordinates__, parent=None):
    """Create a GridCoordinates node."""
    if parent is None:
        node = create_node(name, "GridCoordinates_t")
    else:
        node = create_child(parent, name, "GridCoordinates_t", check=True)
    return node


class GridCoordinates:
    label = "GridCoordinates_t"

    def __new__(cls, name=__GridCoordinates__, parent=None):
        if parent is None:
            node = create_node(name, cls.label)
        else:
            node = create_child(parent, name, cls.label, check=True)
        return node
