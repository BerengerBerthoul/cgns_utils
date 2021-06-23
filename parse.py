from pyparsing  import alphas, alphas, nums
from pyparsing  import Literal, Word, Keyword
from pyparsing  import Regex
from pyparsing  import Group, originalTextFor, nestedExpr
from pyparsing  import Suppress, ZeroOrMore, Optional, Combine
from pyparsing  import ParseException

import re

body = """

List of functions
#################

**-- Node tests**

.. autosummary::

   Converter.Internal.isTopTree
   Converter.Internal.isStdNode
   Converter.Internal.typeOfNode
   Converter.Internal.isType
   Converter.Internal.isName
   Converter.Internal.isValue
   Converter.Internal.isChild
"""

with open("doc/source/Internal.rst", "r") as f:
  body = f.read()
# print(f"body : {body}")

var_ident = Regex("([a-zA-Z0-9_]*)")
var_def = Keyword('Converter') + Regex('.') + Keyword('Internal') + Regex('.') + var_ident.setResultsName("funcname")

found_in_doc = []
for (token, start, end) in var_def.scanString(body):
  if token.funcname:
    found_in_doc.append(str(token.funcname))
# sys.exit(1)

body = """
def isStdNode(node):
    if not isinstance(node, list): return -2
    if len(node) == 0: return 0
    node0 = node[0]
    if (isinstance(node0, str) and len(node) == 4 and
        isinstance(node[2], list)): return -1
    if not isinstance(node0, list): return -2
    if (isinstance(node0[0], str) and len(node0) == 4
        and isinstance(node0[2], list)): return 0
    return -2

def getNodesFromName(t, name):
    result = []
    isStd = isStdNode(t)
    if isStd >= 0:
        if ('*' in name)|('?' in name)|('[' in name):
            for c in t[isStd:]: getNodesFromName___(c, name, result)
        else:
            for c in t[isStd:]: getNodesFromName__(c, name, result)
    else:
        if ('*' in name)|('?' in name)|('[' in name):
            getNodesFromName___(t, name, result)
        else: getNodesFromName__(t, name, result)
    return result

  def getBCDataSet(z, bcNode, withLoc=False):
    datas = []; ploc = 'Vertex'
    # Try from BCDataSet#EndOfRun (etc)
    dataSet = getNodeFromName1(bcNode, 'BCDataSet#EndOfRun')
    if dataSet is not None:
        #print('found new etc style dataSet')
        datas = getNodesFromType2(dataSet, 'DataArray_t')
        if withLoc:
            l = 'Vertex'
            l = getNodeFromType1(dataSet, 'GridLocation_t')
            if l is not None: ploc = getValue(l)
            return datas,ploc
        else: return datas

"""

with open("converter/Internal.py", "r") as f:
  body = f.read()
# print(f"body : {body}")

def camel_to_snake(text, keep_upper=False):
  ptou    = re.compile(r'(2)([A-Z]+)([A-Z][a-z])')
  ptol    = re.compile(r'(2)([A-Z][a-z])')
  tmp = re.sub(ptol, r'_to_\2', re.sub(ptou, r'_to_\2', text))
  pupper = re.compile(r'([A-Z]+)([A-Z][a-z])')
  plower = re.compile(r'([a-z\d])([A-Z])')
  word = plower.sub(r'\1_\2', re.sub(pupper, r'\1_\2', tmp))
  if keep_upper:
    return '_'.join([w if all([i.isupper() for i in w]) else w.lower() for w in word.split('_')])
  else:
    return word.lower()

var_ident = Regex("([a-zA-Z0-9_]*)")
var_def = Keyword('def') + var_ident.setResultsName("funcname") + Regex('\(')

pcc = r"(?<=\^|[\^_a-zA-Z])_*[a-z]+[_a-zA-Z]*"
results = {}
for (token, start, end) in var_def.scanString(body):
  if token.funcname:
    print(f"token.funcname = {token.funcname}")
    funcname = str(token.funcname)
    # if funcname in found_in_doc and not funcname.endswith('__'):
    if not funcname.endswith('__'):
      m = re.search(pcc, funcname)
      if m:
        snake_funcname = camel_to_snake(funcname)
        if funcname != snake_funcname:
          results[camel_to_snake(funcname)] = funcname
print(f"results : {results}")

# Add
additional = {
# 'is_cgns_tree'  : 'is_top_tree',
'is_child_of'   : 'is_child',
# 'is_same_label' : 'is_type',
# 'is_same_name'  : 'is_name',
# 'is_same_value' : 'is_value',
}
for k,v in additional.items():
  results[k] = results[v]

# Transform type -> label
add_results = {}
for k,v in results.items():
  if 'type' in k and not k.startswith('new') and k not in list(additional.values())+['type_of_node']:
    print(f"Found type in = {k}")
    add_results[k.replace('type', 'label')] = results[k]

size_snake_word = max([len(w) for w in results.keys()])
size_camel_word = max([len(w) for w in results.values()])
with open("converter/internal.py", "w") as f:
  f.write(f"from . import Internal as I\n")
  f.write(f"from .additional import *\n\n")
  for k, v in results.items():
    if v not in list(add_results.values())+["pointList2Windows"]: # remove all ..._type method(s)
      f.write(f"{k.ljust(size_snake_word)} = I.{v.ljust(size_camel_word)};  {v.ljust(size_camel_word)} = I.{v};\n")
  f.write(f"\n")
  f.write(f"# Additional(s)\n")
  for k, v in add_results.items():
    f.write(f"{k.ljust(size_snake_word)} = I.{v.ljust(size_camel_word)};  {v.ljust(size_camel_word)} = I.{v};\n")

