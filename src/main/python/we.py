#from __future__ import print_function
import sys
import re
#import collections
from graphviz import Digraph

class Element:
  def __init__(self,value,type):
    self.value = value
    if (type=='variable'):
      self.type = 1
    elif (type=='connect_symbol'):
      self.type = 2
    elif (type=='empty'):
      self.type = 3
    else: # constant character (default)
      self.type = 0
  #def __init__(e):
  #  self.value = e.value
  #  self.type = e.type
  def get_type(self):
    if self.type==1:
      return 'variable'
    elif self.type==2:
      return 'connect_symbol'
    elif self.type==3:
      return 'empty'
    else:
      return 'constant'
  def get_value(self):
    return self.value
  def to_string(self):
    return self.get_type() + ':' + self.value
  def eq(self,e):
    return (self.value==e.value and self.type==e.type)
  def is_const(self):
    return self.type==0
  def is_var(self):
    return self.type==1
  def is_connect(self):
    return self.type==2
  def is_empty(self):
    return self.type==3

def string2elements(str):
  l = list(str)
  ret = []
  for s in l:
    ret.append(Element(s,'constant'))
  return ret

def isCommentLine(line):
  return re.match(r"^#.*$", line)

def isEmptyLine(line):
  return re.match(r"^\s*$", line)

def isString(str):
  return re.match(r"^\".*\"$",str) or re.match(r"^\'.*\'$",str)

def isVariable(str):
  return re.match(r"^[_$a-zA-z].*",str)

def processElements(strList):
  eList = []
  #print strList
  for s in strList:
    if isString(s):  # string
      eList += string2elements(s[1:len(s)-1])
    elif isVariable(s):
      eList.append(Element(s,'variable'))
    else:
      assert(False)  # fail, should not reach this line
  return eList

# Process a line of input and return a word equation
def processLine(line):
  strPair = line.split(' = ')  # separate left and right of a word equation
  #print strPair[0], ', ', strPair[1]
  assert(len(strPair)==2)
  return (processElements(strPair[0].split()),processElements(strPair[1].split()))
  #return (processElements(re.split('[ \t]+',strPair[0])),processElements(re.split('[ \t]+',strPair[1])))

# read word equation from command line
def readInputFromFile(arg):
  weList = []  # list of word equations
  lcList = []  # list of length constraints
  with open(arg[2]) as fp:
    lines = fp.readlines()
  for line in lines:
    line.strip()
    if isCommentLine(line):
      pass
    elif ('\"' in line):  # word equation, need process
      weList.append(processLine(line))
    else:  # length condition, just append
      lcList.append(line)
  return weList, lcList

# connect multiple word equations
def connectWordEqs(wes):
  ret = ([],[])
  for we in wes:
    ret[0].append(Element('#','connect_symbol'))
    ret[1].append(Element('#','connect_symbol'))
    ret = (ret[0]+we[0],ret[1]+we[1])
  ret = (ret[0][1:],ret[1][1:])
  return ret
#
def genWordEqutaionFull(we):
  return [e.to_string() for e in we[0]], ' = ', [e.to_string() for e in we[1]]

def genWordEqutaionSimple(we):
  return [e.get_value() for e in we[0]], ' = ', [e.get_value() for e in we[1]]
#
def genWordEqutaionCompact(we):
  return ''.join([e.get_value() for e in we[0]]) + ' = ' + ''.join([e.get_value() for e in we[1]])
#
def genStrKey(we):  # return a pair of string representing a word equation
  return ''.join([e.get_value() for e in we[0]]), ''.join([e.get_value() for e in we[1]])

# transformation function
def transformEmpty(L1,L2):
  if L1==[] and L2 == []:
    h1, h2 = Element('','empty'), Element('','empty')
  elif L1==[] and L2!=[]:
    h1, h2 = Element('','empty'), L2[0]
  elif L1!=[] and L2==[]:
    h1, h2 = L1[0], Element('','empty')
  else: # should not be reached
    print "error in transformEmpty: case should not be reached..."
    assert(False)
    #h1, h2 = L1[0], L2[0]
    #s = ('','')

  s = genStrKey((L1,L2))
  if h1.is_empty() and h2.is_var(): # case 1: empty-variable
    ret1 = [ ([],[c for c in L2 if not c.eq(h2)]) ]
    ret2 = { genStrKey(ret1[0]):{(h2.get_value()+"=0",h2.get_value()+"=\"\"",s)} }
  elif h1.is_var() and h2.is_empty(): # case 2: variable-empty
    ret1 = [ ([c for c in L1 if not c.eq(h1)],[]) ]
    ret2 = { genStrKey(ret1[0]):{(h1.get_value()+"=0",h1.get_value()+"=\"\"",s)} }
  else:  # case 3: constant-empty or empty-constant, no transform
    ret1 = []
    ret2 = dict()
  return ret1,ret2
#
def transform(L1,L2):
  # special case
  if L1 == [] or L2 == []:
    return transformEmpty(L1,L2)
  #elif L1[0].is_connect or L2[0].is_connect():
  #  return transformConn(L1,L2)

  # usual cases
  h1, h2 = L1[0], L2[0]
  s = genStrKey((L1,L2))
  print
  print 'h1, h2 in one transform step:'
  print 'to_string: ', h1.to_string(), ' , ', h2.to_string()
  print 'get_type:  ', h1.get_type(), ' , ', h2.get_type()
  print 'get_value: ', h1.get_value() + ',' + h2.get_value()
  print 'const?     ', h1.is_const(), ',', h2.is_const()
  print 's:         ', s
  if h1.is_const() and h2.is_const() and not(h1.eq(h2)): # case1: distinct constants
    ret1 = []
    ret2 = dict()
  elif (h1.is_connect() and not h2.is_connect()) or (h2.is_connect() and not h1.is_connect()): # connect element : other element
    ret1 = []
    ret2 = dict()
  elif h1.eq(h2): #case2: identical constants or variables (including connect element)
    ret1 = [ (L1[1:],L2[1:]) ]
    ret2 = { genStrKey(ret1[0]):{("","",s)} }
  elif h1.is_const() and h2.is_var() : # case3: constant-variable
    ret1 = [ ([c for c in L1 if not c.eq(h2)],[c for c in L2 if not c.eq(h2)]), ([h1 if c.eq(h2) else c for c in L1],[h1 if c.eq(h2) else c for c in L2]), ([v for sublist in [[h1,h2] if c.eq(h2) else [c] for c in L1] for v in sublist],[v for sublist in [[h1,h2] if c.eq(h2) else [c] for c in L2] for v in sublist]) ]
    ret2 = { genStrKey(ret1[0]):{(h2.get_value()+"=0",h2.get_value()+"=\"\"",s)}, genStrKey(ret1[1]):{(h2.get_value()+"=1",h2.get_value()+"=\""+h1.get_value()+"\"",s)}, genStrKey(ret1[2]):{(h2.get_value()+"="+h2.get_value()+"+1",h2.get_value()+"="+"\""+h1.get_value()+"\"+"+h2.get_value(),s)} }
  elif h1.is_var() and h2.is_const() : # case4: variable-constant
    ret1 = [ ([c for c in L1 if not c.eq(h1)],[c for c in L2 if not c.eq(h1)]), ([h2 if c.eq(h1) else c for c in L1],[h2 if c.eq(h1) else c for c in L2]), ([v for sublist in [[h2,h1] if c.eq(h1) else [c] for c in L1] for v in sublist],[v for sublist in [[h2,h1] if c.eq(h1) else [c] for c in L2] for v in sublist]) ]
    ret2 = { genStrKey(ret1[0]):{(h1.get_value()+"=0",h1.get_value()+"=\"\"",s)}, genStrKey(ret1[1]):{(h1.get_value()+"=1",h1.get_value()+"=\""+h2.get_value()+"\"",s)}, genStrKey(ret1[2]):{(h1.get_value()+"="+h1.get_value()+"+1",h1.get_value()+"="+"\""+h2.get_value()+"\"+"+h1.get_value(),s)} }
  elif h1.is_var() and h2.is_var() : # case5: variable-variable
    ret1 = [ ([c for c in L1 if not c.eq(h1)],[c for c in L2 if not c.eq(h1)]), ([c for c in L1 if not c.eq(h2)],[c for c in L2 if not c.eq(h2)]), ([h1 if c.eq(h2) else c for c in L1],[h1 if c.eq(h2) else c for c in L2]), ([v for sublist in [[h2,h1] if c.eq(h1) else [c] for c in L1] for v in sublist],[v for sublist in [[h2,h1] if c.eq(h1) else [c] for c in L2] for v in sublist]), ([v for sublist in [[h1,h2] if c.eq(h2) else [c] for c in L1] for v in sublist],[v for sublist in [[h1,h2] if c.eq(h2) else [c] for c in L2] for v in sublist]) ]
    ret2 = { genStrKey(ret1[0]):{(h1.get_value()+"=0",h1.get_value()+"=\"\"",s)}, genStrKey(ret1[1]):{(h2.get_value()+"=0",h2.get_value()+"=\"\"",s)}, genStrKey(ret1[2]):{(h2.get_value()+"="+h1.get_value(),h2.get_value()+"="+h1.get_value(),s)}, genStrKey(ret1[3]):{(h1.get_value()+"="+h2.get_value()+"+"+h1.get_value(),h1.get_value()+"="+h2.get_value()+"+"+h1.get_value(),s)}, genStrKey(ret1[4]):{(h2.get_value()+"="+h1.get_value()+"+"+h2.get_value(),h2.get_value()+"="+h1.get_value()+"+"+h2.get_value(),s)} }
  else:
#    print 'Error, wrong case'
    ret1 = []
    ret2 = dict()
  return ret1,ret2

# add transition (not just dictionary join)
def addTrans(trans,tr) :
  print "processing addTrans..."
  print "--- trans.keys() ---"
  print trans.keys()
  print "--- tr.keys() ---"
  print tr.keys()
  print "--- has_key? ---"
  for k in tr.keys() :
    if trans.has_key(k) :
      print "(T)in trans? ", k, trans.has_key(k)
      trans[k] = trans[k].union(tr[k])
      print "trans[k]: ", trans[k]
      print "tr[k]:    ", tr[k]
      print "after union: ", trans[k]
    else :
      print "(F)in trans? ", k, trans.has_key(k)
      trans[k] = tr[k]
  print "--- ---"

# print out the result of a transformation
def printOneStep(w,res,trans) :
  print
  print 'printOneStep'
  print 'transform from', genStrKey(w)
  #print '-'.join([e.to_string() for e in w[0]])
  #print '-'.join([e.to_string() for e in w[1]])
  print 'successors:'
  for r in res:
    print genStrKey(r)
  print 'transitions:'
  printTrans(trans)

# print transitions
def printTrans(tr) :
  for t in tr.keys():
    print t, ':', tr[t], 'length=', len(tr[t])
    for s in tr[t]:
      assert len(s)==3
    

# print path
def printPath(tr,start,end):
  if start!=end and tr.has_key(start):
    print "node = ", start, ", next = ", tr[start]
    for s in tr[start]:
      print "==", s, "=="
      printPath(tr,s[2],end)
    print "-----", start
  else:
    print "----- no path"

# output transitions to a graphviz dot/png
def printDot(trans,root):
  weStr = root[0] + '=' + root[1]
  dot = Digraph(name = weStr, comment = weStr)
  for t in trans.keys():
    dot.node(str(t),str(t))
    for r in trans[t]:
      dot.edge(str(t),str(r[2]),r[0]+'\n'+r[1])
  print dot.source
  dot.render()

# print out interProc program
def printProg(type,tr,init,final,lengthCons):
  # check type validity
  if (type!="interProc" and type!="UAutomizerC" and type!="EldaricaC"):
    print "Type Error: type should be specified to \"interProc\" or \"UAutomizerC\" or \"EldaricaC\""
    print "No program output..."
    return

  # set some syntax keywords according to type
  if (type=="interProc"):
    progStart = "begin"
    progEnd = "end"
    whileStart = "do"
    whileEnd = "done;"
    ifStart = " then"
    ifEnd = "endif;"
    randomDecl = "      rdn = random;"
  elif (type=="UAutomizerC" or type=="EldaricaC"):
    progStart = ""
    progEnd = "}"
    whileStart = "{"
    whileEnd = "}"
    ifStart = " {"
    ifEnd = "}"
    randomDecl = "      rdn =  __VERIFIER_nondet_int();"
  
  # preprocessing, middle variables declaration
  visitedNode = set()
  node2Count = dict()
  queuedNode = set()
  vars = set()
  nodeCount = 0
  for s in list(final[0]+final[1]):
    if s.islower():
      vars.add(s)

  # variable declaration
  if (type=="interProc"):
    print "var "
    for s in vars:
      print s + ': int,'
    print "rdn: int,"
    print "nodeNo: int,"
    print "reachFinal: int;"
  elif (type=="UAutomizerC"):
    print "extern void __VERIFIER_error() __attribute__ ((__noreturn__));"
    print "extern int __VERIFIER_nondet_int(void);"
    print
    print "int main() {"
    for s in vars:
      print "  int " + s + ';'
    print "  int rdn, nodeNo, reachFinal;"
  elif (type=="EldaricaC"):
    print "int __VERIFIER_nondet_int(void) { int n=_; return n; }"
    print
    print "int main() {"
    for s in vars:
      print "  int " + s + ';'
    print "  int rdn, nodeNo, reachFinal;"

  # program begins
  print progStart
  print "  nodeNo = " + str(nodeCount) + ';'  # set nodeNo to zero (initial node)
  print "  reachFinal = 0;"
  print "  while (reachFinal==0) " + whileStart
  # start traverse from init node to final node
  queuedNode.add(init)
  while(len(queuedNode)>0):
    tmpNode = queuedNode.pop()
    # cases of node
    if (tmpNode in visitedNode): # already processed: skip to next loop
      continue
    else:
      visitedNode.add(tmpNode)

    if (tmpNode==init): # this is the initial node
      print "    if (nodeNo==" + str(nodeCount) + ") " + ifStart  # nodeCount = 0 (the first loop)
      print "    /* node = ", tmpNode, " */"
    else:
      print "    if (nodeNo==" + str(node2Count[tmpNode]) + ") " + ifStart  # node2Count must has key "tmpNode"
      print "    /* node = ", tmpNode, " */"
      if (tmpNode==final): # this is the final node
        print "      reachFinal=1;"
        print "    " + ifEnd
        continue

    tmpLabl = tr[tmpNode]
    tmpLen  = len(tmpLabl)

    if (tmpLen>1):  # two or more parent nodes # currently not completed
      print randomDecl
      #print "      assume rdn>=1 and rdn <=" + str(tmpLen) + ';'
      rdnCount = 1  # start from 1
      for s in tmpLabl:
        if (rdnCount==1):
          print "      if (rdn<=" + str(rdnCount) + ") " + ifStart
        elif (rdnCount==tmpLen):
          print "      if (rdn>=" + str(rdnCount) + ") " + ifStart
        else:
          print "      if (rdn==" + str(rdnCount) + ") " + ifStart
        if (s[0]!=""):
          print "        " + s[0] + ';'
        if (s[1]!=""):
          print "        //" + s[0] + ';'  #infomation for retrieving solution
        if node2Count.has_key(s[2]):
          print "        nodeNo=" + str(node2Count[s[2]]) + ';'
        else:
          nodeCount += 1
          print "        nodeNo=" + str(nodeCount) + ';'
          node2Count[s[2]]=nodeCount
        queuedNode.add(s[2])
        print "      " + ifEnd
        rdnCount += 1
    else:
      for s in tmpLabl:
        if (s[0]!=""):
          print "      " + s[0] + ';'
        if node2Count.has_key(s[2]):
          print "      nodeNo=" + str(node2Count[s[2]]) + ';'
        else:
          nodeCount += 1
          print "      nodeNo=" + str(nodeCount) + ';'
          node2Count[s[2]]=nodeCount
        queuedNode.add(s[2])

    print "    " + ifEnd
  print "  " + whileEnd
  if (type=="UAutomizerC" and lengthCons!=""): # length constraint (for UAutomizer)
    print "  if (" + lengthCons + ") { //length constraint: " + lengthCons
    print "    ERROR: __VERIFIER_error();"
    print "  }"
    print "  else {"
    print "    return 0;"
    print "  }"
  if (type=="EldaricaC" and lengthCons!=""): # length constraint (for UAutomizer)
    print "  assert (!(" + lengthCons + ")); //length constraint: " + lengthCons
  print progEnd



# the main function
def main():
  we = ([],[])
  lengthCons = str()
  # check option
  if (len(sys.argv)==3 and sys.argv[1]=='-f'):
    weList, lcList = readInputFromFile(sys.argv)
    print str(len(weList)) + " word equations read:"
    for (e1,e2) in weList:
      #print [e.to_string() for e in e1], [e.to_string() for e in e2]
      print genWordEqutaionCompact((e1,e2))
    we = connectWordEqs(weList)
    print
    print "after connected:"
    #print genWordEqutaionFull(we)
    #print genWordEqutaionSimple(we)
    print genWordEqutaionCompact(we)
    print
    print 'lenth constraints read:'
    for lc in lcList:
      print lc
  else:
    print 'usage: python we.py -f spec-file'
    print '  spec-file contains word equations (one equation one line) and length constraints (one equation one line)'
    print '  each line will be treated as conjunction connected'
    print
    quit()
    #we, lengthCons = readInputFromCmd(sys.argv)

  pool = list()    # nodes (word equations) to be transformed
  visited = list() # processed nodes
  
  # data types for storing a counter automaton
  # state: a pair-tuple of two strings as a word equation, each string is converted from the corresponding list of elements.
  # transition: dictionary with key=state, value=a tuple of (length statement,target state)
  # transition: dictionary with key=state, value=a tuple of (length statement,string statement,target state)
  #             length statement will be used in solving length constraint
  #             string statement will be used to get the solution of variables
  #
  finalS = genStrKey(we) # final state
  initS  = '',''  # initial state (successful node)
  trans  = dict() # transition
  
  # 1st transformation
  visited += [we]
  tmp,transTmp = transform(we[0],we[1])
  printOneStep(we,tmp,transTmp)
  addTrans(trans,transTmp)
  pool += tmp
  
  # rest transformations
  while pool!=[]:
    we = pool[0]
    pool = pool[1:]
    if we not in visited: # skip visited node
      if not(we[0]==[] and we[1]==[]):
        visited += [we]
        tmp,transTmp = transform(we[0],we[1])
        printOneStep(we,tmp,transTmp)
        addTrans(trans,transTmp)
        pool += tmp
  #  print '\ncurrent status'
  #  print 'pool', pool
  #  print 'visited', visited
  print
  print
  printTrans(trans)
  #print
  #printPath(trans,initS,finalS)
  #print
  #printDot(trans,finalS)
  print
  if trans.has_key(initS):
    #printProg("interProc",trans,initS,finalS,lengthCons)
    print
    #printProg("UAutomizerC",trans,initS,finalS,lengthCons)
    print
    #printProg("EldaricaC",trans,initS,finalS,lengthCons)
  else:
    print "No solution for the word equation."
#

if __name__ == "__main__":
  main()
#

"""
def transformEmptyCharVer(L1,L2):
  if L1==[] and L2 == []:
    h1, h2 = '', ''
    s = ('','')
  elif L1==[] and L2!=[]:
    h1, h2 = '', L2[0]
    s = ('',''.join(L2))
  elif L1!=[] and L2==[]:
    h1, h2 = L1[0], ''
    s = (''.join(L1),'')
  else: # should not be reached
    print "error in transformEmpty: case should not be reached..."
    h1, h2 = L1[0], L2[0]
    s = ('','')

  if h1 == '' and h2.islower(): # case 1: empty-variable
    ret1 = [ ([c for c in L1 if c!=h2],[c for c in L2 if c!=h2]) ]
    ret2 = { ('',''.join(ret1[0][1])):{(h2+"=0",s)} }
  elif h2 == '' and h1.islower(): # case 2: variable-empty
    ret1 = [ ([c for c in L1 if c!=h1],[c for c in L2 if c!=h1]) ]
    ret2 = { (''.join(ret1[0][0]),''):{(h1+"=0",s)} }
  else:
    ret1 = []
    ret2 = dict()
  return ret1,ret2

def transformCharVer(L1,L2):
  if L1 == [] or L2 == []:
    return transformEmpty(L1,L2)
  h1, h2 = L1[0], L2[0]
  s = (''.join(L1),''.join(L2))
  if h1.isupper() and h2.isupper() and not(h1==h2): # case1: distinct constants
    ret1 = []
    ret2 = dict()
  elif h1 == h2: #case2: identical constants or variables
    ret1 = [ (L1[1:],L2[1:]) ]
    ret2 = { (''.join(ret1[0][0]),''.join(ret1[0][1])):{("",s)} }
  elif h1.isupper() and h2.islower() : # case3: constant-variable
    ret1 = [ ([c for c in L1 if c!=h2],[c for c in L2 if c!=h2]), ([h1 if c==h2 else c for c in L1],[h1 if c==h2 else c for c in L2]), ([v for sublist in [[h1,h2] if c==h2 else [c] for c in L1] for v in sublist],[v for sublist in [[h1,h2] if c==h2 else c for c in L2] for v in sublist]) ]
    ret2 = { (''.join(ret1[0][0]),''.join(ret1[0][1])):{(h2+"=0",s)}, (''.join(ret1[1][0]),''.join(ret1[1][1])):{(h2+"=1",s)}, (''.join(ret1[2][0]),''.join(ret1[2][1])):{(h2+"="+h2+"+1",s)} }
  elif h1.islower() and h2.isupper() : # case4: variable-constant
    ret1 = [ ([c for c in L1 if c!=h1],[c for c in L2 if c!=h1]), ([h2 if c==h1 else c for c in L1],[h2 if c==h1 else c for c in L2]), ([v for sublist in [[h2,h1] if c==h1 else [c] for c in L1] for v in sublist],[v for sublist in [[h2,h1] if c==h1 else c for c in L2] for v in sublist]) ]
    ret2 = { (''.join(ret1[0][0]),''.join(ret1[0][1])):{(h1+"=0",s)}, (''.join(ret1[1][0]),''.join(ret1[1][1])):{(h1+"=1",s)}, (''.join(ret1[2][0]),''.join(ret1[2][1])):{(h1+"="+h1+"+1",s)} }
  elif h1.islower() and h2.islower() : # case5: variable-variable
    ret1 = [ ([c for c in L1 if c!=h1],[c for c in L2 if c!=h1]), ([c for c in L1 if c!=h2],[c for c in L2 if c!=h2]), ([h1 if c==h2 else c for c in L1],[h1 if c==h2 else c for c in L2]),	([v for sublist in [[h2,h1] if c==h1 else [c] for c in L1] for v in sublist],[v for sublist in [[h2,h1] if c==h1 else c for c in L2] for v in sublist]), ([v for sublist in [[h1,h2] if c==h2 else [c] for c in L1] for v in sublist],[v for sublist in [[h1,h2] if c==h2 else c for c in L2] for v in sublist]) ]
    ret2 = { (''.join(ret1[0][0]),''.join(ret1[0][1])):{(h1+"=0",s)}, (''.join(ret1[1][0]),''.join(ret1[1][1])):{(h2+"=0",s)}, (''.join(ret1[2][0]),''.join(ret1[2][1])):{(h2+"="+h1,s)}, (''.join(ret1[3][0]),''.join(ret1[3][1])):{(h1+"="+h1+"+"+h2,s)}, (''.join(ret1[4][0]),''.join(ret1[4][1])):{(h2+"="+h1+"+"+h2,s)} }
  else:
#    print 'Error, wrong case'
    ret1 = []
    ret2 = dict()
  return ret1,ret2

# read word equation from command line
def readWordEqFromCmd(arg):
  # check number of arguments
  if len(arg) < 3 :
    print 'please give word equation with two arguments as the two words (and a optional third argument as a length constraint)'
    quit()
  # check alphabets
  if not(arg[1].isalpha() and arg[2].isalpha()):
    print 'strings must be alphabets, where uppercase as constants and lowercases as variables'
    quit()
  # set word equation
  we = (list(arg[1]), list(arg[2]))
  print 'The word equation: ', arg[1] + '=' + arg[2]
  print '  string1 in list:', we[0]
  print '  string2 in list:', we[1]
  if (len(arg)==4): # we have a string length constraint
    lengthCons = arg[3]
  else:
    lengthCons = ""
  #print we[0][0], '---', we[0][1:]
  return we, lengthCons
#



"""






