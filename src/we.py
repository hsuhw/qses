#from __future__ import print_function
import sys
from graphviz import Digraph

# transformation function
def transformEmpty(L1,L2):
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

def transform(L1,L2):
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

# add transition (not just dictionary join)
def addTrans(trans,tr) :
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
def printout(w,res) :
  print
  print 'transform ', '(', ''.join(w[0]), ',', ''.join(w[1]), ')'
  for r in res:
    print '(', ''.join(r[0]), ',', ''.join(r[1]), ')'

# print transitions
def printTrans(tr) :
  for t in tr.keys():
    print t, ':', tr[t]

# print path
def printPath(tr,start,end):
  if start!=end and tr.has_key(start):
    print "node = ", start, ", next = ", tr[start]
    for s in tr[start]:
      print "==", s, "=="
      printPath(tr,s[1],end)
    print "-----", start
  else:
    print "----- no value"

# output transitions to a graphviz dot/png
def printDot(trans,root):
  weStr = root[0] + '=' + root[1]
  dot = Digraph(name = weStr, comment = weStr)
  for t in trans.keys():
    dot.node(str(t),str(t))
    for r in trans[t]:
      dot.edge(str(t),str(r[1]),r[0])
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
        if node2Count.has_key(s[1]):
          print "        nodeNo=" + str(node2Count[s[1]]) + ';'
        else:
          nodeCount += 1
          print "        nodeNo=" + str(nodeCount) + ';'
          node2Count[s[1]]=nodeCount
        queuedNode.add(s[1])
        print "      " + ifEnd
        rdnCount += 1
    else:
      for s in tmpLabl:
        if (s[0]!=""):
          print "      " + s[0] + ';'
        if node2Count.has_key(s[1]):
          print "      nodeNo=" + str(node2Count[s[1]]) + ';'
        else:
          nodeCount += 1
          print "      nodeNo=" + str(nodeCount) + ';'
          node2Count[s[1]]=nodeCount
        queuedNode.add(s[1])

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
  # check arguments
  if len(sys.argv) < 3 :
    print 'please give word equation with two arguments as the two words (and a third argument as a length constraint'
    quit()
  
  # check alphabets
  if not(sys.argv[1].isalpha() and sys.argv[2].isalpha()):
    print 'strings must be alphabets, where uppercase as constants and lowercases as variables'
    quit()
  
  # start
  we = (list(sys.argv[1]), list(sys.argv[2]))
  
  print 'The word equation: ', sys.argv[1] + '=' + sys.argv[2]
  print '  string1 in list:', we[0]
  print '  string2 in list:', we[1]
  
  if (len(sys.argv)==4): # we have a string length constraint
    lengthCons = sys.argv[3]
  else:
    lengthCons = ""
  
  #print we[0][0], '---', we[0][1:]
  
  pool = list()    # nodes (word equations) to be transformed
  visited = list() # processed nodes
  
  # data types for storing a counter automaton
  # state: a pair-tuple of two "list of char" as a word equation
  # transition: dictionary with key=state, value=set of pair tuple of (length statement,target state)
  finalS = ''.join(we[0]),''.join(we[1]) # final state
  initS  = '',''  # initial state (successful node)
  trans  = dict() # transition
  
  # 1st transformation
  visited += [we]
  tmp,transTmp = transform(we[0],we[1])
  printout(we,tmp)
  print transTmp
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
        printout(we,tmp)
        print transTmp
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
    printProg("UAutomizerC",trans,initS,finalS,lengthCons)
    print
    printProg("EldaricaC",trans,initS,finalS,lengthCons)
  else:
    print "No solution for the word equation."
#




if __name__ == "__main__":
  main()