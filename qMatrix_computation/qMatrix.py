#!/usr/bin/python3
import time
tStart = time.time()
tStartTotal = tStart

from qFunction import qMatrix, qMatrixUsingTree, qMatrixUsingTreeFast, projectFeatures, loadTableFromCsv, DATA_AND_SET_SIZES
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys



def showHelp():
  print(f"{sys.argv[0]} [--help]")
  print("          [-o=outputTable]")
  print("          [-tree] [-tree-fast] [-log]")
  print("          [-op=outputPointList] [-oi=outputImage] [-omnb=outputModularityImage] [-cool-down=float]")
  print("          [-numbered] [-pandas]")
  print("          [-i=columnName] inputTable")
  print("")
  print("---[ General ]--------------------------------------------------------------")
  print(" --help             : shows the help of the program.")
  print("")
  print("---[ Q-matrix generation ]--------------------------------------------------")
  print(" -o=outputTable     : writes the Q-matrix as CSV-file. If the file")
  print("                      name is \"-\" then the data will be print to stdout.")
  print("")
  print(" -tree              : uses the tree approach instead computing the")
  print("                      Q-matrix brute force.")
  print("")
  print(" -tree-fast         : uses the fast version of the tree approach")
  print("                      instead computing the Q-matrix brute force.")
  print("")
  print(" -log               : prints debug information after computing the data.")
  print("")
  print("---[ dependency projection ]------------------------------------------------")
  print(" -op=outputPointList: writes the position of the projected feature")
  print("                      points to a CSV-file. If the file name is \"-\"")
  print("                      then the data will be print to stdout.")
  print("")
  print(" -oi=outputImage    : writes the projected feature points to an image-file")
  print("                      (PNG or PDF). If the file name is \"-\" then the")
  print("                      image is shown in a window.")
  print("")
  print(" -cool-down=c       : The coolDown value for the projection phase.")
  print("                      Default is 0.4. The value is expected to be")
  print("                      0 <= c < 1. Smaller values will tend more to be")
  print("                      a line and the probability that bijective dependent")
  print("                      features land at the same point increases.")
  print("                      Greater values tend more to be a cloud.")
  print("")
  print("---[ prediction ]-----------------------------------------------------------")
  print(" -layer2=outputTable   : predicts the second layer. Actual q-Values for")
  print("                       q((a,b), c) are less ore equal to the predicted values.")
  print("")
  print(" -functions=outputTable : predicts the functions for a -> b and (a,b) -> c.")
  print("")
  print("---[ data loader ]----------------------------------------------------------")
  print("")
  print(" -numbered          : Use own data loader for faster table access. (default)")
  print("")
  print(" -pandas            : Use pandas data loader for comparability.")
  print("")
  print("---[ data input ]-----------------------------------------------------------")
  print(" -i=columnName      : Ignores the given column during the computation.")
  print("                      This parameter can be given multiple times.")
  print("")
  print(" inputTable         : A file name of a CSV-file. This first line is")
  print("                      expected to be column names.")


def cleanupName(name):
  return (f"{name}"
    .replace(",", ";")
    .replace("\r\n", " ")
    .replace("\n\r", " ")
    .replace("\r", " ")
    .replace("\n", " ")
    )

def writeMatrix(matrix, columns, f):
  print("," + ",".join(columns), file=f)
  for row, c in zip(matrix, columns):
    print(",".join([c] + [f"{v}" for v in row]), file=f)
  
def writeMatrix2(matrix, columns, f):
  print("," + ",".join(columns), file=f)
  pColumns = []
  for i in range(len(columns)):
    for j in range(i):
      pColumns.append(f"'{columns[i]}|{columns[j]}'")
  for row, c in zip(matrix, pColumns):
    print(",".join([c] + [f"{v}" for v in row]), file=f)
  
def writePoints(matrix, columns, f):
  print("Column,X,Y", file=f)
  for row, c in zip(matrix, columns):
    print(",".join([c] + [f"{v}" for v in row]), file=f)
  

def checkForOldFlags(name):
  replacements = [
    ( '--coolDown', '-cool-down' )
    , ( '--tree', '-tree' )
    , ( '--tree-fast', '-tree-fast' )
    , ( '--log', '-log' )
    ]

  for (o, n) in replacements:
    if name == o:
      print(f"The parameter '{o}' was removed.")
      print(f"Please use '{n}' instead.")
      showHelp()
      exit(1)

def timeStep(tStart, title):
  now = time.time()
  if debug:
    print(f"{title}: {now - tStart}s", file=sys.stderr)
  return now

if __name__ == "__main__":
  inFileName = None
  outFileName = None
  outFileNameImage = None
  outFileNamePoints = None
  outFileNameDendrogram = None
  outFileNameUmap = None
  outFileNameModularityNotebook = None
  outFileName2ndlayer = None
  outFileNameFunctions = None
  dropColumns = []
  doLog = False
  useTree = False
  useTreeFast = False
  useNumberedData = True
  coolDown = 0.4
  debug = False

  n = 1
  for a in sys.argv[1:]:
    checkForOldFlags(a)

    if a == '-o':
      outFileName = "-"
      continue

    if a.startswith('-o='):
      outFileName = a[3:]
      continue

    if a == '-oi':
      outFileNameImage = "-"
      continue

    if a.startswith('-oi='):
      outFileNameImage = a[4:]
      continue

    if a == '-op':
      outFileNamePoints = "-"
      continue

    if a.startswith('-op='):
      outFileNamePoints = a[4:]
      continue

    if a == '-od':
      outFileNameDendrogram = "-"
      continue

    if a.startswith('-od='):
      outFileNameDendrogram = a[4:]
      continue

    if a == '-ou':
      outFileNameUmap = "-"
      continue

    if a.startswith('-ou='):
      outFileNameUmap = a[4:]
      continue

    if a == '-omnb':
      outFileNameModularityNotebook = "-"
      continue
      
    if a.startswith('-omnb='):
      outFileNameModularityNotebook = a[6:]
      continue

    if a == '-layer2':
      outFileName2ndlayer = "-"
      continue

    if a.startswith('-layer2='):
      outFileName2ndlayer = a[8:]
      continue

    if a == '-functions':
      outFileNameFunctions = "-"
      continue

    if a.startswith('-functions='):
      outFileNameFunctions = a[11:]
      continue

    if a.startswith('-i='):
      dropColumns.append(a[3:])
      continue

    if a == '-numbered':
      useNumberedData = True
      continue

    if a == '-pandas':
      useNumberedData = False
      continue

    if a.startswith('-cool-down='):
      coolDown = float(a[11:])
      if coolDown < 0 or coolDown >= 1:
        print(f"coolDown is expected to be >= 0 and < 1 but {coolDown} was given.")
        exit(1)
      continue

    if a == '-tree':
      useTree = True
      continue

    if a == '-tree-fast':
      useTreeFast = True
      continue

    if a == '-log':
      doLog = True
      continue

    if a == '-debug':
      debug = True
      continue

    if a == '--help' or a == '-h':
      showHelp()
      exit(0)

    inFileName = a

  if inFileName is None:
    showHelp()
    exit(1)

  tStart = timeStep(tStart, "Initialization")
  if useNumberedData:
    tl = loadTableFromCsv(inFileName, DATA_AND_SET_SIZES, dropColumns)
    columns = tl.heading
    data = tl.data
    del tl
  else:
    data = pd.read_csv(inFileName)
    if len(dropColumns) > 0:
      cs = list(data.columns)
      dc = [c for c in dropColumns if c in cs]
      data = data.drop(columns=dc)
      del cs
      del dc
      del dropColumns
    columns = [cleanupName(n) for n in list(data.columns)]
    data = np.array(data)

  tStart = timeStep(tStart, "Load data")
  if useTree:
    matrix, qf = qMatrixUsingTree(data, debug=doLog)
  elif useTreeFast:
    matrix, qf = qMatrixUsingTreeFast(data, debug=doLog)
  else:
    matrix, qf = qMatrix(data)
  del data

  tStart = timeStep(tStart, "Compute Q-Matrix")
  
  if outFileName is not None:
    if outFileName != "-":
      with open(outFileName, "wt") as f:
        writeMatrix(matrix, columns, f)
    else:
      writeMatrix(matrix, columns, sys.stdout)

  if doLog and qf is not None:
    qf.statistics()

  if outFileNamePoints is not None or outFileNameImage is not None:
    tStart = time.time()
    projection = projectFeatures(matrix, coolDown=coolDown)    
    tStart = timeStep(tStart, "Projection")

    if outFileNamePoints is not None:
      if outFileNamePoints != "-":
        with open(outFileNamePoints, "wt") as f:
          writePoints(projection, columns, f)
      else:
        writePoints(projection, columns, sys.stdout)

    if outFileNameImage is not None:
      plt.title("Projection")
      plt.scatter(projection[:,0], projection[:,1])
      plt.xlim(-110,110)
      plt.ylim(-110,110)

      for i, feat in enumerate(columns):
        plt.text(projection[i, 0] + 0.1, projection[i, 1] + 0.1, feat)

      if outFileNameImage != "-":
        plt.savefig(outFileNameImage)
      else:
        plt.show()

  if outFileNameDendrogram is not None:
    from qFunction.dendrogram import plot_manual_dendrogram
    p = plot_manual_dendrogram(matrix, columns)
    if outFileNameDendrogram != "-":
      p.savefig(outFileNameDendrogram)
    else:
      p.show()
    
  if outFileNameUmap is not None:
    from qFunction.umap_projection import visualize_umap_embeddings
    umap_params = {}
    workflow_type = "single"
    umap_fusion_params = None # dict()
    plot_title = "UMAP embedding of features",
    if outFileNameUmap != "-":
      visualize_umap_embeddings(columns, umap_params, matrix, workflow_type, umap_fusion_params, plot_title, outFileNameUmap)
    else:
      visualize_umap_embeddings(columns, umap_params, matrix, workflow_type, umap_fusion_params, plot_title)
    
  if outFileNameModularityNotebook is not None:
    from qFunction.modularity_notebook_projection import visualize_modularity_embedding_notebook
    visualize_modularity_embedding_notebook(
        feature_names=columns,
        q_matrix=matrix,
        save_path=None if outFileNameModularityNotebook == "-" else outFileNameModularityNotebook
    )

  if outFileName2ndlayer is not None or outFileNameFunctions is not None:
    from qFunction.prediction import predict2ndLayer, predictFunctions, showFunctions
    print("* predict 2nd layer", file=sys.stderr)
    matrix = predict2ndLayer(qf)
    if outFileName2ndlayer is not None:
      if outFileName2ndlayer != "-":
        with open(outFileName2ndlayer, "wt") as f:
          writeMatrix2(matrix, columns, f)
      else:
        writeMatrix2(matrix, columns, sys.stdout)

    if outFileNameFunctions is not None:
      print("* predict functions", file=sys.stderr)
      functions = predictFunctions(qf, matrix, columns)
      if outFileNameFunctions != "-":
        with open(outFileNameFunctions, "wt") as f:
          showFunctions(functions, f)
      else:
        showFunctions(functions, sys.stdout)

print("* done", file=sys.stderr)
timeStep(tStartTotal, "Total")
