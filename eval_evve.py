import sys, os, pdb, random

def usage():
  print >> sys.stderr, """eval_evve.py [-annotdir dir] [-v] result_file.dat

Evaluates the precision-recall for queries in the EVVE dataset (or a subset of events).

Result format: a text file with lines

q1 db11 db12 db13 ....
q2 db21 db22 db23 ....

where qi is the query video, and qij are the returned database videos,
ordered by decreasing relevance. Both are raw ids (without directory
name and suffix). 

"""
  sys.exit(1)
  
resfile = None
annotdir = "annots"
verbose = 0

args = sys.argv[1:]

while args:
  a = args.pop(0)
  if a in ['-h','--help']:      usage()
  elif a == '-annotdir':        annotdir = args.pop(0)
  elif a == '-v':               verbose += 1
  elif not resfile:             resfile = a
  else:
      sys.stderr.write("unknown arg %s\n"%a)
      usage()


if verbose > 0:
  print ("parsing input annotations from directory", annotdir)

events = {}
all_db = set()     # all videos in the database
for fname in sorted(os.listdir(annotdir)):
  if not fname.endswith('.dat'): continue
  evname = fname.split('.')[0]
  queries = set()
  pos = set()
  null = set()
  
  for l in open(annotdir + '/' + fname, "r"):
    vidname, gt, split = l.split()
    gt = int(gt)
    if split == "query":
      queries.add(vidname)
    else:
      all_db.add(vidname)
      if gt > 0:           
        pos.add(vidname) 
      elif gt == 0:        
        null.add(vidname)
      
  events[evname] = (queries, pos, null)



def score_ap_from_ranks_1 (ranks, nres):
    """ Compute the average precision of one search.
    ranks = ordered list of ranks of true positives (best rank = 0)
    nres  = total number of positives in dataset  
    """
    if nres==0 or ranks==[]:
      return 0.0
    
    ap=0.0
    
    # accumulate trapezoids in PR-plot. All have an x-size of:
    recall_step=1.0/nres
    
    for ntp,rank in enumerate(ranks):
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far

        # y-size on left side of trapezoid:
        if rank==0: precision_0=1.0
        else:       precision_0=ntp/float(rank)      
        # y-size on right side of trapezoid:
        precision_1=(ntp+1)/float(rank+1)
        ap+=(precision_1+precision_0)*recall_step/2.0
    
    return ap


#parse result file
query_to_event = {qname: evname 
        for evname, (queries, _, _) in events.items() 
        for qname in queries}



n_ext = 0
results = {e: [] for e in events}

if verbose > 0:
  print ("parsing result file", resfile)

for l in open(resfile, "r"):
  fields = l.split()
  qname = fields[0]                # query name
  res = fields[1:]                 # results
  evname = query_to_event[qname]   # corresponding event
  del query_to_event[qname]        # remove to avoid duplicate queries
  _, pos, null = events[evname]    

  pos_ranks = []                   # ranks of TPs (0-based)
  rank_shift = 0                   # adjust rank to ignore nulls

  for rank, dbname in enumerate(res):
    if dbname in pos:
      pos_ranks.append(rank - rank_shift)
    elif dbname in null:
      rank_shift += 1
      
    if dbname not in all_db:
      n_ext += 1

  ap = score_ap_from_ranks_1(pos_ranks, len(pos))

  # p@10
  # ap = len([1 for i in pos_ranks if i < 10]) / 10.0

  if verbose > 1:
    print ("query %s (%s): AP=%.3f, ranks of positives (out of %d) = %s" % (
      qname, evname, ap, len(pos), pos_ranks))

  results[evname].append(ap)

if n_ext > 0:
  print ("warn: some results do not belong to the EVVE dataset (assuming those are distractors)")


# display result
sum_ap = 0.0
n_ap = 0
sum_mAP = 0.0
n_mAP = 0

for evname in sorted(events):
  print ("%-35s" % evname),
  queries, _, _ = events[evname]
  nq = len(queries)           # expected nb of queries
  res = results[evname]       # APs for this event
  nr = len(res)
  if nr < nq:
    print ("missing %d/%d queries" % (nq - nr, nq))
    n_ap = None               # refuse to compute overall mAP
    continue
  mAP = sum(res) / nq
  print ("mAP = %.4f" % mAP)
  if n_ap != None:
    n_ap += nq
    sum_ap += sum(res)
    sum_mAP += mAP
    n_mAP += 1


if n_ap != None:
  print ("=" * 45)
  print ("overall mAP = %.4f" % (sum_ap / n_ap))
  print ("avg-mAP = %.4f" % (sum_mAP / n_mAP))

