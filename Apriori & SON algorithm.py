import pyspark as ps
sc=ps.SparkContext("local")
import sys

case_num=int(sys.argv[1])
ratings_path=sys.argv[2]
users_path=sys.argv[3]
support=int(sys.argv[4])

def Get_Candidates(data,threshold):
    local_candidate=dict()
    candidate_set=set()
    data_sets=list()
    items_set=dict()
    for i in data:
        data_sets.append(set(i))
    
    for i in data_sets:
        for j in i:
            if items_set.has_key(j):items_set[j]+=1
            else: items_set[j]=1
    
    for i in items_set.keys():
        if items_set[i]>=threshold:
            candidate_set.add(frozenset([i]))
            
    k=2
    local_candidate[k-1]=candidate_set
    
    while candidate_set != set([]):
        
        candidate_set = set([i.union(j) for i in candidate_set for j in candidate_set if len(i.union(j)) == k])
        
        intermediate_set = set()
        intermediateDict = dict()
        for i in candidate_set:
            for j in data_sets:
                if i.issubset(j):
                    if intermediateDict.has_key(i): intermediateDict[i]+=1
                    else: intermediateDict[i]=1
        
        for i in intermediateDict.keys():
            if intermediateDict[i]>=threshold:intermediate_set.add(i)

        if intermediate_set != set([]): local_candidate[k]=intermediate_set
        candidate_set = intermediate_set
        k = k + 1
    
    return local_candidate

    
###########################################################################################

def SON_apriori(ratings_path,users_path,case_num,support):

    ratings=sc.textFile(ratings_path)

    users=sc.textFile(users_path)
    
    users= users.map(lambda x:x.split("::"))\
                .map(lambda x:(x[0],x[1]))

    ratings=ratings.map(lambda x:x.split('::'))\
                   .map(lambda x:(x[0],x[1]))

    user_rating=ratings.join(users)
# user1,(rating1,gender)
# user1, rating1
# user1, (r1::r2::r3::r4)
# user1 rating profile
    if case_num == 1:
        baskets=user_rating.filter(lambda x: x[1][1]=='M')\
                            .map(lambda x: (x[0],x[1][0]))\
                            .reduceByKey(lambda x,y: str(x)+"::"+str(y))\
                            .map(lambda x: (x[1].split("::")))
    elif case_num == 2:
        baskets=user_rating.filter(lambda x: x[1][1]=='F')\
                              .map(lambda x: (x[1][0],x[0]))\
                              .reduceByKey(lambda x,y: str(x)+"::"+str(y))\
                              .map(lambda x: (x[1].split("::")))
    

    numPartitions=baskets.getNumPartitions()

    basketSets = baskets.map(set).persist()
    
    candidates_chunk=baskets.mapPartitions(lambda data:[i for j in Get_Candidates(data,support/numPartitions).values() for i in j],True)
    candidates_all=candidates_chunk.map(lambda x:(x,1))\
                                   .reduceByKey(lambda x,y: x)\
                                   .map(lambda x: x[0])
    
    candidates=sc.broadcast(candidates_all.collect())
    
    candidates_counts = basketSets.flatMap(lambda x: [(i,1) for i in candidates.value if i.issubset(x)])\
                                  .reduceByKey(lambda x,y: x+y)
    
    candidates_final=candidates_counts.filter(lambda x: x[1]>=support)


    finalItems = candidates_final.map(lambda x: x[0])\
                              .map(lambda x: (sorted([int(i) for i in x]),len(x)))\
                              .sortBy(lambda x: (x[1],x[0]))\
                              .map(lambda x: x[0])\
                              .map(lambda x: "("+", ".join([str(i) for i in x])+")"+", " )
    

    finalItems_ls=finalItems.collect()
    output=open('CHONG_LI_SON.'+'case'+str(case_num)+'_'+str(support)+'.txt','w')
    for i in finalItems_ls:
        output.write(str(i))


SON_apriori(ratings_path,users_path,case_num,support)


