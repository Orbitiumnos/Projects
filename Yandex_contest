n = int(input())
i = 0
l = []
while i != n:
    eff_tm,end_tm,vote = input().split()
    d = dict(eff_tm = eff_tm, end_tm = end_tm, vote = vote)
    l.append(d)
    i = i + 1
#print(l)

def calc(eff_time, end_time, ch_eff_tm, ch_end_tm, res, vote, opt_vote):
    #print (end_time,ch_eff_tm,ch_end_tm,eff_time)
    if end_time >= ch_eff_tm:
        if ch_end_tm >= eff_time:
            res = res + 1
            opt_vote = opt_vote + int(vote)
    return res, opt_vote

def show(l,n):
    i = 0
    while i < n:
        print (l[i])
        i = i + 1 
        
def find_max(l,n):
    i = 0
    max_1 = 0
    while i < n:
        max_1 = max(l[i].get('res'), max_1)
        i = i + 1
        
    i = 0
    max_2 = 0
    while i < n:
        if l[i].get('res') == max_1:
            max_2 = max(max_2,l[i].get('opt_vote'))    
        i = i + 1
    print(max_2)
    return

i = 0
while i < n:
    i_2 = 0
    res = 0
    opt_vote = 0
    while i_2 < n:
        res, opt_vote = calc(l[i_2].get('eff_tm'), l[i_2].get('end_tm'), 
                             l[i].get('eff_tm'), l[i].get('end_tm'), res, l[i_2].get('vote'), opt_vote)
        i_2 = i_2 + 1
    l[i]['res'], l[i]['opt_vote'] = res, opt_vote
    i = i + 1

#show(l,n)    
find_max(l,n) 
    
    
'''
5
3 7 1
7 8 2
3 4 1
1 2 2
4 7 3
'''
