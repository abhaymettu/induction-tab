import json
c = {"V":96,"T":64,"d":64,"H":4,"L":2,"unique":2,"B":16,"mlp":4,"ln":True,"lr":0.003,"uniform":True,"tile":True,"pRep":1,"rMin":8,"rMax":32,"probeB":16,"probeT":64}
arms=[c]
def add(**k): arms.append({**c,**k})
for lr in (0.001,0.002,0.005,0.01): add(lr=lr)
for d in (32,48,96,128): add(d=d)
for T in (32,48,96): add(T=T,probeT=T,rMin=T//8,rMax=T//2)
for B in (4,8,32,64): add(B=B)
for m in (0,1,2): add(mlp=m)
add(ln=False); add(mlp=0,ln=False); add(mlp=0,lr=0.005)
for H in (1,2,8): add(H=H)
add(rMin=4,rMax=32); add(rMin=16,rMax=32); add(rMin=8,rMax=16); add(tile=False); add(warmup=0); add(warmup=25); add(V=128); add(V=256)
add(pRep=0.75); add(uniform=False)
print(json.dumps(arms))
