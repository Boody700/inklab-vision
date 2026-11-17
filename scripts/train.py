"""Official different-writer holdout; validation selection; portable classifiers."""
from pathlib import Path
import os, json, hashlib, platform
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
ROOT=Path(__file__).resolve().parents[1]
NAMES={'linear':'Linear softmax','neighbors':'3-nearest neighbors','neural':'Small neural network'}

def load():
    train=np.loadtxt(ROOT/'data/optdigits.tra',delimiter=',',dtype=int)
    test=np.loadtxt(ROOT/'data/optdigits.tes',delimiter=',',dtype=int)
    if train.shape!=(3823,65) or test.shape!=(1797,65):raise ValueError('Unexpected official split')
    return train[:,:64],train[:,64],test[:,:64],test[:,64]

def knn_proba(x,reference,labels,k=3):
    """Stable distance ties resolve by the original training-row index."""
    result=[]
    for row in x:
        distance=np.sum((reference.astype(float)-row)**2,axis=1)
        ix=np.argsort(distance,kind='stable')[:k]
        result.append(np.bincount(labels[ix],minlength=10)/k)
    return np.array(result)

def fit(name,x,y):
    if name=='neighbors':return (x.copy(),y.copy())
    model=LogisticRegression(C=10,max_iter=2000,tol=1e-6,random_state=42) if name=='linear' else MLPClassifier(hidden_layer_sizes=(64,),activation='relu',alpha=.001,batch_size=128,learning_rate_init=.001,max_iter=350,tol=1e-5,n_iter_no_change=30,early_stopping=False,random_state=42)
    return model.fit(x/16,y)

def predict(name,model,x):
    return knn_proba(x,*model) if name=='neighbors' else model.predict_proba(x/16)

def export(name,model):
    if name=='neighbors':return {'kind':name,'pixels':model[0].tolist(),'labels':model[1].tolist(),'k':3}
    if name=='linear':return {'kind':name,'weights':model.coef_.T.tolist(),'bias':model.intercept_.tolist()}
    return {'kind':name,'weights':[w.tolist() for w in model.coefs_],'bias':[b.tolist() for b in model.intercepts_]}

def wilson(correct,n):
    z=1.95996398454;p=correct/n;denom=1+z*z/n;center=(p+z*z/(2*n))/denom;radius=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/denom
    return [float(center-radius),float(center+radius)]

def scores(y,p):
    label=p.argmax(1);acc=float(accuracy_score(y,label));correct=int((label==y).sum())
    return {'accuracy':acc,'macroF1':float(f1_score(y,label,average='macro')),'logLoss':float(log_loss(y,p,labels=list(range(10)))),'brier':float(np.mean(np.sum((p-np.eye(10)[y])**2,axis=1))),'correct':correct,'total':len(y),'errors':int(len(y)-correct),'wilson95':wilson(correct,len(y)),'confusion':confusion_matrix(y,label,labels=list(range(10))).tolist()}

def coverage_curve(y,p):
    out=[]
    for threshold in [0,.5,.6,.7,.8,.9,.95,.99]:
        accepted=p.max(1)>=threshold;n=int(accepted.sum())
        out.append({'threshold':threshold,'coverage':float(accepted.mean()),'accepted':n,'accuracy':float(np.mean(p.argmax(1)[accepted]==y[accepted])) if n else None})
    return out

def perturb(pixels,dx=0,dy=0,noise=0,seed=42,contrast=1):
    a=np.array(pixels,dtype=float).reshape(8,8);out=np.zeros((8,8));rng=seed&0xffffffff
    for r in range(8):
        for c in range(8):
            value=a[r-dy,c-dx] if 0<=r-dy<8 and 0<=c-dx<8 else 0
            rng=(1664525*rng+1013904223)&0xffffffff
            n=(rng/4294967296*2-1)*noise
            out[r,c]=np.clip(np.floor(value*contrast+n+.5),0,16)
    return out.ravel()

def main():
    x,y,xt,yt=load();indices=np.arange(len(y))
    tr,va=train_test_split(indices,test_size=.2,stratify=y,random_state=42)
    results={};val_prob={};models={};test_prob={}
    for name in NAMES:
        print('Validation',name,flush=True);m=fit(name,x[tr],y[tr]);p=predict(name,m,x[va]);val_prob[name]=p
        results[name]={'label':NAMES[name],'validation':scores(y[va],p)}
    chosen=max(NAMES,key=lambda k:(results[k]['validation']['accuracy'],results[k]['validation']['macroF1']))
    curve=coverage_curve(y[va],val_prob[chosen]);eligible=[v for v in curve if v['accuracy'] is not None and v['accuracy']>=.99 and v['coverage']>=.5]
    threshold=min(eligible,key=lambda v:v['threshold'])['threshold'] if eligible else .8
    # Model identity and review threshold are now frozen. The official test is from different writers.
    for name in NAMES:
        print('Official holdout',name,flush=True);m=fit(name,x,y);p=predict(name,m,xt)
        models[name]=export(name,m);test_prob[name]=p;results[name]['test']=scores(yt,p)
    chosen_model=models[chosen];chosen_prob=test_prob[chosen];pred=chosen_prob.argmax(1)
    # Reconstruct fitted objects only as necessary for robustness scoring, using already fitted full-training models.
    def portable_predict(name,model,rows):
        if name=='neighbors':return knn_proba(rows,np.array(model['pixels']),np.array(model['labels']))
        a=rows/16
        if name=='linear':a=a@np.array(model['weights'])+np.array(model['bias'])
        else:
            for i,(w,b) in enumerate(zip(model['weights'],model['bias'])):
                a=a@np.array(w)+np.array(b)
                if i<len(model['weights'])-1:a=np.maximum(a,0)
        a-=a.max(1)[:,None];e=np.exp(a);return e/e.sum(1)[:,None]
    robustness=[]
    for label,args in [('Unchanged',{}),('Shift right 1px',{'dx':1}),('Shift left 1px',{'dx':-1}),('Uniform noise ±2',{'noise':2}),('Uniform noise ±4',{'noise':4}),('Half contrast',{'contrast':.5})]:
        changed=np.array([perturb(a,seed=42+i,**args) for i,a in enumerate(xt)])
        pp=portable_predict(chosen,chosen_model,changed);robustness.append({'label':label,**scores(yt,pp)})
    overlap=set(map(tuple,x))&set(map(tuple,xt))
    report={'selected':chosen,'names':NAMES,'models':results,'split':{'officialTrain':len(y),'fit':len(tr),'validation':len(va),'officialTest':len(yt),'trainWriters':30,'testWriters':13,'validationPolicy':'Stratified 80/20 of official training rows, random_state=42; writer IDs unavailable inside this file. Final official test writers differ from training writers.','crossSplitIdenticalVectors':len(overlap)},'reviewThreshold':threshold,'reviewPolicy':'Lowest validation threshold achieving >=99% accepted accuracy with >=50% coverage if one exists; otherwise preset 0.8. Uncalibrated model scores; no accuracy guarantee on new inputs.','validationCoverage':curve,'testCoverage':coverage_curve(yt,chosen_prob),'robustness':robustness,'perClass':classification_report(yt,pred,labels=list(range(10)),output_dict=True),'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/optdigits.tra',ROOT/'data/optdigits.tes']},'parameters':{'linear':'C=10, max_iter=2000, tol=1e-6','neighbors':'k=3, uniform Euclidean votes, stable row-index tie breaks','neural':'64-ReLU-10 softmax, Adam, alpha=.001, batch=128, lr=.001, max_iter=350, tol=1e-5, n_iter_no_change=30, no early-stopping split, random_state=42'}}
    test_rows=[{'id':i,'label':int(yt[i]),'pixels':xt[i].tolist(),'predictions':{k:int(test_prob[k][i].argmax()) for k in NAMES},'scores':{k:float(test_prob[k][i].max()) for k in NAMES}} for i in range(len(yt))]
    out=ROOT/'dist/data';out.mkdir(parents=True,exist_ok=True);reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    for name,obj in [('models',models),('samples',test_rows),('evaluation',report)]:
        (out/(name+'.json')).write_text(json.dumps(obj,separators=(',',':'),allow_nan=False),encoding='utf-8')
    (reports/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    # Full-probability parity fixtures cover every official test image, not only selected screenshots.
    (reports/'parity-reference.json').write_text(json.dumps({k:p.tolist() for k,p in test_prob.items()},separators=(',',':')),encoding='utf-8')
    print(json.dumps({'selected':chosen,'threshold':threshold,'test':results[chosen]['test']['accuracy'],'overlap':len(overlap),'robustness':[(r['label'],r['accuracy']) for r in robustness]}),flush=True)
if __name__=='__main__':main()
