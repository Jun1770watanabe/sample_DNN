#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                        optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from tqdm import tqdm

jvocab = {}
<<<<<<< HEAD
jlines = open('jp.txt').read().split('\n')
print(jlines)
=======
jlines = open('text/JEC_jap2.txt').read().split('\n')
>>>>>>> 3df80b5bdccc39ac2be656fe2abbb7f11b7298da
for i in range(len(jlines)):
    lt = jlines[i].split()
    print(lt)
    for w in lt:
        if w not in jvocab:
            jvocab[w] = len(jvocab)

jvocab['<eos>'] = len(jvocab)
jv = len(jvocab)

evocab = {}
elines = open('text/JEC_eng2.txt',encoding="utf-8").read().split('\n')
for i in range(len(elines)):
    lt = elines[i].split()
    for w in lt:
        if w not in evocab:
            evocab[w] = len(evocab)

evocab['<eos>'] = len(evocab)
ev = len(evocab)

class MyMT(chainer.Chain):
    def __init__(self, jv, ev, k):
        super(MyMT, self).__init__(
            embedx = L.EmbedID(jv, k),
            embedy = L.EmbedID(ev, k),  
            H = L.LSTM(k, k),
            W = L.Linear(k, ev),
        )
    def __call__(self, jline, eline):
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)
        x_k = self.embedx(Variable(np.array([jvocab['<eos>']], dtype=np.int32)))
        tx = Variable(np.array([evocab[eline[0]]], dtype=np.int32))
        h = self.H(x_k)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        for i in range(len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))                        
            next_wid = evocab['<eos>']  if (i == len(eline) - 1) else evocab[eline[i+1]]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss += loss 
        return accum_loss

demb = 100
print(jv)
print(ev)
model = MyMT(jv, ev, demb)
print(model)
exit()
optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(100):
    for i in tqdm(range(len(jlines)-1)):
        jln = jlines[i].split()
        jlnr = jln[::-1]
        eln = elines[i].split()
        model.H.reset_state()
        model.zerograds()        
        loss = model(jlnr, eln)
        loss.backward()
        loss.unchain_backward()  # truncate
        optimizer.update()
<<<<<<< HEAD
        print(i)
        print(" finished")
=======
        # print("{} finished".format(i))
>>>>>>> 3df80b5bdccc39ac2be656fe2abbb7f11b7298da
    outfile = "mt-" + str(epoch) + ".model"

    dir_pass = "{out_dir}/{outfile}".format(out_dir="model", outfile=outfile)

    serializers.save_npz(dir_pass, model)

