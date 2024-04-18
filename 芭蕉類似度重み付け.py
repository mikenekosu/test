# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:07:42 2023

@author: sugitasyuoutarou
"""

import numpy as np
import numpy.linalg as npl
from matplotlib import pyplot as pyp
import copy
import math



#ファイルの読み込み
#textsに改行ごとに区切られて格納される
f = open(r"Basho.txt", 'r',encoding = "UTF-8")
s=f.read() 
text=s
texts=text.split("\n")
texts=list(filter(None,texts))

#形態素解析
from janome.tokenizer import Tokenizer
t= Tokenizer()
def extract_words(text):
    tokens = t.tokenize(text)
    return [token.base_form for token in tokens 
            #欲しい品詞を入力
            if token.part_of_speech.split(',')[0] in['名詞', '動詞',"形容詞","助詞","助動詞","副詞"]]
#text_listに形態素解析した結果を格納
text_list=[extract_words(text) for text in texts]   

#word2vecの導入
from gensim.models import word2vec
# size: 圧縮次元数
# min_count: 出現頻度の低いものをカットする
# window: 前後の単語を拾う際の窓の広さを決める
# iter: 機械学習の繰り返し回数(デフォルト:5)十分学習できていないときにこの値を調整する
# model.wv.most_similarの結果が1に近いものばかりで、model.dict['wv']のベクトル値が小さい値ばかりの 
# ときは、学習回数が少ないと考えられます。
# その場合、iterの値を大きくして、再度学習を行います
model = word2vec.Word2Vec(text_list, size=100,min_count=1,window=5,iter=300)

#idf
word_dic={}
#重複している単語の消去
count_list=copy.deepcopy(text_list)
for h in range(len(count_list)):
     for i in count_list[h]:
         count=count_list[h].count(i)
         if count == 1:
             pass
         else:
             for j in range(count-1):
               num=count_list[h].index(i)
               count_list[h][num]=" "
#使用されている文章数               
for h in range(len(count_list)):
    for i in count_list[h]:
        if i in word_dic.keys():
            word_dic[i]=word_dic[i]+1
        else:
            word_dic[i]=1
       
        
#text_listの何個目の要素について比較するか
x=input("何句目?:")
#類似度の高い上位何件を表示するか
y=input("上位何件?:")
x=int(x)
y=int(y)
#textsの数だけ100次元の空リストの生成
apv=[]
for h in range(len(text_list)):
    for i in range(100):
        hy=[]        
        hy.append(0)
    apv.append(hy)
#単語の意味を足し算
#重み付け
tfidf=[]
for h in range(len(text_list)):
    for i in text_list[h]:
        v=model.wv[i]
        nv=np.array(v)
        tf=text_list[h].count(i)/len(text_list[h])
        idf=math.log2(len(text_list)/word_dic[i])
        tfidf.append(tf*idf)
        nv=tf*idf*nv
        apv[h]+=nv

rui_list=[]
jyun_list=[] 
c=0
for i in range(len(text_list)):
    try:
        cos=(np.dot(apv[x],apv[i])/(npl.norm(apv[x])*npl.norm(apv[i])))
        #cos類似度の結果をrui_listに格納
        rui_list.append(cos)
    except ValueError:
        rui_list.append(0)
rui_list[x]=0
#rui_listを大きい順に並び替える
maxrui=sorted(rui_list,reverse=True)
#maxruiとtextsを対応づけ
for i in maxrui:
    jyun_list.append(rui_list.index(i))
#結果の表示
print(texts[x])
print()
for i in jyun_list[:y]:
    print(texts[i])
    print("類似度:"+str(maxrui[c]))
    c+=1
#グラフの作成
#縦軸は類似度
#横軸は
pyp.plot(range(len(maxrui)),maxrui)
pyp.show
