import random

objSize_loc = {'sky-other':{
    'size':0,
    'location':[0]
},
'grass':{
    'size':0,
    'location':[0]
},
'3':{
    'size':4,
    'location':[12]
},
'4':{
    'size':2,
    'location':[12,14]
},
'5':{
    'size':0,
    'location':[11,13,15]
}}


_object = ['elephant','sheep','zebra']
behaviour = ['walking','eating','running','drinking','standing']
relationship1 = ['left of', 'right of', 'above', 'below']

text = []
text2 = []

text3 = ['']



for x in range(10):
    text3.append(
        random.choice(_object)+"  " +
        random.choice(behaviour) + " " +
        random.choice(relationship1) +" " +
        random.choice(_subject)+ " and " +
        random.choice(_object) + "  " +
        random.choice(behaviour) + " " +
        random.choice(relationship1) + " "+
        random.choice(_subject)
        )

print(text3)
