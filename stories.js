let _object = ['elephant','sheep','zebra','grass']
let _subject = ['bear','sheep','zebra','grass']
let relationship = ['left of', 'right of', 'above', 'below']
let behaviour = ['walking','eating','running','drinking','standing']

let text = []

_object.forEach((obj)=>{
        _subject.forEach((sub)=>{
    behaviour.forEach((beh)=>{
    relationship.forEach((rel)=>{
    text.push(obj+ " is " + beh + " " + rel + " " +sub)
})
})
})
})

console.log(text)