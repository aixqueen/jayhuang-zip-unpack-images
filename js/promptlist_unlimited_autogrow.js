import { app } from "/scripts/app.js";
const MAX=64;
const CLS="easy promptList (Unlimited)";
function findI(node,name){const a=node.inputs||[];for(let i=0;i<a.length;i++){if(a[i]&&a[i].name===name)return i;}return -1;}
function removeByName(node,name){const i=findI(node,name);if(i>=0)node.removeInput(i);}
function ensure(node,i){const n=`prompt_${i}`;if(findI(node,n)<0)node.addInput(n,"STRING");}
function linked(node,i){const j=findI(node,`prompt_${i}`);return j>=0 && node.inputs[j] && node.inputs[j].link!=null;}
function pruneTo(node,keep){for(let i=MAX;i>keep;i--){removeByName(node,`prompt_${i}`);} }
function refresh(node){let last=0;for(let i=1;i<=MAX;i++){if(linked(node,i))last=i;}const show=Math.min(Math.max(last+1,1),MAX);for(let i=1;i<=show;i++)ensure(node,i);pruneTo(node,show);node.setDirtyCanvas(true,true);}
app.registerExtension({name:"jayhuang.promptlist.unlimited.autoports",nodeCreated(node){if(node.comfyClass!==CLS)return;for(let i=MAX;i>=2;i--)removeByName(node,`prompt_${i}`);ensure(node,1);const orig=node.onConnectionsChange;node.onConnectionsChange=function(...args){if(orig)orig.apply(this,args);refresh(this);};refresh(node);}});
