// =====================================================================
//  Anatomie Explorer — Engine v2
//  - lädt echte GLB-Modelle aus models/ (mit Platzhalter-Fallback)
//  - Aufschneiden (Schnittebene), Teile auseinanderziehen (Explosion)
//  - schlagendes Herz, Hand- & Maussteuerung, Notizen, Infos
// =====================================================================
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { HandLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12";

// ---------------------------------------------------------------------
//  One-Euro-Filter (Hand-Glättung)
// ---------------------------------------------------------------------
class LowPass { constructor(){this.s=null;} filter(x,a){this.s=(this.s===null)?x:a*x+(1-a)*this.s;return this.s;} }
class OneEuro {
  constructor(minCutoff=1.2,beta=0.02,dCutoff=1.0){this.minCutoff=minCutoff;this.beta=beta;this.dCutoff=dCutoff;this.xPrev=null;this.tPrev=null;this.xF=new LowPass();this.dxF=new LowPass();}
  alpha(c,dt){const tau=1/(2*Math.PI*c);return 1/(1+tau/dt);}
  filter(x,t){if(this.tPrev===null){this.tPrev=t;this.xPrev=x;return x;}let dt=(t-this.tPrev)/1000;if(dt<=0)dt=1/60;const dx=(x-this.xPrev)/dt;const edx=this.dxF.filter(dx,this.alpha(this.dCutoff,dt));const cutoff=this.minCutoff+this.beta*Math.abs(edx);const fx=this.xF.filter(x,this.alpha(cutoff,dt));this.xPrev=x;this.tPrev=t;return fx;}
}

// ---------------------------------------------------------------------
//  Hilfen
// ---------------------------------------------------------------------
function tagPart(obj,id,label,info){obj.traverse(o=>{o.userData.part={partId:id,label,info};});obj.userData.part={partId:id,label,info};return obj;}
function wrinkle(geo,amp,freq){const p=geo.attributes.position;for(let i=0;i<p.count;i++){const x=p.getX(i),y=p.getY(i),z=p.getZ(i);const n=Math.sin(x*freq)*Math.cos(y*freq)*Math.sin(z*freq);const f=1+amp*n;p.setXYZ(i,x*f,y*f,z*f);}p.needsUpdate=true;geo.computeVertexNormals();}

// ---------------------------------------------------------------------
//  Platzhalter-Modelle (aus Code) — Fallback, falls kein GLB vorhanden
// ---------------------------------------------------------------------
function buildDNA(){
  const g=new THREE.Group();const N=26,step=0.34,radius=1.15,turn=0.42;
  const backbone=new THREE.Group(),rungs=new THREE.Group();
  const sph=new THREE.SphereGeometry(0.14,16,16);
  const mA=new THREE.MeshStandardMaterial({color:0x2dd4bf,metalness:0.2,roughness:0.4});
  const mB=new THREE.MeshStandardMaterial({color:0x7c9cff,metalness:0.2,roughness:0.4});
  const cols=[0xf5a524,0xef4f6b,0x9d7bff,0x49c96d];
  for(let i=0;i<N;i++){const a=i*turn,y=i*step-(N*step)/2;
    const ax=Math.cos(a)*radius,az=Math.sin(a)*radius,bx=Math.cos(a+Math.PI)*radius,bz=Math.sin(a+Math.PI)*radius;
    const s1=new THREE.Mesh(sph,mA);s1.position.set(ax,y,az);backbone.add(s1);
    const s2=new THREE.Mesh(sph,mB);s2.position.set(bx,y,bz);backbone.add(s2);
    const dx=bx-ax,dz=bz-az,len=Math.sqrt(dx*dx+dz*dz);
    const r=new THREE.Mesh(new THREE.CylinderGeometry(0.05,0.05,len,8),new THREE.MeshStandardMaterial({color:cols[i%cols.length],roughness:0.5}));
    r.position.set((ax+bx)/2,y,(az+bz)/2);r.rotation.z=Math.PI/2;r.rotation.y=-Math.atan2(dz,dx);rungs.add(r);}
  tagPart(backbone,"backbone","Zucker-Phosphat-Rückgrat","Die beiden äußeren Stränge der Doppelhelix aus Zucker und Phosphat. Sie geben der DNA Stabilität und Form.");
  tagPart(rungs,"bases","Basenpaare","Die Sprossen der Leiter: je ein Paar Basen (A-T oder G-C). Ihre Reihenfolge speichert die gesamte Erbinformation.");
  g.add(backbone,rungs);return{group:g,zoom:7,spin:0.004};
}
function buildBrain(){
  const g=new THREE.Group();const mk=c=>new THREE.MeshStandardMaterial({color:c,roughness:0.8});
  function hemi(s){const geo=new THREE.IcosahedronGeometry(1.15,5);wrinkle(geo,0.10,9);const m=new THREE.Mesh(geo,mk(0xe8a0a8));m.scale.set(0.9,0.85,1.1);m.position.set(s*0.62,0.25,0);return m;}
  const L=hemi(-1),R=hemi(1);
  tagPart(L,"hemL","Linke Großhirnhälfte","Steuert vor allem die rechte Körperseite. Bei den meisten Menschen sitzen hier Sprache und logisches Denken.");
  tagPart(R,"hemR","Rechte Großhirnhälfte","Steuert vor allem die linke Körperseite. Stark beteiligt an räumlichem Denken, Kreativität und Musik.");
  const cg=new THREE.IcosahedronGeometry(0.6,4);wrinkle(cg,0.06,16);const cere=new THREE.Mesh(cg,mk(0xd98a93));cere.scale.set(1.25,0.7,0.9);cere.position.set(0,-0.85,-0.85);
  tagPart(cere,"cere","Kleinhirn","Koordiniert Bewegungen, Gleichgewicht und Feinmotorik. Enthält über die Hälfte aller Nervenzellen des Gehirns.");
  const stem=new THREE.Mesh(new THREE.CylinderGeometry(0.22,0.32,1.0,20),mk(0xc77a82));stem.position.set(0,-1.25,-0.35);stem.rotation.x=0.35;
  tagPart(stem,"stem","Hirnstamm","Verbindet Gehirn und Rückenmark. Steuert lebenswichtige Funktionen wie Atmung, Herzschlag und Schlaf.");
  g.add(L,R,cere,stem);return{group:g,zoom:6,spin:0.003};
}
function buildHeart(){
  // Anatomische Ausrichtung: Vorderansicht. +Z = vorne, +Y = oben, +X = (Patienten-)links.
  const g=new THREE.Group();
  const C={myo:0x8f2733,myoR:0x9c3540,atrium:0xb24b50,aorta:0xd23b3b,artery:0xe8403c,
           pulm:0x4f63c4,vein:0x4a6fb0,pulmVein:0xc24a48,coroVein:0x7d5fae,valve:0xe9e2cf,fat:0xe8c98f};
  const mk=(c,rough=0.62)=>new THREE.MeshStandardMaterial({color:c,roughness:rough,metalness:0.04});
  const ell=(rx,ry,rz,c)=>{const m=new THREE.Mesh(new THREE.SphereGeometry(1,40,28),mk(c));m.scale.set(rx,ry,rz);return m;};
  const V=a=>new THREE.Vector3(a[0],a[1],a[2]);
  const tube=(pts,r,c,rough=0.5)=>{const curve=new THREE.CatmullRomCurve3(pts.map(V));return new THREE.Mesh(new THREE.TubeGeometry(curve,Math.max(24,pts.length*14),r,12,false),mk(c,rough));};

  // ---------- KAMMERN (Ventrikel) ----------
  const lv=ell(0.95,1.35,0.95,C.myo); lv.position.set(0.35,-0.35,-0.05); lv.rotation.z=0.16;        // linke Kammer -> Apex
  const rv=ell(0.92,1.15,0.78,C.myoR); rv.position.set(-0.5,-0.1,0.42); rv.rotation.z=-0.12;          // rechte Kammer (vorne)
  const ventricles=new THREE.Group(); ventricles.add(lv,rv);
  tagPart(ventricles,"ventricles","Herzkammern (Ventrikel)","Die beiden unteren, muskelstärksten Räume. Die linke Kammer (dickere Wand) pumpt sauerstoffreiches Blut über die Aorta in den ganzen Körper, die rechte Kammer pumpt sauerstoffarmes Blut in die Lunge. Ihre Spitze bildet den Apex des Herzens.");

  // ---------- VORHÖFE (Atrien) ----------
  const la=ell(0.62,0.6,0.62,C.atrium); la.position.set(0.55,0.95,-0.5);
  const laApp=ell(0.3,0.26,0.36,C.atrium); laApp.position.set(0.7,0.75,0.25);  // linkes Herzohr (Auricula)
  const leftAtrium=new THREE.Group(); leftAtrium.add(la,laApp);
  tagPart(leftAtrium,"la","Linker Vorhof","Nimmt das sauerstoffreiche Blut aus den vier Lungenvenen auf und leitet es durch die Mitralklappe in die linke Kammer. Das ausgestülpte linke Herzohr (Auricula) liegt vorne auf.");
  const ra=ell(0.66,0.66,0.62,C.atrium); ra.position.set(-0.8,0.7,0.05);
  const raApp=ell(0.3,0.28,0.34,C.atrium); raApp.position.set(-0.55,0.85,0.45);
  const rightAtrium=new THREE.Group(); rightAtrium.add(ra,raApp);
  tagPart(rightAtrium,"ra","Rechter Vorhof","Sammelt das sauerstoffarme Blut aus dem Körper (über obere und untere Hohlvene) und gibt es durch die Trikuspidalklappe an die rechte Kammer weiter. Enthält den Sinusknoten — den natürlichen Taktgeber des Herzens.");

  // ---------- AORTA + Bogen + Abgänge ----------
  const aorta=tube([[0.05,0.55,-0.05],[0.05,1.3,-0.1],[0.05,1.95,-0.15],[0.2,2.35,-0.25],[0.55,2.4,-0.55],[0.6,2.0,-0.95],[0.55,1.0,-1.05],[0.55,0.2,-1.05]],0.24,C.aorta);
  const bc=tube([[0.18,2.32,-0.28],[0.0,2.85,-0.2],[-0.15,3.2,-0.15]],0.09,C.aorta);   // Truncus brachiocephalicus
  const lcc=tube([[0.35,2.4,-0.4],[0.4,2.95,-0.4],[0.42,3.3,-0.4]],0.075,C.aorta);       // linke Halsschlagader
  const lsc=tube([[0.52,2.36,-0.5],[0.7,2.85,-0.55],[0.8,3.15,-0.6]],0.075,C.aorta);      // linke Schlüsselbeinarterie
  const aortaGrp=new THREE.Group(); aortaGrp.add(aorta,bc,lcc,lsc);
  tagPart(aortaGrp,"aorta","Aorta (Hauptschlagader)","Die größte Arterie des Körpers. Sie steigt aus der linken Kammer auf (Aorta ascendens), bildet den Aortenbogen und führt sauerstoffreiches Blut in den Kreislauf. Vom Bogen zweigen drei Gefäße ab: Truncus brachiocephalicus, linke Halsschlagader und linke Schlüsselbeinarterie.");

  // ---------- LUNGENARTERIENSTAMM (Truncus pulmonalis) ----------
  const pt=tube([[-0.45,0.55,0.45],[-0.3,1.2,0.25],[-0.1,1.7,-0.05],[0.0,1.9,-0.25]],0.22,C.pulm);
  const rpa=tube([[0.0,1.9,-0.25],[-0.5,1.85,-0.4],[-1.0,1.7,-0.5]],0.12,C.pulm);  // rechte Lungenarterie
  const lpa=tube([[0.0,1.9,-0.25],[0.5,1.9,-0.45],[0.95,1.7,-0.6]],0.12,C.pulm);   // linke Lungenarterie
  const pulmGrp=new THREE.Group(); pulmGrp.add(pt,rpa,lpa);
  tagPart(pulmGrp,"pulmonary","Lungenarterienstamm","Führt das sauerstoffarme Blut aus der rechten Kammer zur Lunge. Er kreuzt vor der Aorta und teilt sich in die rechte und linke Lungenarterie. Eine der wenigen Arterien, die sauerstoffarmes Blut transportieren.");

  // ---------- HOHLVENEN ----------
  const svc=tube([[-0.75,2.1,-0.05],[-0.78,1.5,0.0],[-0.8,0.95,0.05]],0.16,C.vein);
  const ivc=tube([[-0.7,-1.0,-0.2],[-0.72,-0.4,-0.1],[-0.7,0.25,0.0]],0.17,C.vein);
  const cavae=new THREE.Group(); cavae.add(svc,ivc);
  tagPart(cavae,"cavae","Obere & untere Hohlvene","Die beiden größten Venen. Die obere Hohlvene (Vena cava superior) bringt das Blut aus Kopf und Armen, die untere (Vena cava inferior) aus Bauch und Beinen — beide münden in den rechten Vorhof.");

  // ---------- LUNGENVENEN (4) ----------
  const pv=new THREE.Group();
  [[0.95,1.25,-0.85],[1.05,0.95,-0.95],[0.95,0.65,-0.95],[1.0,1.05,-0.6]].forEach(p=>{pv.add(tube([[p[0],p[1],p[2]],[0.6,p[1]*0.85+0.1,-0.6],[0.5,0.95,-0.5]],0.085,C.pulmVein));});
  tagPart(pv,"pulmveins","Lungenvenen","Vier Venen, die das in der Lunge mit Sauerstoff angereicherte Blut zurück zum linken Vorhof bringen. Sie sind die einzigen Venen, die sauerstoffreiches Blut führen.");

  // ---------- HERZKRANZARTERIEN ----------
  // Linke Koronararterie: Hauptstamm -> LAD (vorderer Sulcus) + Diagonaläste
  const lad=tube([[0.1,0.5,0.3],[0.0,0.1,0.55],[-0.05,-0.5,0.55],[-0.05,-1.15,0.45],[0.0,-1.7,0.3],[0.1,-2.05,0.15]],0.07,C.artery,0.4);
  const diag1=tube([[-0.02,-0.2,0.55],[0.35,-0.5,0.45],[0.6,-0.85,0.25]],0.045,C.artery,0.4);
  const diag2=tube([[-0.04,-0.7,0.52],[0.3,-1.0,0.4],[0.5,-1.3,0.2]],0.045,C.artery,0.4);
  const lmca=tube([[0.05,0.55,0.05],[0.1,0.5,0.3]],0.07,C.artery,0.4);
  const lca=new THREE.Group(); lca.add(lmca,lad,diag1,diag2);
  tagPart(lca,"lad","Linke Koronararterie (LAD)","Der Hauptstamm (LMCA) entspringt der Aorta und teilt sich. Der vordere absteigende Ast (LAD / Ramus interventricularis anterior) zieht in der vorderen Furche zur Herzspitze und versorgt die Vorderwand und den größten Teil der Scheidewand. Seine Verstopfung verursacht den klassischen Vorderwandinfarkt.");
  // Circumflex (linke Kranzfurche -> hinten) + Marginalast
  const cx=tube([[0.1,0.5,0.3],[0.6,0.55,0.1],[0.95,0.35,-0.25],[0.8,0.15,-0.65],[0.4,0.05,-0.85]],0.06,C.artery,0.4);
  const om=tube([[0.9,0.4,-0.15],[1.0,-0.1,-0.1],[0.85,-0.6,-0.05]],0.045,C.artery,0.4);
  const cxGrp=new THREE.Group(); cxGrp.add(cx,om);
  tagPart(cxGrp,"circumflex","Ramus circumflexus","Der zweite große Ast der linken Koronararterie. Er verläuft in der linken Kranzfurche um die Seitenwand herum nach hinten und versorgt mit seinen Randästen (Rami marginales / obtuse marginal) die Seiten- und Hinterwand der linken Kammer.");
  // RCA (rechte Kranzfurche) + SA-Knoten-Ast + Marginalast + PDA (hinten)
  const rca=tube([[-0.15,0.5,0.15],[-0.7,0.5,0.3],[-1.0,0.25,-0.1],[-0.85,0.0,-0.55],[-0.35,-0.05,-0.8]],0.07,C.artery,0.4);
  const sa=tube([[-0.55,0.55,0.25],[-0.7,0.9,0.1],[-0.75,1.15,0.0]],0.04,C.artery,0.4);
  const rmarg=tube([[-0.95,0.15,0.05],[-0.85,-0.5,0.2],[-0.5,-1.1,0.25]],0.05,C.artery,0.4);
  const pda=tube([[-0.35,-0.05,-0.8],[-0.2,-0.7,-0.65],[-0.1,-1.4,-0.45],[-0.02,-1.85,-0.25]],0.055,C.artery,0.4);
  const rcaGrp=new THREE.Group(); rcaGrp.add(rca,sa,rmarg,pda);
  tagPart(rcaGrp,"rca","Rechte Koronararterie (RCA)","Entspringt der Aorta und läuft in der rechten Kranzfurche. Sie gibt den Sinusknoten-Ast (versorgt den Taktgeber), den rechten Randast und am Herzkreuz die hintere absteigende Arterie (PDA) ab, die Hinterwand und hinteres Septum versorgt — bei ~80 % der Menschen (Rechtsversorgertyp).");

  // ---------- HERZVENEN ----------
  const gcv=tube([[0.05,-1.9,0.2],[0.0,-1.0,0.45],[0.1,-0.2,0.5],[0.5,0.45,0.2],[0.9,0.3,-0.3]],0.06,C.coroVein,0.45);
  const cs=tube([[0.9,0.3,-0.3],[0.4,0.2,-0.75],[-0.2,0.2,-0.8],[-0.6,0.35,-0.5]],0.1,C.coroVein,0.45); // Sinus coronarius -> RA
  const veinGrp=new THREE.Group(); veinGrp.add(gcv,cs);
  tagPart(veinGrp,"coronaryveins","Herzvenen & Sinus coronarius","Die große Herzvene (V. cordis magna) begleitet die LAD und sammelt das verbrauchte Blut der Herzmuskulatur. Sie mündet hinten in den Sinus coronarius — eine kurze, weite Vene, die in den rechten Vorhof entleert.");

  // ---------- KLAPPEN (innen, sichtbar beim Aufschneiden) ----------
  const valves=new THREE.Group();
  const ring=(x,y,z,r,rot)=>{const t=new THREE.Mesh(new THREE.TorusGeometry(r,0.05,12,28),mk(C.valve,0.4));t.position.set(x,y,z);if(rot)t.rotation.set(rot[0],rot[1],rot[2]);else t.rotation.x=Math.PI/2;return t;};
  valves.add(ring(-0.4,0.42,0.15,0.2));   // Trikuspidalklappe (RA->RV)
  valves.add(ring(0.45,0.4,-0.15,0.19));  // Mitralklappe (LA->LV)
  valves.add(ring(0.05,0.52,-0.05,0.14)); // Aortenklappe
  valves.add(ring(-0.42,0.6,0.42,0.13));  // Pulmonalklappe
  tagPart(valves,"valves","Herzklappen","Vier Ventile sorgen für die Einbahnstraße des Blutes: Trikuspidal- und Mitralklappe zwischen Vorhöfen und Kammern, Aorten- und Pulmonalklappe an den Ausgängen zu Aorta und Lungenarterie. Beim Schließen entsteht das typische „Lub-Dub“-Herzgeräusch. (Beim Aufschneiden sichtbar.)");

  // ---------- INNERE STRUKTUREN (sichtbar beim Aufschneiden) ----------
  const septum=ell(0.13,1.05,0.66,0x7e2230); septum.position.set(-0.05,-0.35,0.12); septum.rotation.z=0.05;
  tagPart(septum,"septum","Kammerscheidewand (Septum)","Die muskulöse Wand zwischen linker und rechter Kammer. Sie trennt sauerstoffreiches von sauerstoffarmem Blut. Durch ihren oberen, häutigen Teil verläuft das Erregungsleitungssystem. (Beim Aufschneiden sichtbar.)");
  const papGrp=new THREE.Group();
  const pap=(x,y,z)=>{const m=new THREE.Mesh(new THREE.ConeGeometry(0.12,0.42,12),mk(C.myoR));m.position.set(x,y,z);return m;};
  const p1=pap(0.25,-0.75,0.05),p2=pap(0.52,-0.6,-0.15);
  papGrp.add(p1,p2);
  // Sehnenfäden (Chordae) von den Papillarmuskeln zur Mitralklappe
  papGrp.add(tube([[0.25,-0.55,0.05],[0.35,-0.1,-0.05],[0.4,0.35,-0.12]],0.018,C.valve,0.4));
  papGrp.add(tube([[0.52,-0.4,-0.15],[0.48,-0.0,-0.14],[0.5,0.35,-0.16]],0.018,C.valve,0.4));
  tagPart(papGrp,"papillary","Papillarmuskeln & Sehnenfäden","Zapfenförmige Muskeln in den Kammern. Über feine Sehnenfäden (Chordae tendineae) halten sie die Segelklappen fest, damit diese beim kräftigen Pumpen nicht in die Vorhöfe zurückschlagen. (Beim Aufschneiden sichtbar.)");

  g.add(ventricles,leftAtrium,rightAtrium,aortaGrp,pulmGrp,cavae,pv,lca,cxGrp,rcaGrp,veinGrp,septum,papGrp,valves);
  return {group:g,zoom:9,spin:0.0035};
}
function buildLungs(){
  const g=new THREE.Group();const mk=c=>new THREE.MeshStandardMaterial({color:c,roughness:0.7});
  function lung(s){const grp=new THREE.Group();const m=new THREE.Mesh(new THREE.SphereGeometry(0.85,28,28),mk(0xe0959c));m.scale.set(0.75,1.4,0.8);m.position.set(s*0.95,-0.2,0);const lo=new THREE.Mesh(new THREE.SphereGeometry(0.55,24,24),mk(0xe0959c));lo.scale.set(0.8,1.0,0.8);lo.position.set(s*0.95,0.85,0);grp.add(m,lo);return grp;}
  const L=lung(-1),R=lung(1);
  tagPart(L,"lungL","Linker Lungenflügel","Etwas kleiner als der rechte, weil das Herz Platz braucht. Besteht aus zwei Lappen.");
  tagPart(R,"lungR","Rechter Lungenflügel","Der größere Flügel mit drei Lappen. Hier findet ein Großteil des Gasaustauschs statt.");
  const tr=new THREE.Mesh(new THREE.CylinderGeometry(0.16,0.16,1.2,18),mk(0xcf7d86));tr.position.set(0,1.2,0);
  const bL=new THREE.Mesh(new THREE.CylinderGeometry(0.1,0.12,0.8,14),mk(0xcf7d86));bL.position.set(-0.45,0.55,0);bL.rotation.z=0.7;
  const bR=new THREE.Mesh(new THREE.CylinderGeometry(0.1,0.12,0.8,14),mk(0xcf7d86));bR.position.set(0.45,0.55,0);bR.rotation.z=-0.7;
  const air=new THREE.Group();air.add(tr,bL,bR);
  tagPart(air,"trachea","Luftröhre & Bronchien","Die Luftröhre teilt sich in zwei Hauptbronchien, die sich in den Lungen immer feiner verzweigen.");
  g.add(L,R,air);return{group:g,zoom:7,spin:0.0035};
}
function buildKidney(){
  const g=new THREE.Group();const mk=c=>new THREE.MeshStandardMaterial({color:c,roughness:0.6});
  function bean(s){const b=new THREE.Mesh(new THREE.SphereGeometry(0.8,28,28),mk(0x9c3b2e));b.scale.set(0.62,1.15,0.6);b.position.set(s*1.1,0.3,0);b.rotation.z=s*-0.22;return b;}
  const L=bean(-1),R=bean(1);
  tagPart(L,"kidL","Linke Niere","Filtert Abfallstoffe aus dem Blut und bildet Urin. Sitzt meist etwas höher als die rechte.");
  tagPart(R,"kidR","Rechte Niere","Filtert mit der linken zusammen täglich rund 180 Liter Blut. Liegt unter der Leber, daher tiefer.");
  const um=mk(0xc77a6a);
  const uL=new THREE.Mesh(new THREE.CylinderGeometry(0.07,0.07,1.6,12),um);uL.position.set(-0.7,-0.9,0);uL.rotation.z=0.35;
  const uR=new THREE.Mesh(new THREE.CylinderGeometry(0.07,0.07,1.6,12),um);uR.position.set(0.7,-0.9,0);uR.rotation.z=-0.35;
  const bl=new THREE.Mesh(new THREE.SphereGeometry(0.5,24,24),mk(0xd9a14a));bl.position.set(0,-1.85,0);bl.scale.set(1,0.8,0.9);
  const ur=new THREE.Group();ur.add(uL,uR,bl);
  tagPart(ur,"urinary","Harnleiter & Blase","Die Harnleiter transportieren Urin von den Nieren zur Blase, wo er gesammelt wird.");
  g.add(L,R,ur);return{group:g,zoom:7,spin:0.004};
}
function buildSmallIntestine(){
  const g=new THREE.Group();const pts=[],loops=5;
  for(let i=0;i<=150;i++){const t=i/150,ang=t*Math.PI*2*loops,r=0.95+0.28*Math.sin(t*Math.PI*9);
    pts.push(new THREE.Vector3(Math.cos(ang)*r,(t-0.5)*2.7+0.14*Math.sin(t*Math.PI*14),Math.sin(ang)*r));}
  const tube=new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts),320,0.22,12,false),new THREE.MeshStandardMaterial({color:0xe08a7a,roughness:0.7}));
  tagPart(tube,"small","Dünndarm","Rund 6 Meter lang und stark gewunden. Hier wird der Großteil der Nährstoffe ins Blut aufgenommen.");
  g.add(tube);return{group:g,zoom:7,spin:0.004};
}
function buildLargeIntestine(){
  const g=new THREE.Group();
  const p=[new THREE.Vector3(1.2,-1.6,0),new THREE.Vector3(1.45,-0.5,0),new THREE.Vector3(1.45,0.9,0),new THREE.Vector3(1.2,1.65,0),new THREE.Vector3(0,1.95,0),new THREE.Vector3(-1.2,1.65,0),new THREE.Vector3(-1.45,0.9,0),new THREE.Vector3(-1.45,-0.6,0),new THREE.Vector3(-0.9,-1.6,0),new THREE.Vector3(-0.2,-2.0,0)];
  const tube=new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(p),220,0.34,14,false),new THREE.MeshStandardMaterial({color:0xc98a5a,roughness:0.75}));
  tagPart(tube,"colon","Dickdarm","Bildet den Rahmen um den Dünndarm. Entzieht dem Nahrungsbrei Wasser und formt den Stuhl.");
  g.add(tube);return{group:g,zoom:7,spin:0.0035};
}
function buildCrystal(){
  const g=new THREE.Group();const geo=new THREE.IcosahedronGeometry(1.6,1);
  const m=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:0x2dd4bf,metalness:0.3,roughness:0.35,flatShading:true}));
  tagPart(m,"crystal","Testobjekt","Ein einfaches Objekt zum Ausprobieren der Steuerung.");
  g.add(m,new THREE.LineSegments(new THREE.WireframeGeometry(geo),new THREE.LineBasicMaterial({color:0xaef7ec,transparent:true,opacity:0.25})));
  return{group:g,zoom:6,spin:0.0035};
}

// ---------------------------------------------------------------------
//  Presets:  build = Platzhalter,  model = echtes GLB (optional)
// ---------------------------------------------------------------------
const PRESETS = {
  dna:{title:"DNA Doppelhelix",subtitle:"Rückgrat · Basenpaare",build:buildDNA,
    intro:"Die DNA ist der Bauplan des Lebens. In jeder Zelle liegt sie als verdrillte Doppelhelix vor. Die Abfolge der vier Basen (A, T, G, C) speichert alle Erbinformationen."},
  brain:{title:"Menschliches Gehirn",subtitle:"Großhirn · Kleinhirn · Hirnstamm",build:buildBrain,model:"models/brain.glb",
    intro:"Das Gehirn ist die Steuerzentrale des Körpers. Es wiegt rund 1,3 kg und besitzt etwa 86 Milliarden Nervenzellen. Es verarbeitet Sinneseindrücke, steuert Bewegungen und ist Sitz von Denken, Gefühl und Gedächtnis."},
  heart:{title:"Menschliches Herz",subtitle:"Kammern · Klappen · Kranzgefäße · große Gefäße",build:buildHeart,beat:true,
    intro:"Das Herz ist ein faustgroßer Muskel, der unermüdlich Blut durch den Körper pumpt — rund 100.000 Mal am Tag. Es hat vier Räume (zwei Vorhöfe, zwei Kammern), vier Klappen, und wird von den Herzkranzgefäßen selbst versorgt. Klick die einzelnen Strukturen an — oder schneide das Herz auf, um die Klappen im Inneren zu sehen."},
  lungs:{title:"Menschliche Lunge",subtitle:"Lungenflügel · Atemwege",build:buildLungs,model:"models/lungs.glb",
    intro:"Die Lunge versorgt das Blut mit Sauerstoff und gibt Kohlendioxid ab. In den rund 300 Millionen Lungenbläschen findet der Gasaustausch statt. Ein Erwachsener atmet etwa 12–16 Mal pro Minute."},
  kidney:{title:"Nieren",subtitle:"Niere · Harnleiter · Blase",build:buildKidney,model:"models/kidney.glb",
    intro:"Die beiden Nieren sind die Kläranlage des Körpers. Sie filtern täglich rund 180 Liter Blut, entfernen Abfallstoffe und regulieren Wasser- und Salzhaushalt sowie den Blutdruck."},
  smallintestine:{title:"Dünndarm",subtitle:"Nährstoffaufnahme",build:buildSmallIntestine,model:"models/smallintestine.glb",
    intro:"Der Dünndarm ist mit rund 6 Metern der längste Abschnitt des Verdauungstrakts. Seine faltige Innenwand vergrößert die Oberfläche enorm, damit möglichst viele Nährstoffe ins Blut gelangen."},
  largeintestine:{title:"Dickdarm",subtitle:"Wasserentzug · Kolon",build:buildLargeIntestine,model:"models/largeintestine.glb",
    intro:"Der Dickdarm umrahmt den Dünndarm. Er entzieht dem Nahrungsbrei Wasser und Salze, beherbergt Billionen nützlicher Bakterien und formt den Stuhl."},
  crystal:{title:"Testobjekt",subtitle:"Phase 1 Prototyp",build:buildCrystal,
    intro:"Ein einfaches Objekt zum Testen der Steuerung."}
};

// ---------------------------------------------------------------------
//  CSS
// ---------------------------------------------------------------------
const CSS=`
:root{--bg:#070b0f;--bg2:#0d141b;--accent:#2dd4bf;--accent-dim:#155e57;--text:#e6f1ef;--text-dim:#6b8a86;--warn:#f5a524;--line:rgba(45,212,191,.18)}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:'JetBrains Mono',monospace}
#ae-scene{position:fixed;inset:0;z-index:1}
body::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;background:radial-gradient(circle at 50% 45%,rgba(45,212,191,.10),transparent 55%),radial-gradient(circle at 80% 90%,rgba(45,212,191,.05),transparent 50%)}
.ae-hud{position:fixed;z-index:10;pointer-events:none}
#ae-top{top:0;left:0;right:0;display:flex;align-items:center;justify-content:space-between;padding:16px 22px}
.ae-brand{display:flex;align-items:center;gap:12px}
.ae-back{pointer-events:auto;text-decoration:none;color:var(--text-dim);font-size:18px;border:1px solid var(--line);border-radius:8px;width:34px;height:34px;display:flex;align-items:center;justify-content:center;transition:.2s}
.ae-back:hover{color:var(--accent);border-color:var(--accent)}
.ae-dot{width:9px;height:9px;border-radius:50%;background:var(--text-dim);transition:.3s}
.ae-dot.live{background:var(--accent);box-shadow:0 0 12px 1px var(--accent);animation:aepulse 2s infinite}
@keyframes aepulse{0%,100%{opacity:1}50%{opacity:.4}}
.ae-brand h1{font-size:13px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase}
.ae-brand span{font-size:11px;color:var(--text-dim);letter-spacing:1px}
#ae-status{font-size:11px;color:var(--text-dim);letter-spacing:1px;text-align:right;line-height:1.6}
#ae-status b{color:var(--accent);font-weight:500}
#ae-legend{bottom:20px;left:22px;font-size:11px;color:var(--text-dim);line-height:1.9;letter-spacing:.5px}
#ae-legend .k{color:var(--text)}#ae-legend .sep{color:var(--accent-dim)}
#ae-cam{bottom:20px;right:22px;width:200px;height:150px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--bg2)}
#ae-cam.hidden{display:none}
#ae-video,#ae-overlay{position:absolute;inset:0;width:100%;height:100%;transform:scaleX(-1)}
#ae-video{object-fit:cover}
#ae-camlabel{position:absolute;top:6px;left:8px;z-index:2;font-size:9px;letter-spacing:1px;color:var(--accent);text-shadow:0 0 6px var(--bg)}
#ae-tools{top:64px;right:22px;display:flex;flex-direction:column;gap:8px;align-items:flex-end}
.ae-btn{pointer-events:auto;font-family:inherit;font-size:11px;letter-spacing:1px;color:var(--text);background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer;display:flex;align-items:center;gap:6px;transition:.2s;white-space:nowrap}
.ae-btn:hover{border-color:var(--accent);color:var(--accent)}
.ae-btn.active{border-color:var(--accent);color:var(--accent)}
#ae-panel{position:fixed;z-index:12;top:50%;right:22px;transform:translateY(-50%);width:290px;background:var(--bg2);border:1px solid var(--line);border-radius:12px;padding:18px;opacity:0;pointer-events:none;transition:opacity .25s}
#ae-panel.show{opacity:1;pointer-events:auto}
#ae-panel .ph{display:flex;align-items:center;gap:8px;margin-bottom:10px}
#ae-panel .ph .pdot{width:9px;height:9px;border-radius:50%;background:var(--warn)}
#ae-panel .ph.ov .pdot{background:var(--accent)}
#ae-panel h3{font-size:15px;font-weight:700}
#ae-panel .pinfo{font-size:12px;color:var(--text-dim);line-height:1.7;margin-bottom:14px}
#ae-note-wrap label{font-size:10px;letter-spacing:1px;color:var(--accent);text-transform:uppercase}
#ae-panel textarea{width:100%;margin-top:6px;background:var(--bg);border:1px solid var(--line);border-radius:8px;color:var(--text);font-family:inherit;font-size:12px;padding:8px;resize:vertical;min-height:60px}
#ae-panel textarea:focus{outline:none;border-color:var(--accent)}
#ae-pclose{position:absolute;top:12px;right:14px;cursor:pointer;color:var(--text-dim);background:none;border:none;font-size:18px;font-family:inherit}
#ae-pclose:hover{color:var(--text)}
#ae-saved{font-size:10px;color:var(--accent);margin-top:6px;height:12px;letter-spacing:1px}
#ae-load{position:fixed;inset:0;z-index:20;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;background:var(--bg);transition:opacity .4s}
#ae-load.gone{opacity:0;pointer-events:none}
#ae-load .ring{width:46px;height:46px;border:3px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#ae-load p{font-size:12px;color:var(--text-dim);letter-spacing:1px}
#ae-gate{position:fixed;inset:0;z-index:30;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px;text-align:center;padding:24px;background:radial-gradient(circle at 50% 40%,rgba(13,20,27,.6),var(--bg));backdrop-filter:blur(2px)}
#ae-gate.gone{opacity:0;pointer-events:none;transition:opacity .6s}
#ae-gate h2{font-size:22px;font-weight:700}
#ae-gate p{font-size:13px;color:var(--text-dim);max-width:430px;line-height:1.7}
#ae-start{pointer-events:auto;font-family:inherit;font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--bg);background:var(--accent);border:none;border-radius:8px;padding:14px 34px;cursor:pointer;box-shadow:0 0 30px rgba(45,212,191,.4);transition:.2s}
#ae-start:hover{transform:translateY(-2px);box-shadow:0 0 44px rgba(45,212,191,.6)}
#ae-start:disabled{opacity:.5;cursor:wait}
#ae-skip{pointer-events:auto;background:none;border:none;cursor:pointer;color:var(--text-dim);font-family:inherit;font-size:11px;letter-spacing:1px;text-decoration:underline;text-underline-offset:4px}
#ae-skip:hover{color:var(--text)}
#ae-err{color:var(--warn);font-size:12px;min-height:16px;max-width:430px}
@media(max-width:640px){#ae-cam{width:130px;height:98px}#ae-panel{width:calc(100% - 44px);top:auto;bottom:170px;transform:none}#ae-tools{top:58px;right:12px}}
`;

// ---------------------------------------------------------------------
//  Hauptfunktion
// ---------------------------------------------------------------------
export function mountViewer(config){
  const preset = PRESETS[config.preset] || PRESETS.crystal;
  const modelUrl = config.model || preset.model || null;

  const style=document.createElement("style");style.textContent=CSS;document.head.appendChild(style);

  document.body.innerHTML=`
  <div id="ae-scene"></div>
  <div id="ae-top" class="ae-hud">
    <div class="ae-brand"><a class="ae-back" href="index.html" title="Übersicht">&#8592;</a>
      <div class="ae-dot" id="ae-dot"></div>
      <div><h1>${preset.title}</h1><span>${preset.subtitle}</span></div></div>
    <div id="ae-status">Kamera: <b id="ae-camstate">aus</b><br>Hände: <b id="ae-handstate">0</b></div>
  </div>
  <div id="ae-tools" class="ae-hud">
    <button class="ae-btn" id="ae-infobtn">&#9432; Überblick</button>
    <button class="ae-btn" id="ae-openbtn">&#9697; Aufschneiden</button>
    <button class="ae-btn" id="ae-modebtn">Modus: Schnitt</button>
    <button class="ae-btn active" id="ae-rotbtn">&#8635; Auto-Drehung</button>
    <button class="ae-btn" id="ae-resetbtn">&#8634; Zurücksetzen</button>
    <button class="ae-btn" id="ae-loadbtn">&#8675; Modell laden</button>
    <input id="ae-file" type="file" accept=".glb,.gltf" style="display:none">
  </div>
  <div id="ae-legend" class="ae-hud">
    <div><span class="k">Pinch</span> <span class="sep">&#8594;</span> drehen</div>
    <div><span class="k">2 Hände</span> <span class="sep">&#8594;</span> zoomen</div>
    <div><span class="k">2 Hände + Pinch</span> <span class="sep">&#8594;</span> aufschneiden</div>
    <div><span class="k">Klick</span> <span class="sep">&#8594;</span> Teil-Info</div>
  </div>
  <div id="ae-cam" class="ae-hud hidden"><span id="ae-camlabel">LIVE</span>
    <video id="ae-video" autoplay playsinline muted></video><canvas id="ae-overlay"></canvas></div>
  <div id="ae-panel"><button id="ae-pclose">&times;</button>
    <div class="ph" id="ae-phead"><span class="pdot"></span><h3 id="ae-ptitle"></h3></div>
    <p class="pinfo" id="ae-pinfo"></p>
    <div id="ae-note-wrap"><label>Deine Notiz</label>
      <textarea id="ae-pnote" placeholder="z.B. für die Klausur merken..."></textarea>
      <div id="ae-saved"></div></div>
  </div>
  <div id="ae-load"><div class="ring"></div><p id="ae-loadtxt">lädt Modell…</p></div>
  <div id="ae-gate"><h2>${preset.title}</h2>
    <p>Kamera öffnen und mit der Hand steuern: Pinch zum Drehen, zwei Hände zum Zoomen, zwei gegriffene Hände auseinander zum Aufschneiden. Klick auf ein Teil zeigt Infos. Es wird nichts aufgenommen.</p>
    <button id="ae-start">Kamera starten</button>
    <button id="ae-skip">Ohne Kamera ansehen (nur Maus)</button>
    <div id="ae-err"></div></div>`;

  const sceneEl=document.getElementById("ae-scene");
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.setSize(window.innerWidth,window.innerHeight);
  renderer.localClippingEnabled=true;
  sceneEl.appendChild(renderer.domElement);

  const scene=new THREE.Scene();
  const camera=new THREE.PerspectiveCamera(45,window.innerWidth/window.innerHeight,0.1,100);
  scene.add(new THREE.AmbientLight(0x99ffee,0.6));
  const key=new THREE.DirectionalLight(0xffffff,1.5);key.position.set(4,5,6);scene.add(key);
  const fill=new THREE.DirectionalLight(0xbfe9ff,0.5);fill.position.set(-5,2,-4);scene.add(fill);
  const rim=new THREE.PointLight(0x2dd4bf,1.6,40);rim.position.set(-6,-3,4);scene.add(rim);

  const group=new THREE.Group();scene.add(group);
  const clipPlane=new THREE.Plane(new THREE.Vector3(0,0,-1),5); // konstant 5 = nichts geschnitten
  let topParts=[];           // für Explosion
  let defaultZoom=7, spin=0.004;

  // --- Inhalt vorbereiten (GLB oder Platzhalter) ---
  function applyContent(obj){
    group.clear();topParts=[];
    group.add(obj);
    obj.traverse(o=>{
      if(o.isMesh && o.material){
        const mats=Array.isArray(o.material)?o.material:[o.material];
        mats.forEach(m=>{m.side=THREE.DoubleSide;m.clippingPlanes=[clipPlane];m.clipShadows=true;});
      }
    });
    topParts=group.children.filter(c=>c.userData && c.userData.part).map(c=>({obj:c,base:c.position.clone()}));
    if(topParts.length===0){topParts=obj.children.filter(c=>c.userData&&c.userData.part).map(c=>({obj:c,base:c.position.clone()}));}
    camera.position.set(0,0,defaultZoom);
    targetZoom=defaultZoom;
    document.getElementById("ae-load").classList.add("gone");
  }

  function buildPlaceholder(){
    const b=preset.build();spin=b.spin;defaultZoom=b.zoom;
    return b.group;
  }

  function fitModel(root){
    const box=new THREE.Box3().setFromObject(root);
    const size=box.getSize(new THREE.Vector3());
    const center=box.getCenter(new THREE.Vector3());
    root.position.sub(center);
    const maxDim=Math.max(size.x,size.y,size.z)||1;
    const s=3.2/maxDim;root.scale.setScalar(s);
    const wrap=new THREE.Group();wrap.add(root);
    tagPart(wrap,"model",preset.title,preset.intro);
    return wrap;
  }

  const gltfLoader=new GLTFLoader();
  const draco=new DRACOLoader();
  draco.setDecoderPath("https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/draco/");
  gltfLoader.setDRACOLoader(draco);
  function showLoader(t){const l=document.getElementById("ae-load");l.classList.remove("gone");document.getElementById("ae-loadtxt").textContent=t||"lädt Modell…";}
  function loadFromURL(url){showLoader();spin=0.004;defaultZoom=7;gltfLoader.load(url,g=>applyContent(fitModel(g.scene)),undefined,err=>{console.warn("GLB nicht gefunden ("+url+") — Platzhalter.",err);applyContent(buildPlaceholder());});}
  function loadFromBuffer(buf){showLoader();spin=0.004;defaultZoom=7;try{gltfLoader.parse(buf,"",g=>{applyContent(fitModel(g.scene));setTimeout(()=>openPanel({partId:null,label:preset.title,info:preset.intro,overview:true}),300);},err=>{console.error(err);document.getElementById("ae-load").classList.add("gone");alert("Modell konnte nicht geladen werden. Versuch ein anderes .glb.");});}catch(e){console.error(e);document.getElementById("ae-load").classList.add("gone");alert("Modell konnte nicht geladen werden. Versuch ein anderes .glb.");}}

  // --- gemeinsamer Eingabe-Zustand ---
  let targetRotX=0,targetRotY=0,targetZoom=7;
  const ZOOM_MIN=3,ZOOM_MAX=14;
  let autoRotate=true,interacting=false;
  let openMode="cut";          // "cut" | "explode"
  let targetOpen=0,curOpen=0;   // 0..1

  // Inhalt jetzt laden (nach den Zustands-Variablen — wichtig!)
  if(modelUrl)loadFromURL(modelUrl);else applyContent(buildPlaceholder());

  // eigenes Modell laden: Button + Drag & Drop
  const fileInput=document.getElementById("ae-file");
  document.getElementById("ae-loadbtn").addEventListener("click",()=>fileInput.click());
  fileInput.addEventListener("change",e=>{const f=e.target.files[0];if(f)f.arrayBuffer().then(loadFromBuffer);});
  window.addEventListener("dragover",e=>{e.preventDefault();});
  window.addEventListener("drop",e=>{e.preventDefault();const f=e.dataTransfer&&e.dataTransfer.files[0];if(f&&/\.(glb|gltf)$/i.test(f.name))f.arrayBuffer().then(loadFromBuffer);else if(f)alert("Bitte eine .glb- oder .gltf-Datei verwenden.");});

  // Maus
  let dragging=false,lastMX=0,lastMY=0,moved=0;
  renderer.domElement.addEventListener("pointerdown",e=>{dragging=true;moved=0;lastMX=e.clientX;lastMY=e.clientY;});
  window.addEventListener("pointerup",e=>{if(dragging&&moved<6)tryPick(e.clientX,e.clientY);dragging=false;});
  window.addEventListener("pointermove",e=>{if(!dragging)return;moved+=Math.abs(e.clientX-lastMX)+Math.abs(e.clientY-lastMY);interacting=true;targetRotY+=(e.clientX-lastMX)*0.006;targetRotX+=(e.clientY-lastMY)*0.006;targetRotX=Math.max(-1.4,Math.min(1.4,targetRotX));lastMX=e.clientX;lastMY=e.clientY;});
  renderer.domElement.addEventListener("wheel",e=>{e.preventDefault();interacting=true;targetZoom=Math.max(ZOOM_MIN,Math.min(ZOOM_MAX,targetZoom+e.deltaY*0.005));},{passive:false});

  // Buttons
  const rotBtn=document.getElementById("ae-rotbtn");
  rotBtn.addEventListener("click",()=>{autoRotate=!autoRotate;rotBtn.classList.toggle("active",autoRotate);});
  document.getElementById("ae-resetbtn").addEventListener("click",()=>{targetRotX=0;targetRotY=0;targetZoom=defaultZoom;targetOpen=0;});
  const openBtn=document.getElementById("ae-openbtn");
  openBtn.addEventListener("click",()=>{targetOpen=targetOpen>0.05?0:1;openBtn.classList.toggle("active",targetOpen>0.05);});
  const modeBtn=document.getElementById("ae-modebtn");
  modeBtn.addEventListener("click",()=>{openMode=openMode==="cut"?"explode":"cut";modeBtn.textContent="Modus: "+(openMode==="cut"?"Schnitt":"Teile");});

  // Raycast / Panel
  const raycaster=new THREE.Raycaster();const mouse=new THREE.Vector2();
  const panel=document.getElementById("ae-panel");
  const pTitle=document.getElementById("ae-ptitle"),pInfo=document.getElementById("ae-pinfo");
  const pNote=document.getElementById("ae-pnote"),pSaved=document.getElementById("ae-saved");
  const pHead=document.getElementById("ae-phead"),noteWrap=document.getElementById("ae-note-wrap");
  let currentPartId=null;
  function tryPick(x,y){mouse.x=(x/window.innerWidth)*2-1;mouse.y=-(y/window.innerHeight)*2+1;raycaster.setFromCamera(mouse,camera);const hits=raycaster.intersectObjects(group.children,true);for(const h of hits){let o=h.object;while(o&&!(o.userData&&o.userData.part))o=o.parent;if(o&&o.userData.part){openPanel(o.userData.part);return;}}}
  function noteKey(id){return "anote:"+config.preset+":"+id;}
  function openPanel(part){
    currentPartId=part.partId;pTitle.textContent=part.label;pInfo.textContent=part.info;
    if(part.overview){pHead.classList.add("ov");noteWrap.style.display="none";}
    else{pHead.classList.remove("ov");noteWrap.style.display="block";
      let s="";try{s=localStorage.getItem(noteKey(part.partId))||"";}catch(e){}pNote.value=s;pSaved.textContent="";}
    panel.classList.add("show");
  }
  pNote.addEventListener("input",()=>{if(!currentPartId)return;try{localStorage.setItem(noteKey(currentPartId),pNote.value);pSaved.textContent="gespeichert";}catch(e){pSaved.textContent="speichern nicht möglich";}});
  document.getElementById("ae-pclose").addEventListener("click",()=>panel.classList.remove("show"));
  document.getElementById("ae-infobtn").addEventListener("click",()=>openPanel({partId:null,label:preset.title,info:preset.intro,overview:true}));
  // Überblick beim Start zeigen
  setTimeout(()=>openPanel({partId:null,label:preset.title,info:preset.intro,overview:true}),400);

  // Render-Schleife
  const _v=new THREE.Vector3();
  function animate(){
    requestAnimationFrame(animate);
    if(autoRotate&&!interacting)targetRotY+=spin;
    group.rotation.x+=(targetRotX-group.rotation.x)*0.12;
    group.rotation.y+=(targetRotY-group.rotation.y)*0.12;
    camera.position.z+=(targetZoom-camera.position.z)*0.12;
    curOpen+=(targetOpen-curOpen)*0.12;
    // Aufschneiden / Explosion anwenden
    if(openMode==="cut"){
      clipPlane.constant=5-curOpen*5.5;            // 5 (zu) -> -0.5 (offen)
      topParts.forEach(p=>p.obj.position.copy(p.base));
    }else{
      clipPlane.constant=5;                        // Schnitt aus
      topParts.forEach(p=>{const dir=p.base.length()>0.001?_v.copy(p.base).normalize():_v.set(0,1,0);p.obj.position.copy(p.base).add(dir.multiplyScalar(curOpen*1.4));});
    }
    // Herzschlag
    if(preset.beat){const t=performance.now()/1000*(72/60);const ph=t%1;const pulse=Math.exp(-((ph-0.0)**2)/0.004)+0.6*Math.exp(-((ph-0.28)**2)/0.004);group.scale.setScalar(1+0.045*pulse);}
    interacting=false;
    renderer.render(scene,camera);
  }
  animate();
  window.addEventListener("resize",()=>{camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight);});

  // --- Hand-Tracking ---
  const video=document.getElementById("ae-video"),overlay=document.getElementById("ae-overlay"),octx=overlay.getContext("2d");
  const camState=document.getElementById("ae-camstate"),handState=document.getElementById("ae-handstate"),dot=document.getElementById("ae-dot"),errEl=document.getElementById("ae-err");
  let handLandmarker=null,lastVideoTime=-1,pinching=false,lastPX=0,lastPY=0,twoStartDist=null,twoStartZoom=null,openStartDist=null,openStartVal=null;
  const fPX=new OneEuro(),fPY=new OneEuro(),fDist=new OneEuro(1.0,0.01),fOpen=new OneEuro(1.0,0.01);
  const CONN=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  const d2=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);

  async function startCamera(){
    errEl.textContent="";
    try{
      const fs=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm");
      handLandmarker=await HandLandmarker.createFromOptions(fs,{baseOptions:{modelAssetPath:"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",delegate:"GPU"},runningMode:"VIDEO",numHands:2});
      const stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:"user"}});
      video.srcObject=stream;await video.play();
      overlay.width=video.videoWidth||640;overlay.height=video.videoHeight||480;
      document.getElementById("ae-cam").classList.remove("hidden");camState.textContent="an";dot.classList.add("live");detectLoop();
    }catch(e){console.error(e);errEl.textContent="Kamera/Modell-Fehler: "+(e.message||e);}
  }
  function detectLoop(){if(handLandmarker&&video.readyState>=2&&video.currentTime!==lastVideoTime){lastVideoTime=video.currentTime;handleHands(handLandmarker.detectForVideo(video,performance.now()));}requestAnimationFrame(detectLoop);}
  function handleHands(res){
    const hands=res.landmarks||[];handState.textContent=hands.length;
    octx.clearRect(0,0,overlay.width,overlay.height);octx.strokeStyle="rgba(45,212,191,.7)";octx.fillStyle="#aef7ec";octx.lineWidth=2;
    for(const lm of hands){for(const [a,b] of CONN){octx.beginPath();octx.moveTo(lm[a].x*overlay.width,lm[a].y*overlay.height);octx.lineTo(lm[b].x*overlay.width,lm[b].y*overlay.height);octx.stroke();}for(const p of lm){octx.beginPath();octx.arc(p.x*overlay.width,p.y*overlay.height,3,0,6.28);octx.fill();}}
    const now=performance.now();
    if(hands.length>=2){
      pinching=false;
      const sizeA=Math.max(d2(hands[0][0],hands[0][9]),1e-4),sizeB=Math.max(d2(hands[1][0],hands[1][9]),1e-4);
      const pinA=d2(hands[0][4],hands[0][8])/sizeA,pinB=d2(hands[1][4],hands[1][8])/sizeB;
      const wristDist=d2(hands[0][0],hands[1][0]);
      if(pinA<0.45&&pinB<0.45){ // beide gegriffen -> aufschneiden
        twoStartDist=null;const d=fOpen.filter(wristDist,now);
        if(openStartDist===null){openStartDist=d;openStartVal=targetOpen;}
        interacting=true;targetOpen=Math.max(0,Math.min(1,openStartVal+(d-openStartDist)*2.2));
        if(targetOpen>0.05)openBtn.classList.add("active");else openBtn.classList.remove("active");
      } else { // zoomen
        openStartDist=null;const d=fDist.filter(wristDist,now);
        if(twoStartDist===null){twoStartDist=d;twoStartZoom=targetZoom;}
        interacting=true;targetZoom=Math.max(ZOOM_MIN,Math.min(ZOOM_MAX,twoStartZoom*(twoStartDist/Math.max(d,1e-4))));
      }
      return;
    }
    twoStartDist=null;openStartDist=null;
    if(hands.length===1){
      const lm=hands[0];const handSize=Math.max(d2(lm[0],lm[9]),1e-4);const ratio=d2(lm[4],lm[8])/handSize;
      const px=fPX.filter(1-(lm[4].x+lm[8].x)/2,now),py=fPY.filter((lm[4].y+lm[8].y)/2,now);
      if(!pinching&&ratio<0.40){pinching=true;lastPX=px;lastPY=py;}
      else if(pinching&&ratio>0.55){pinching=false;}
      if(pinching){interacting=true;targetRotY+=(px-lastPX)*5.0;targetRotX+=(py-lastPY)*5.0;targetRotX=Math.max(-1.4,Math.min(1.4,targetRotX));lastPX=px;lastPY=py;}
      return;
    }
    pinching=false;
  }

  const gate=document.getElementById("ae-gate"),startBtn=document.getElementById("ae-start");
  document.getElementById("ae-skip").addEventListener("click",()=>gate.classList.add("gone"));
  startBtn.addEventListener("click",async()=>{startBtn.disabled=true;startBtn.textContent="lädt…";await startCamera();if(!errEl.textContent)gate.classList.add("gone");startBtn.disabled=false;startBtn.textContent="Kamera starten";});
}
