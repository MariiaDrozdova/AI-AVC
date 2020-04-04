
penspacing=18
numpens=6

body = cube(penspacing*numpens, 15, 50)

rfoot = translate(-penspacing*numpens/2,0,0)*cone(15,15/2,50)
lfoot = translate(penspacing*numpens/2,0,0)*cone(15,15/2,50)

--emit(rfoot)
--emit(lfoot)

--emit(sphere(1))

c = translate(-penspacing * numpens/2 ,14,28) * scale(1.0, 0.6,1.0)* rotate(0,90,0)*cylinder(20, penspacing * numpens)

body = union{body,lfoot,rfoot}
pentip=18
pen = union(
    cylinder(5,150),
    translate(0,0,-pentip)*cone(1,5,pentip)
)
pen = translate(-numpens/2*penspacing+penspacing/2,0,19)*pen
--emit(pen)
allpens={}
for i=0,numpens-1 do
  table.insert(allpens,pen)
  pen = translate(penspacing,0,0)*pen
end
o = difference{body,union(allpens)}
emit(
  difference(o, c)
)