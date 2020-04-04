nbArm = 3
width=30
length0 = 60 + 1
thickness = 5.2
r = 6
ring_radius = 5



ring = translate(0, 0, -thickness/2) * difference(cylinder(ring_radius + r, thickness), cylinder(ring_radius, thickness + 1))

--emit(sphere(1))
arm = translate(-length0/2-ring_radius,0,0)*ccube(length0,width,thickness)
--emit(arm)
cb = translate(-length0-ring_radius,0,-thickness/2)*cube(length0/20, width, thickness)

arms = {}
cubes = {}

for i=0,nbArm+1 do
  table.insert(arms, arm)
  table.insert(cubes, cb)
  arm = rotate(0, 0, 360/nbArm)*arm
  cb = rotate(0, 0, 360/nbArm)*cb
end

c = translate(0,0,-thickness/2) * cylinder(r, thickness)
tripod = difference{union(arms), c}
cubes = union(cubes)
emit(ring, 0)
emit(
  tripod, 1
)
emit(cubes, 0)

--density = tex3d_r8f(64,64,64)
--angle   = tex3d_r8f(64,64,64)
--shrink  = tex3d_r8f(64,64,64)

-- Fill them in
-- Values in 3D texture are always in [0,1], thus for
--   density, 0 is min, 1 is max
--   angle,   0 is 0 degree, 1 is 360 degrees
--   etc.
-- Only the first component of the textures is used ('red')

anglefield = tex3d_r8f(64,64,64)
density = tex3d_r8f(64,64,64)
for i = 0,63 do
    for j = 0,63 do
        for k = 0,63 do
          tu = (i - 31)/32
          tv = (j - 31)/32
          f = math.atan2(tu, tv)/math.pi
          a = f 
          if (a > 1.0/3+1.0/9) then a=2.0/3 end
          if (a < -1.0+1.0/9) then a=2.0/3 end
          if (a < -1.0/3+1.0/9) then a=3.0/3 end
          if (a < 1.0/3+1.0/9) then a = 1.0/3 end

          p = tu*tu + tv*tv
          max_density= 50
          min_density = 20
          d = 0
          if p >= 1 then d = 1 end
          v1 = (1 - p)
          v2 = p
          v3 = 0
          threshold = 0.8
          if v1 < threshold then v1=0 end 
          if v2 < threshold then v2=0 end
          if v1 < threshold and v2 < threshold then v3=threshold end
          d = v1 + v2 + v3
          if d > 1 then d =0.5 end
          anglefield:set(i,j,k,v(a,0,0,0))
          density:set(i,j,k,v(d,0,0,0))
        end
    end
end 
 
-- Set the 3D textures to the field settings
-- The binding requires a field (!), a bounding box where it is applied, and the internal name of the parameter (see tooltip in UI)
set_setting_value('infill_angle_1',1)--anglefield,v(-length0,-length0,0),v(length0,length0,thickness))
set_setting_value('kgon_x_shrink_1', 0.2)
set_setting_value('infill_type_1', 'Polyfoam')
set_setting_value('infill_percentage_1', density, v(-length0,-length0,0),v(length0,length0,thickness))
set_setting_value('print_perimeter_1', false)
set_setting_value('cover_thickness_mm_1', 0)
set_setting_value('num_shells_1', 0)

set_setting_value('num_shells_0', 10)
set_setting_value('print_perimeter_0', true)
