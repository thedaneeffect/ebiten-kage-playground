package main

func interpolate_vec4(v1, v2, v3 vec4, f vec3) (result vec4) {
	result = result.Add(v1.Mul(f.X()))
	result = result.Add(v2.Mul(f.Y()))
	result = result.Add(v3.Mul(f.Z()))
	return
}

func interpolate_vec3_4(v1, v2, v3 vec4, f vec3) (result vec4) {
	result = result.Add(v1.Mul(f.X()))
	result = result.Add(v2.Mul(f.Y()))
	result = result.Add(v3.Mul(f.Z()))
	return
}

func interpolate_vec3(v1, v2, v3 vec3, f vec3) (result vec3) {
	result = result.Add(v1.Mul(f.X()))
	result = result.Add(v2.Mul(f.Y()))
	result = result.Add(v3.Mul(f.Z()))
	return
}

func interpolate_vec2(v1, v2, v3 vec2, f vec3) (result vec2) {
	result = result.Add(v1.Mul(f.X()))
	result = result.Add(v2.Mul(f.Y()))
	result = result.Add(v3.Mul(f.Z()))
	return
}

func interpolate_vertex(v1, v2, v3 vertex, f vec3) (result vertex) {
	result.pos = interpolate_vec3_4(v1.pos, v2.pos, v3.pos, f)
	result.rgba = interpolate_vec4(v1.rgba, v2.rgba, v3.rgba, f)
	result.uv = interpolate_vec2(v1.uv, v2.uv, v3.uv, f)
	return
}

func point_in_triangle_bounds(x, y, xA, yA, xB, yB, xC, yC float32) bool {
	if (y < yA) && (y < yB) && (y < yC) {
		return false
	}
	if (y > yA) && (y > yB) && (y > yC) {
		return false
	}
	if (x < xA) && (x < xB) && (x < xC) {
		return false
	}
	return (x <= xA) || (x <= xB) || (x <= xC)
}

func point_in_triangle(x, y, x1, y1, x2, y2, x3, y3 float32) bool {
	v0x, v0y := x3-x1, y3-y1
	v1x, v1y := x2-x1, y2-y1
	v2x, v2y := x-x1, y-y1
	dot00 := v0x*v0x + v0y*v0y
	dot01 := v0x*v1x + v0y*v1y
	dot02 := v0x*v2x + v0y*v2y
	dot11 := v1x*v1x + v1y*v1y
	dot12 := v1x*v2x + v1y*v2y
	b := dot00*dot11 - dot01*dot01
	var inv float32
	if b != 0 {
		inv = 1.0 / b
	}
	u := (dot11*dot02 - dot01*dot12) * inv
	v := (dot00*dot12 - dot01*dot02) * inv
	return u >= 0 && v >= 0 && (u+v < 1.0)
}

type plane struct {
	origin vec4
	normal vec4
}

// test determines if `v` is in front of the plane.
func (p plane) test(v vec4) bool {
	return v.Sub(p.origin).Dot(p.normal) > 0
}

// intersection returns the point of contact of a line segment between a->b to our plane.
func (p plane) intersection(a, b vec4) vec4 {
	u := b.Sub(a)
	w := a.Sub(p.origin)
	d := p.normal.Dot(u)
	n := -p.normal.Dot(w)
	return a.Add(u.Mul(n / d))
}

var clip_planes = [...]plane{
	{origin: vec4{1, 0, 0, 1}, normal: vec4{-1, 0, 0, 1}}, // right
	{origin: vec4{-1, 0, 0, 1}, normal: vec4{1, 0, 0, 1}}, // left
	{origin: vec4{0, 1, 0, 1}, normal: vec4{0, -1, 0, 1}}, // bottom
	{origin: vec4{0, -1, 0, 1}, normal: vec4{0, 1, 0, 1}}, // top
	{origin: vec4{0, 0, 1, 1}, normal: vec4{0, 0, -1, 1}}, // front
	{origin: vec4{0, 0, -1, 1}, normal: vec4{0, 0, 1, 1}}, // back
}

var scratch1 = [9]vec4{} // 9 is a safe number to ensure we never
var scratch2 = [9]vec4{} // run out of space while clipping

// https://en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm
// thread-unsafe: scratch1, scratch2
func sutherland_hodgman_3d(p1, p2, p3 vec4) []vec4 {
	output := append(scratch2[:0], p1, p2, p3)
	for _, plane := range clip_planes {
		copy(scratch1[:], output)       // copy output polygon to our input
		input := scratch1[:len(output)] //
		output = scratch2[:0]           // clear our output polygon
		if len(input) == 0 {
			return nil
		}
		prev_point := input[len(input)-1]
		for _, point := range input {
			if plane.test(point) {
				if !plane.test(prev_point) {
					output = append(output, plane.intersection(prev_point, point))
				}
				output = append(output, point)
			} else if plane.test(prev_point) {
				output = append(output, plane.intersection(prev_point, point))
			}
			prev_point = point
		}
	}
	return output
}

// https://en.wikipedia.org/wiki/Barycentric_coordinate_system
func barycentric(p1, p2, p3, p vec3) vec3 {
	v0 := p2.Sub(p1)
	v1 := p3.Sub(p1)
	v2 := p.Sub(p1)
	d00 := v0.Dot(v0)
	d01 := v0.Dot(v1)
	d11 := v1.Dot(v1)
	d20 := v2.Dot(v0)
	d21 := v2.Dot(v1)
	d := d00*d11 - d01*d01
	v := (d11*d20 - d01*d21) / d
	w := (d00*d21 - d01*d20) / d
	u := 1 - v - w
	return vec3{u, v, w}
}
