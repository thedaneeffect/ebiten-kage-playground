package main

import (
	"bytes"
	_ "embed"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
	"unsafe"

	"net/http"
	_ "net/http/pprof"

	"github.com/go-gl/mathgl/mgl32"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const (
	game_width  = 800
	game_height = 600
	game_aspect = float(game_width) / float(game_height)
)

type (
	float = float32
	vec2  = mgl32.Vec2
	vec3  = mgl32.Vec3
	vec4  = mgl32.Vec4
	mat4  = mgl32.Mat4
	quat  = mgl32.Quat
)

var shader = `
//kage:unit pixels
package main

func Fragment(dst vec4, src vec2, rgba vec4, custom vec4) vec4 {
	src_origin := imageSrc0Origin()

	// atlas -> texture space
	texel := src - src_origin

	// perspective divide
	if custom.w != 0.0 {
		texel /= custom.w
	}

	// scale uv to pixels
	texel *= imageSrc0Size()

	// move back to atlas space
	texel += src_origin
	
	return imageSrc0At(texel)
}
`

var cpu_profile = flag.String("cpuprofile", "", "write cpu profile to `file`")
var mem_profile = flag.String("memprofile", "", "write memory profile to `file`")

//go:embed plane.obj
var plane_obj []byte

//go:embed topology_mask.png
var topology_mask_png []byte

func main() {
	flag.Parse()

	if *cpu_profile != "" {
		f, err := os.Create(*cpu_profile)
		if err != nil {
			log.Fatal("could not create CPU profile: ", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile()
	}

	if *mem_profile != "" {
		_ = pprof.Lookup("heap")

		defer func() {
			f, err := os.Create(*mem_profile)
			if err != nil {
				log.Fatal("could not create memory profile:", err)
			}
			defer f.Close()
			runtime.GC() // get up-to-date statistics
			if err := pprof.WriteHeapProfile(f); err != nil {
				log.Fatal("could not write memory profile:", err)
			}
		}()
	}

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	shader, err := ebiten.NewShader([]byte(shader))

	if err != nil {
		panic(err)
	}

	const texture_size = 128
	const texture_subdivisions = 32
	const tile_size = texture_size / texture_subdivisions

	texture := ebiten.NewImage(texture_size, texture_size)
	texture.Fill(color.Black)

	for row := range texture_subdivisions {
		for col := range texture_subdivisions {
			if (row+col)%2 == 0 {
				continue
			}
			x := float(col * tile_size)
			y := float(row * tile_size)
			vector.DrawFilledRect(texture, x, y, tile_size, tile_size, color.White, false)
		}
	}

	vector.StrokeRect(texture, 1, 1, texture_size-1, texture_size-1, 1, color.RGBA{255, 0, 0, 255}, false)

	mesh, err := load_obj(plane_obj)
	if err != nil {
		log.Fatal(err)
	}

	for i, p := range mesh.points {
		mesh.points[i] = p.Add(vec3{0, 1 * rand.Float32(), 0})
	}

	game := &game{
		texture:            texture,
		mesh:               mesh,
		topology_radius:    2.5,
		topology_scale:     3,
		topology_curvature: 1.5,
		camera: camera{
			pitch: 0.35,
			yaw:   0,
			pos:   vec3{0, 7, 19},
		},
		context: &context{
			shader: shader,
		},
	}

	img, _, err := image.Decode(bytes.NewReader(topology_mask_png))
	if err != nil {
		log.Fatal(err)
	}

	size := img.Bounds().Max
	i := 0
	for y := range size.Y {
		for x := range size.X {
			if r, _, _, _ := img.At(x, y).RGBA(); r > 0 {
				game.topology_mask[x+y*32] = 1
			}
			i++
		}
	}

	ebiten.SetWindowTitle("004-topology")
	ebiten.SetWindowSize(game_width, game_height)
	ebiten.SetVsyncEnabled(true)

	err = ebiten.RunGameWithOptions(game, &ebiten.RunGameOptions{
		// GraphicsLibrary: ebiten.GraphicsLibraryOpenGL,
	})

	if err != nil {
		panic(err)
	}
}

type game struct {
	context            *context
	cycle              float32
	texture            *ebiten.Image
	mesh               *mesh_t
	topology_mask      [32 * 32]byte
	topology_radius    float64
	topology_curvature float64
	topology_scale     float64
	frametime          time.Duration
	camera             camera
}

type camera struct {
	pitch float
	yaw   float
	pos   vec3

	drag_x   int
	drag_y   int
	dragging bool

	up      vec3
	forward vec3
	right   vec3

	view_matrix mat4
}

type triangle struct {
	v1, v2, v3 uint16
	t1, t2, t3 uint16
}

type mesh_t struct {
	triangles []triangle
	points    []vec3
	uvs       []vec2
}

type vertex struct {
	pos vec4
	uv  vec2
}

func interpolate_vec4(v1, v2, v3 vec4, f vec3) (result vec4) {
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
	result.pos = interpolate_vec4(v1.pos, v2.pos, v3.pos, f)
	result.uv = interpolate_vec2(v1.uv, v2.uv, v3.uv, f)
	return
}

type viewport struct {
	x      int
	y      int
	w      int
	h      int
	w_half int
	h_half int
}

func (g *game) Layout(outerWidth, outerHeight int) (int, int) {
	return game_width, game_height
}

func (g *game) Update() error {
	g.cycle++

	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		cx, cy := ebiten.CursorPosition()

		// doing the logic in the next update ensures we don't get some crazy snapping
		if !g.camera.dragging {
			g.camera.dragging = true
		} else {
			dx := float(cx-g.camera.drag_x) / 100.0
			dy := float(cy-g.camera.drag_y) / 100.0

			g.camera.pitch = mgl32.Clamp(g.camera.pitch+dy, -math.Pi/2, math.Pi/2)
			g.camera.yaw -= dx

			view := mgl32.Ident4()
			view = view.Mul4(mgl32.HomogRotate3DX(g.camera.pitch))
			view = view.Mul4(mgl32.HomogRotate3DY(g.camera.yaw))

			g.camera.right = view.Row(0).Vec3().Mul(-1)
			g.camera.up = view.Row(1).Vec3()
			g.camera.forward = view.Row(2).Vec3().Mul(-1)

			if ebiten.IsKeyPressed(ebiten.KeyW) || ebiten.IsKeyPressed(ebiten.KeyUp) {
				g.camera.pos = g.camera.pos.Add(g.camera.forward.Mul(0.1))
			} else if ebiten.IsKeyPressed(ebiten.KeyS) || ebiten.IsKeyPressed(ebiten.KeyDown) {
				g.camera.pos = g.camera.pos.Sub(g.camera.forward.Mul(0.1))
			}

			if ebiten.IsKeyPressed(ebiten.KeyD) || ebiten.IsKeyPressed(ebiten.KeyRight) {
				g.camera.pos = g.camera.pos.Add(g.camera.right.Mul(0.1))
			} else if ebiten.IsKeyPressed(ebiten.KeyA) || ebiten.IsKeyPressed(ebiten.KeyLeft) {
				g.camera.pos = g.camera.pos.Sub(g.camera.right.Mul(0.1))
			}

			g.camera.view_matrix = view.Mul4(mgl32.Translate3D(
				-g.camera.pos.X(),
				-g.camera.pos.Y(),
				-g.camera.pos.Z(),
			))
		}

		g.camera.drag_x = cx
		g.camera.drag_y = cy
	} else {
		g.camera.dragging = false
	}

	return nil
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

type context struct {
	shader      *ebiten.Shader
	view_matrix mat4
	proj_matrix mat4
	// viewport is used to convert normalized device coordinates to screen coordinates
	viewport viewport

	// statistics
	drawn_triangles int

	// the following are not required to be stored here,
	// they serve as buffers to reduce overall allocations.

	triangle_buffer   []screen_triangle
	clip_space_points []vec4
	vertices          []ebiten.Vertex
	indices           []uint16

	use_cpu        bool
	cpu_buffer     *ebiten.Image
	cpu_pixels     []uint32
	cpu_pixels_raw []byte
}

func init() {

}

type screen_triangle struct {
	v1, v2, v3 vertex
	data       uint64
}

func (c *context) set_viewport(x, y, w, h int) {
	c.viewport.x = x
	c.viewport.y = y
	c.viewport.w = w
	c.viewport.h = h
	c.viewport.w_half = w / 2
	c.viewport.h_half = h / 2
}

func out_of_bounds(a vec4) bool {
	x, y, z, w := a.X(), a.Y(), a.Z(), a.W()
	return x < -w || x > w || y < -w || y > w || z < -w || z > w
}

func viewport_transform(ndc, dimension_half float) float {
	return dimension_half*ndc + dimension_half
}

func (ctx *context) push_mesh(mesh *mesh_t, position vec3, orientation quat, mesh_data uint64) int {
	// TODO: determine if mesh is culled

	// save us some calculations by doing this here instead of per point
	projection_view_matrix := ctx.proj_matrix.Mul4(
		ctx.view_matrix.Mul4(
			orientation.Mat4().Mul4(
				mgl32.Translate3D(
					position.X(),
					position.Y(),
					position.Z(),
				),
			),
		),
	)

	// transform all the mesh points into clip space
	for _, point := range mesh.points {
		// TODO: perform local transformations here (skeleton?)

		point := projection_view_matrix.Mul4x1(point.Vec4(1))
		ctx.clip_space_points = append(ctx.clip_space_points, point)
	}

	for _, triangle := range mesh.triangles {
		v1 := vertex{
			pos: ctx.clip_space_points[triangle.v1],
			uv:  mesh.uvs[triangle.t1],
		}
		v2 := vertex{
			pos: ctx.clip_space_points[triangle.v2],
			uv:  mesh.uvs[triangle.t2],
		}
		v3 := vertex{
			pos: ctx.clip_space_points[triangle.v3],
			uv:  mesh.uvs[triangle.t3],
		}

		if out_of_bounds(v1.pos) || out_of_bounds(v2.pos) || out_of_bounds(v3.pos) {
			points := sutherland_hodgman_3d(v1.pos, v2.pos, v3.pos)

			p1 := v1.pos.Vec3()
			p2 := v2.pos.Vec3()
			p3 := v3.pos.Vec3()

			for i := 2; i < len(points); i++ {
				b1 := barycentric(p1, p2, p3, points[0].Vec3())
				b2 := barycentric(p1, p2, p3, points[i-1].Vec3())
				b3 := barycentric(p1, p2, p3, points[i].Vec3())

				ctx.triangle_buffer = append(ctx.triangle_buffer, screen_triangle{
					v1:   interpolate_vertex(v1, v2, v3, b1),
					v2:   interpolate_vertex(v1, v2, v3, b2),
					v3:   interpolate_vertex(v1, v2, v3, b3),
					data: mesh_data,
				})
			}
		} else {
			ctx.triangle_buffer = append(ctx.triangle_buffer, screen_triangle{
				v1:   v1,
				v2:   v2,
				v3:   v3,
				data: mesh_data,
			})
		}

		w_half := float(ctx.viewport.w_half)
		h_half := float(ctx.viewport.h_half)

		var r1 ebiten.Vertex
		var r2 ebiten.Vertex
		var r3 ebiten.Vertex

		for _, t := range ctx.triangle_buffer {
			v1 := t.v1
			v2 := t.v2
			v3 := t.v3

			// perspective divide (clip -> ndc)
			inv_w1 := 1.0 / v1.pos.W()
			inv_w2 := 1.0 / v2.pos.W()
			inv_w3 := 1.0 / v3.pos.W()

			r1.DstX = v1.pos.X() * inv_w1
			r1.DstY = v1.pos.Y() * inv_w1
			r2.DstX = v2.pos.X() * inv_w2
			r2.DstY = v2.pos.Y() * inv_w2
			r3.DstX = v3.pos.X() * inv_w3
			r3.DstY = v3.pos.Y() * inv_w3

			// 2d cross product
			dx12 := (r2.DstX - r1.DstX)
			dy12 := (r2.DstY - r1.DstY)
			dx13 := (r3.DstX - r1.DstX)
			dy13 := (r3.DstY - r1.DstY)

			// back-face culling
			if dx12*dy13-dx13*dy12 <= 0 {
				continue
			}

			// ndc to screen space
			r1.DstX = viewport_transform(r1.DstX, w_half)
			r1.DstY = viewport_transform(r1.DstY, h_half)
			r2.DstX = viewport_transform(r2.DstX, w_half)
			r2.DstY = viewport_transform(r2.DstY, h_half)
			r3.DstX = viewport_transform(r3.DstX, w_half)
			r3.DstY = viewport_transform(r3.DstY, h_half)

			// perspective correction
			if true {
				r1.SrcX = v1.uv.X() * inv_w1
				r1.SrcY = v1.uv.Y() * inv_w1
				r1.Custom3 = inv_w1

				r2.SrcX = v2.uv.X() * inv_w2
				r2.SrcY = v2.uv.Y() * inv_w2
				r2.Custom3 = inv_w2

				r3.SrcX = v3.uv.X() * inv_w3
				r3.SrcY = v3.uv.Y() * inv_w3
				r3.Custom3 = inv_w3
			} else {
				r1.SrcX = v1.uv.X()
				r1.SrcY = v1.uv.Y()
				r2.SrcX = v2.uv.X()
				r2.SrcY = v2.uv.Y()
				r3.SrcX = v3.uv.X()
				r3.SrcY = v3.uv.Y()
			}

			ctx.vertices = append(ctx.vertices, r1, r2, r3)

			// gpu needs an index buffer
			if !ctx.use_cpu {
				first_index := uint16(len(ctx.indices))
				ctx.indices = append(ctx.indices, first_index, first_index+1, first_index+2)
			}
		}

		ctx.triangle_buffer = ctx.triangle_buffer[:0]
	}

	ctx.clip_space_points = ctx.clip_space_points[:0]

	return 0
}

func (ctx *context) draw(texture, target *ebiten.Image) {
	if ctx.use_cpu {
		ctx.draw_cpu(texture, target)
	} else {
		ctx.draw_gpu(texture, target)
	}
}

func (ctx *context) draw_cpu(texture, target *ebiten.Image) {
	dst_bounds := target.Bounds()
	dst_size := dst_bounds.Dx() * dst_bounds.Dy()
	if ctx.cpu_pixels == nil || len(ctx.cpu_pixels) != dst_size {
		ctx.cpu_pixels = make([]uint32, dst_bounds.Dx()*dst_bounds.Dy())
		ctx.cpu_pixels_raw = unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(ctx.cpu_pixels))), len(ctx.cpu_pixels)*4)
	}

	target.WritePixels(ctx.cpu_pixels_raw)
}

func (ctx *context) draw_gpu(texture, target *ebiten.Image) {
	target.DrawTrianglesShader(ctx.vertices, ctx.indices, ctx.shader, &ebiten.DrawTrianglesShaderOptions{
		Images: [4]*ebiten.Image{
			texture,
		},
		Uniforms: map[string]any{
			"HoverID": 0,
		},
		AntiAlias: false,
	})

	ctx.drawn_triangles = len(ctx.indices) / 3
	ctx.vertices = ctx.vertices[:0]
	ctx.indices = ctx.indices[:0]
}

func (g *game) Draw(screen *ebiten.Image) {
	defer func(t time.Time) {
		ft := time.Now().Sub(t)
		if g.frametime == 0 {
			g.frametime = ft
		} else {
			g.frametime += (ft - g.frametime) / 2
		}
	}(time.Now())

	ctx := g.context

	w := screen.Bounds().Dx()
	h := screen.Bounds().Dy()

	ctx.set_viewport(0, 0, w, h)

	// If you use orthographic then the Z axis will invert for everything.
	// https://www.songho.ca/opengl/gl_projectionmatrix.html#perspective
	// ctx.set_orthographic(-eye_distance*game_aspect, eye_distance*game_aspect, eye_distance, -eye_distance, 0.1, 10)

	ctx.proj_matrix = mgl32.Perspective(30, game_aspect, 0.1, 100)

	// the camera view matrix is invalid until the user controls it
	if g.camera.view_matrix.Det() == 0 {
		ctx.view_matrix = mgl32.LookAtV(
			vec3{0, 7, 19},
			vec3{0, 0, 0},
			vec3{0, 1, 0},
		)
	} else {
		ctx.view_matrix = g.camera.view_matrix
	}

	screen.Fill(color.RGBA{130, 130, 130, 255})

	// position := vec3{0, 0, 0}

	orientation := mgl32.QuatIdent()
	angle := g.cycle / 60 / math.Pi
	orientation = orientation.Mul(mgl32.QuatRotate(angle, vec3{0, 1, 0}))

	// hover_id := ctx.push_mesh(g.mesh, position, orientation, 0)

	ctx.draw(g.texture, screen)

	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("TPS: %.0f", ebiten.ActualTPS()), 0, 0)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("FPS: %.0f CPU: %v", ebiten.ActualFPS(), g.context.use_cpu), 0, 14)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Ft: %v", g.frametime), 0, 28)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Mem: %d", mem.Alloc/1024), 0, 42)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Triangles: %d", ctx.drawn_triangles), 0, 56)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Eye: %.2f, %.2f", g.camera.pitch, g.camera.yaw), 0, 70)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Cam: %v", g.camera.pos), 0, 84)
	// ebitenutil.DebugPrintAt(screen, fmt.Sprintf("HoverID: %v", hover_id), 0, 98)
}

func load_obj(src []byte) (*mesh_t, error) {
	reader := bytes.NewReader(src)
	mesh := &mesh_t{}
	for {
		var typ string
		if _, err := fmt.Fscan(reader, &typ); err != nil {
			if errors.Is(io.EOF, err) {
				break
			}
			return nil, fmt.Errorf("bad type: %w", err)
		}
		switch typ {
		default:
			return nil, fmt.Errorf("unknown type: %s", typ)
		case "#", "o", "s", "l":
			fmt.Fscanln(reader)
		case "v":
			var x, y, z float
			if _, err := fmt.Fscanf(reader, "%f %f %f", &x, &y, &z); err != nil {
				return nil, fmt.Errorf("bad vertex: %w", err)
			}
			mesh.points = append(mesh.points, vec3{x, y, z})
		case "vt":
			var s, t float
			if _, err := fmt.Fscanf(reader, "%f %f", &s, &t); err != nil {
				return nil, fmt.Errorf("bad texcoord: %w", err)
			}
			mesh.uvs = append(mesh.uvs, vec2{s, t})
		case "f":
			var v1, v2, v3 uint16
			var t1, t2, t3 uint16
			if _, err := fmt.Fscanf(reader, "%d/%d %d/%d %d/%d", &v1, &t1, &v2, &t2, &v3, &t3); err != nil {
				return nil, fmt.Errorf("bad face: %w", err)
			}
			mesh.triangles = append(mesh.triangles, triangle{
				v1: v1 - 1,
				v2: v2 - 1,
				v3: v3 - 1,
				t1: t1 - 1,
				t2: t2 - 1,
				t3: t3 - 1,
			})
		}
	}
	return mesh, nil
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
