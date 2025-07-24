package main

import (
	"bytes"
	_ "embed"
	"flag"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
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

	mgl "github.com/go-gl/mathgl/mgl32"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const (
	game_width  = 1024
	game_height = 1024
	game_aspect = float(game_width) / float(game_height)
)

type (
	float = float32
	vec2  = mgl.Vec2
	vec3  = mgl.Vec3
	vec4  = mgl.Vec4
	mat4  = mgl.Mat4
	quat  = mgl.Quat
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
		rgba /= custom.w
	}

	// scale uv to pixels
	texel *= imageSrc0Size()

	// move back to atlas space
	texel += src_origin
	
	return imageSrc0At(texel) * rgba
}
`

var cpu_profile = flag.String("cpuprofile", "", "write cpu profile to `file`")
var mem_profile = flag.String("memprofile", "", "write memory profile to `file`")

//go:embed plane3.obj
var plane_obj []byte

//go:embed topology_mask.png
var topology_mask_png []byte

const texture_size = 128

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

	const texture_subdivisions = 8
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
			vector.DrawFilledRect(texture, float32(x), float32(y), tile_size, tile_size, color.White, false)
		}
	}

	vector.StrokeRect(texture, 1, 1, texture_size-1, texture_size-1, 1, color.RGBA{255, 0, 0, 255}, false)

	mesh, err := load_obj(plane_obj)
	if err != nil {
		log.Fatal(err)
	}

	for i, p := range mesh.points {
		mesh.points[i] = p.Add(vec3{0, float(3 * rand.Float32()), 0}).Mul(3)
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
			pos:   vec3{0, 32, 64},
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
	move_speed         float
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
	pos  vec4
	rgba vec4
	uv   vec2
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
	if g.cycle == 0 {

		g.context.cpu.texture = &cpu_texture{
			texels: make([]uint32, texture_size*texture_size),
			width:  texture_size,
			height: texture_size,
		}

		g.texture.ReadPixels(
			unsafe.Slice(
				(*byte)(unsafe.Pointer(
					&g.context.cpu.texture.texels[0],
				)),
				len(g.context.cpu.texture.texels)*4,
			),
		)
	}
	g.cycle++

	if inpututil.IsKeyJustPressed(ebiten.KeySpace) {
		g.context.use_cpu = !g.context.use_cpu
	}

	if _, yoff := ebiten.Wheel(); yoff != 0 {
		g.move_speed += float(yoff) / 10.0
	}

	if g.move_speed < 0.1 {
		g.move_speed = 0.1
	}

	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		cx, cy := ebiten.CursorPosition()

		// doing the logic in the next update ensures we don't get some crazy snapping
		if !g.camera.dragging {
			g.camera.dragging = true
		} else {
			dx := float(cx-g.camera.drag_x) / 100.0
			dy := float(cy-g.camera.drag_y) / 100.0

			g.camera.pitch = mgl.Clamp(g.camera.pitch+dy, -math.Pi/2, math.Pi/2)
			g.camera.yaw += dx

			view := mgl.Ident4()
			view = view.Mul4(mgl.HomogRotate3DX(g.camera.pitch))
			view = view.Mul4(mgl.HomogRotate3DY(g.camera.yaw))

			g.camera.right = view.Row(0).Vec3()
			g.camera.up = view.Row(1).Vec3()
			g.camera.forward = view.Row(2).Vec3().Mul(-1)

			if ebiten.IsKeyPressed(ebiten.KeyW) || ebiten.IsKeyPressed(ebiten.KeyUp) {
				g.camera.pos = g.camera.pos.Add(g.camera.forward.Mul(g.move_speed))
			} else if ebiten.IsKeyPressed(ebiten.KeyS) || ebiten.IsKeyPressed(ebiten.KeyDown) {
				g.camera.pos = g.camera.pos.Sub(g.camera.forward.Mul(g.move_speed))
			}

			if ebiten.IsKeyPressed(ebiten.KeyD) || ebiten.IsKeyPressed(ebiten.KeyRight) {
				g.camera.pos = g.camera.pos.Add(g.camera.right.Mul(g.move_speed))
			} else if ebiten.IsKeyPressed(ebiten.KeyA) || ebiten.IsKeyPressed(ebiten.KeyLeft) {
				g.camera.pos = g.camera.pos.Sub(g.camera.right.Mul(g.move_speed))
			}

			g.camera.view_matrix = view.Mul4(mgl.Translate3D(
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

	use_cpu bool
	cpu     cpu_context
}

type cpu_texture struct {
	texels []uint32
	width  int
	height int
}

type cpu_context struct {
	texture     *cpu_texture
	buffer      *ebiten.Image
	pixels      []uint32
	depth       []float
	pixels_raw  []byte
	width       int
	height      int
	left, right int
	top, bottom int
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

	model := orientation.Mat4().Mul4(
		mgl.Translate3D(
			position.X(),
			position.Y(),
			position.Z(),
		),
	)

	// save us some calculations by doing this here instead of per point
	project := ctx.proj_matrix
	model_view_project := project.Mul4(ctx.view_matrix).Mul4(model)

	// transform all the mesh points into clip space
	for _, point := range mesh.points {
		// TODO: perform local transformations here (skeleton?)

		point := model_view_project.Mul4x1(point.Vec4(1))
		ctx.clip_space_points = append(ctx.clip_space_points, point)
	}

	for _, triangle := range mesh.triangles {
		v1 := vertex{
			pos:  ctx.clip_space_points[triangle.v1],
			rgba: vec4{1, 0, 0, 1},
			uv:   mesh.uvs[triangle.t1],
		}
		v2 := vertex{
			pos:  ctx.clip_space_points[triangle.v2],
			rgba: vec4{0, 1, 0, 1},
			uv:   mesh.uvs[triangle.t2],
		}
		v3 := vertex{
			pos:  ctx.clip_space_points[triangle.v3],
			rgba: vec4{0, 0, 1, 1},
			uv:   mesh.uvs[triangle.t3],
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
		h := float32(ctx.viewport.h)

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

			r1.DstX = float32(v1.pos.X() * inv_w1)
			r1.DstY = float32(v1.pos.Y() * inv_w1)

			r2.DstX = float32(v2.pos.X() * inv_w2)
			r2.DstY = float32(v2.pos.Y() * inv_w2)

			r3.DstX = float32(v3.pos.X() * inv_w3)
			r3.DstY = float32(v3.pos.Y() * inv_w3)

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
			r1.DstX = float32(viewport_transform(float(r1.DstX), w_half))
			r1.DstY = h - float32(viewport_transform(float(r1.DstY), h_half))
			r1.ColorR = float32(t.v1.rgba.X() * inv_w1)
			r1.ColorG = float32(t.v1.rgba.Y() * inv_w1)
			r1.ColorB = float32(t.v1.rgba.Z() * inv_w1)
			r1.ColorA = float32(t.v1.rgba.W() * inv_w1)
			r1.Custom3 = float32(inv_w1)

			r2.DstX = float32(viewport_transform(float(r2.DstX), w_half))
			r2.DstY = h - float32(viewport_transform(float(r2.DstY), h_half))
			r2.ColorR = float32(t.v2.rgba.X() * inv_w2)
			r2.ColorG = float32(t.v2.rgba.Y() * inv_w2)
			r2.ColorB = float32(t.v2.rgba.Z() * inv_w2)
			r2.ColorA = float32(t.v2.rgba.W() * inv_w2)
			r2.Custom3 = float32(inv_w2)

			r3.DstX = float32(viewport_transform(float(r3.DstX), w_half))
			r3.DstY = h - float32(viewport_transform(float(r3.DstY), h_half))
			r3.ColorR = float32(t.v3.rgba.X() * inv_w3)
			r3.ColorG = float32(t.v3.rgba.Y() * inv_w3)
			r3.ColorB = float32(t.v3.rgba.Z() * inv_w3)
			r3.ColorA = float32(t.v3.rgba.W() * inv_w3)
			r3.Custom3 = float32(inv_w3)

			// perspective correction
			if true {
				r1.SrcX = float32(v1.uv.X() * inv_w1)
				r1.SrcY = float32(v1.uv.Y() * inv_w1)
				r2.SrcX = float32(v2.uv.X() * inv_w2)
				r2.SrcY = float32(v2.uv.Y() * inv_w2)
				r3.SrcX = float32(v3.uv.X() * inv_w3)
				r3.SrcY = float32(v3.uv.Y() * inv_w3)
			} else {
				r1.SrcX = float32(v1.uv.X())
				r1.SrcY = float32(v1.uv.Y())
				r2.SrcX = float32(v2.uv.X())
				r2.SrcY = float32(v2.uv.Y())
				r3.SrcX = float32(v3.uv.X())
				r3.SrcY = float32(v3.uv.Y())
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
		ctx.cpu.draw(ctx.vertices, texture, target)
	} else {
		ctx.draw_gpu(texture, target)
	}
	ctx.drawn_triangles = len(ctx.vertices) / 3
	ctx.vertices = ctx.vertices[:0]
	ctx.indices = ctx.indices[:0]
}

func (ctx *context) draw_gpu(texture, target *ebiten.Image) {
	target.DrawTrianglesShader(ctx.vertices, ctx.indices, ctx.shader, &ebiten.DrawTrianglesShaderOptions{
		Images: [4]*ebiten.Image{
			texture,
		},
		AntiAlias: false,
	})
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

	ctx.proj_matrix = mgl.Perspective(math.Pi/2, game_aspect, 1, 1000)

	// the camera view matrix is invalid until the user controls it
	ctx.view_matrix = g.camera.view_matrix

	screen.Fill(color.RGBA{130, 130, 130, 255})

	position := vec3{0, 0, 0}

	orientation := mgl.QuatIdent()
	// angle := g.cycle / 60 / math.Pi
	// orientation = orientation.Mul(mgl32.QuatRotate(angle, vec3{0, 1, 0}))

	hover_id := ctx.push_mesh(g.mesh, position, orientation, 0)

	ctx.draw(g.texture, screen)

	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("TPS: %.0f", ebiten.ActualTPS()), 0, 0)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("FPS: %.0f CPU: %v", ebiten.ActualFPS(), g.context.use_cpu), 0, 14)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Ft: %v", g.frametime), 0, 28)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Mem: %dkB", mem.Alloc/1024), 0, 42)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Triangles: %d", ctx.drawn_triangles), 0, 56)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Eye: %.2f, %.2f", g.camera.pitch, g.camera.yaw), 0, 70)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Cam: %v", g.camera.pos), 0, 84)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("HoverID: %v", hover_id), 0, 98)
}
