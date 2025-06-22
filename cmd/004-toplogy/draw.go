package main

import (
	"unsafe"

	"github.com/hajimehoshi/ebiten/v2"
)

func (ctx *cpu_context) draw(vertices []ebiten.Vertex, texture, target *ebiten.Image) {
	dst_bounds := target.Bounds()
	dst_size := dst_bounds.Dx() * dst_bounds.Dy()
	if ctx.pixels == nil || len(ctx.pixels) != dst_size {
		ctx.buffer = ebiten.NewImageWithOptions(dst_bounds, &ebiten.NewImageOptions{
			Unmanaged: true,
		})
		ctx.pixels = make([]uint32, dst_size)
		ctx.depth = make([]float32, dst_size)
		ctx.pixels_raw = unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(ctx.pixels))), len(ctx.pixels)*4)
	}

	ctx.width = dst_bounds.Dx()
	ctx.height = dst_bounds.Dy()
	ctx.left = 0
	ctx.right = dst_bounds.Dx() - 1
	ctx.top = 0
	ctx.bottom = dst_bounds.Dy() - 1

	clear(ctx.pixels)
	clear(ctx.depth)

	for i := 0; i < len(vertices); i += 3 {
		a := vertices[i]
		b := vertices[i+1]
		c := vertices[i+2]

		ctx.fill_triangle(
			int(a.DstX), int(a.DstY),
			int(b.DstX), int(b.DstY),
			int(c.DstX), int(c.DstY),
			a, b, c,
		)
	}

	ctx.buffer.WritePixels(ctx.pixels_raw)
	target.DrawImage(ctx.buffer, nil)
}

func (ctx *cpu_context) fill_triangle(x0, y0, x1, y1, x2, y2 int, a, b, c ebiten.Vertex) {
	var a_u, a_v, a_w float32
	var b_u, b_v, v_w float32
	var c_u, c_v, c_w float32

	a_u = 1.0
	b_v = 1.0
	c_w = 1.0

	if y0 > y2 {
		x0, x2 = x2, x0
		y0, y2 = y2, y0
		a_u, c_u = c_u, a_u
		a_v, c_v = c_v, a_v
		a_w, c_w = c_w, a_w
	}

	if y0 > y1 {
		x0, x1 = x1, x0
		y0, y1 = y1, y0
		a_u, b_u = b_u, a_u
		a_v, b_v = b_v, a_v
		a_w, v_w = v_w, a_w
	}

	if y1 > y2 {
		y1, y2 = y2, y1
		x1, x2 = x2, x1
		b_u, c_u = c_u, b_u
		b_v, c_v = c_v, b_v
		v_w, c_w = c_w, v_w
	}

	if y0 >= ctx.bottom {
		return
	} else if y2 < ctx.top {
		return
	}

	var (
		x_step_ab int
		x_step_bc int
		x_step_ac int
		u_step_ab float32
		u_step_bc float32
		u_step_ac float32
		v_step_ab float32
		v_step_bc float32
		v_step_ac float32
		w_step_ab float32
		w_step_bc float32
		w_step_ac float32
	)

	if d := y1 - y0; d != 0 {
		x_step_ab = ((x1 - x0) << 16) / d
		u_step_ab = (b_u - a_u) / float32(d)
		v_step_ab = (b_v - a_v) / float32(d)
		w_step_ab = (v_w - a_w) / float32(d)
	}

	if d := y2 - y1; d != 0 {
		x_step_bc = ((x2 - x1) << 16) / d
		u_step_bc = (c_u - b_u) / float32(d)
		v_step_bc = (c_v - b_v) / float32(d)
		w_step_bc = (c_w - v_w) / float32(d)
	}

	if d := y2 - y0; d != 0 {
		x_step_ac = ((x2 - x0) << 16) / d
		u_step_ac = (c_u - a_u) / float32(d)
		v_step_ac = (c_v - a_v) / float32(d)
		w_step_ac = (c_w - a_w) / float32(d)
	}

	x0 <<= 16
	x2 = x0
	c_u = a_u
	c_v = a_v
	c_w = a_w
	x1 <<= 16

	if trim := ctx.top - y0; trim > 0 {
		x0 += x_step_ab * trim
		a_u += u_step_ab * float32(trim)
		a_v += v_step_ab * float32(trim)
		a_w += w_step_ab * float32(trim)
		x2 += x_step_ac * trim
		c_u += u_step_ac * float32(trim)
		c_v += v_step_ac * float32(trim)
		c_w += w_step_ac * float32(trim)
		y0 += trim
	}

	if trim := ctx.top - y1; trim > 0 {
		x1 += x_step_bc * trim
		b_u += u_step_bc * float32(trim)
		b_v += v_step_bc * float32(trim)
		v_w += w_step_bc * float32(trim)
		y1 += trim
	}

	if y1 > ctx.bottom {
		y1 = ctx.bottom
	}

	if y2 > ctx.bottom {
		y2 = ctx.bottom
	}

	offset := y0 * ctx.width

	for h := y1 - y0; h > 0; h-- {
		ctx.draw_scanline(offset, x0>>16, x2>>16, a_u, c_u, a_v, c_v, a_w, c_w, a, b, c)
		x0 += x_step_ab
		a_u += u_step_ab
		a_v += v_step_ab
		a_w += w_step_ab
		x2 += x_step_ac
		c_u += u_step_ac
		c_v += v_step_ac
		c_w += w_step_ac
		offset += ctx.width
	}

	for h := y2 - y1; h > 0; h-- {
		ctx.draw_scanline(offset, x1>>16, x2>>16, b_u, c_u, b_v, c_v, v_w, c_w, a, b, c)
		x1 += x_step_bc
		b_u += u_step_bc
		b_v += v_step_bc
		v_w += w_step_bc
		x2 += x_step_ac
		c_u += u_step_ac
		c_v += v_step_ac
		c_w += w_step_ac
		offset += ctx.width
	}
}

func (ctx *cpu_context) draw_scanline(offset, x0, x1 int, u0, u1, v0, v1, w0, w1 float, a, b, c ebiten.Vertex) {
	if x0 == x1 {
		return
	}

	if x0 > x1 {
		x0, x1 = x1, x0
		u0, u1 = u1, u0
		v0, v1 = v1, v0
		w0, w1 = w1, w0
	}

	length := x1 - x0
	u_step := (u1 - u0) / float(length)
	v_step := (v1 - v0) / float(length)
	w_step := (w1 - w0) / float(length)

	if trim := ctx.left - x0; trim > 0 {
		u0 += u_step * float(trim)
		v0 += v_step * float(trim)
		w0 += w_step * float(trim)
		x0 = ctx.left
	}

	if x1 >= ctx.right {
		x1 = ctx.right - 1
	}

	if length = x1 - x0; length > 0 {
		offset += x0

		for x := 0; x < length; x++ {
			depth := (u0 * a.Custom3) + (v0 * b.Custom3) + (w0 * c.Custom3)

			if ctx.depth[offset] < depth {
				inv_depth := 1.0 / depth

				u := (u0*a.SrcX + v0*b.SrcX + w0*c.SrcX) * inv_depth
				v := (u0*a.SrcY + v0*b.SrcY + w0*c.SrcY) * inv_depth

				u = min(1, max(0, u))
				v = min(1, max(0, v))

				real_u := min(ctx.texture.width-1, int(u*float32(ctx.texture.width)))
				real_v := min(ctx.texture.height-1, int(v*float32(ctx.texture.height)))

				abgr := ctx.texture.texels[real_u+(real_v*ctx.texture.width)]

				r := float32((abgr>>0)&0xFF) * (u0*a.ColorR + v0*b.ColorR + w0*c.ColorR)
				g := float32((abgr>>8)&0xFF) * (u0*a.ColorG + v0*b.ColorG + w0*c.ColorG)
				b := float32((abgr>>16)&0xFF) * (u0*a.ColorB + v0*b.ColorB + w0*c.ColorB)

				ctx.pixels[offset] = (0xFF << 24) | uint32(b)<<16 | uint32(g)<<8 | uint32(r)
				ctx.depth[offset] = depth
			}
			u0 += u_step
			v0 += v_step
			w0 += w_step
			offset++
		}
	}
}
