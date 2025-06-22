package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
)

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
