#!/usr/bin/env bash

# render shader
./../../bin/dd_shader_reflect -o svis_shader_enums.h -e RE_LineDot \
	-v "./LineDot_V.vert" \
	-f "./LineDot_F.frag"

./../../bin/dd_shader_reflect -o svis_shader_enums.h -a -e RE_Point \
	-v "./PointRend_V.vert" \
	-g "./PointRend_G.geom" \
	-f "./PointRend_F.frag"
